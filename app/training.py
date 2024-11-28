import pandas as pd
from app.schemas import Prediction, SimplePrediction
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging
from typing import List, Tuple, Dict, Optional, Any
import boto3
import random  
import os
from datetime import date, datetime
import pickle
import json
from .utils import ModelRegistry, setup_logging

setup_logging()

class ModelTrainer:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet training"""
        try:
            logging.info("=== Initial Data Inspection ===")
            logging.info(f"DataFrame head:\n{df.head().to_string()}")
            
            # Convert to Prophet expected format
            prophet_df = df.rename(columns={
                'Date': 'ds',
                'Quantity': 'y'
            })
            
            # Ensure datetime format
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Create complete date range and fill missing values with 0
            date_range = pd.date_range(
                start=prophet_df['ds'].min(),
                end=prophet_df['ds'].max(),
                freq='D'
            )
            
            # Reindex and fill missing values
            prophet_df = prophet_df.set_index('ds')['y'].reindex(date_range, fill_value=0).reset_index()
            prophet_df = prophet_df.rename(columns={'index': 'ds'})
            
            # Calculate rolling statistics
            prophet_df['y_ma7'] = prophet_df['y'].rolling(window=7, min_periods=1).mean()
            prophet_df['y_ma30'] = prophet_df['y'].rolling(window=30, min_periods=1).mean()
            
            # Add capacity for logistic growth
            max_qty = prophet_df['y'].max()
            prophet_df['cap'] = max_qty * 2  # Set capacity to 2x max historical value
            prophet_df['floor'] = 0  # Set minimum to 0
            
            logging.info("\n=== Processed Data Statistics ===")
            logging.info(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
            logging.info(f"Total days: {len(prophet_df)}")
            logging.info(f"Days with sales: {(prophet_df['y'] > 0).sum()}")
            logging.info(f"Days without sales: {(prophet_df['y'] == 0).sum()}")
            logging.info(f"\nQuantity statistics:")
            logging.info(f"Min (non-zero): {prophet_df[prophet_df['y'] > 0]['y'].min()}")
            logging.info(f"Max: {prophet_df['y'].max()}")
            logging.info(f"Mean (overall): {prophet_df['y'].mean()}")
            logging.info(f"Mean (non-zero): {prophet_df[prophet_df['y'] > 0]['y'].mean()}")
            
            return prophet_df
            
        except Exception as e:
            logging.error(f"Error in prepare_training_data: {str(e)}")
            raise
    
    def train_model(self, sku: str, data: pd.DataFrame, custom_parameters: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Prophet], Optional[Dict]]:
        """Train Prophet model with given data and optional parameters"""
        try:
            logging.info(f"\n=== Starting Model Training for SKU: {sku} ===")
            
            # Prepare data with zero filling
            prophet_data = self.prepare_training_data(data)
            
            # Calculate data characteristics
            sparsity = (prophet_data['y'] == 0).mean()
            non_zero_mean = prophet_data[prophet_data['y'] > 0]['y'].mean()
            non_zero_std = prophet_data[prophet_data['y'] > 0]['y'].std()
            overall_mean = prophet_data['y'].mean()
            
            logging.info(f"Data characteristics before transformation:")
            logging.info(f"Sparsity: {sparsity:.2f}")
            logging.info(f"Non-zero mean: {non_zero_mean:.2f}")
            logging.info(f"Non-zero std: {non_zero_std:.2f}")
            logging.info(f"Overall mean: {overall_mean:.2f}")
            
            # Transform data for better handling of sparsity
            if sparsity > 0.7:  # If very sparse
                # Log transform with offset to handle zeros
                min_non_zero = prophet_data[prophet_data['y'] > 0]['y'].min()
                prophet_data['y'] = np.log1p(prophet_data['y'] + min_non_zero/2)
                max_y = prophet_data['y'].max()
                
                # Add capacity as 150% of max observed value
                prophet_data['cap'] = max_y * 1.5
                prophet_data['floor'] = 0
                
                logging.info(f"Applied log transformation and set capacity to {prophet_data['cap'].max():.2f}")
            else:
                # For less sparse data, just use regular logistic growth
                max_y = prophet_data['y'].max()
                prophet_data['cap'] = max_y * 2
                prophet_data['floor'] = 0
            
            # Set default parameters
            default_params = {
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 0.1,
                'holidays_prior_scale': 0.1,
                'n_changepoints': min(25, len(prophet_data) // 4),
                'changepoint_range': 0.9,
                'weekly_seasonality': True,
                'yearly_seasonality': len(prophet_data) >= 365,
                'daily_seasonality': False,
                'growth': 'logistic',
                'seasonality_mode': 'additive',
                'interval_width': 0.95
            }
            
            # Update with custom parameters
            if custom_parameters:
                default_params.update(custom_parameters)
                
            logging.info(f"Training with parameters: {default_params}")
            
            # Initialize model
            model = Prophet(**default_params)
            
            # Add regressors
            model.add_regressor('y_ma7', standardize=True)
            model.add_regressor('y_ma30', standardize=True)
            
            # Fit model
            model.fit(prophet_data)
            
            # Make predictions
            future = model.make_future_dataframe(periods=30)
            future['cap'] = prophet_data['cap'].max()
            future['floor'] = 0
            future['y_ma7'] = prophet_data['y_ma7'].iloc[-1]
            future['y_ma30'] = prophet_data['y_ma30'].iloc[-1]
            
            forecast = model.predict(future)
            
            # Transform predictions back if we used log transform
            if sparsity > 0.7:
                forecast['yhat'] = np.expm1(forecast['yhat']) - min_non_zero/2
                forecast['yhat_lower'] = np.expm1(forecast['yhat_lower']) - min_non_zero/2
                forecast['yhat_upper'] = np.expm1(forecast['yhat_upper']) - min_non_zero/2
                
                # Ensure no negative values
                forecast['yhat'] = forecast['yhat'].clip(lower=0)
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
                forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
            
            # Calculate metrics
            metrics = self.calculate_metrics(prophet_data, forecast)
            metrics['parameters'] = default_params
            metrics['sparsity'] = sparsity
            
            logging.info(f"\nFinal Forecast Statistics:")
            logging.info(f"Mean prediction: {forecast['yhat'].mean():.2f}")
            logging.info(f"Min prediction: {forecast['yhat'].min():.2f}")
            logging.info(f"Max prediction: {forecast['yhat'].max():.2f}")
            
            return model, metrics
            
        except Exception as e:
            logging.error(f"Error training model for SKU {sku}: {e}")
            raise
        
    def validate_model(self, model: Prophet, data: pd.DataFrame) -> Dict:
        """Perform cross-validation on the model"""
        try:
            # Initial validation on training set
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            
            # Calculate metrics
            metrics = self.calculate_metrics(data, forecast)
            
            # Perform cross validation
            if len(data) >= 90:  # Only if we have enough data
                cv_results = cross_validation(
                    model,
                    initial='60 days',
                    period='30 days',
                    horizon='30 days',
                    parallel="processes"
                )
                cv_metrics = performance_metrics(cv_results)
                metrics['cv_mae'] = cv_metrics['mae'].mean()
                metrics['cv_mape'] = cv_metrics['mape'].mean()
                
            return metrics
            
        except Exception as e:
            logging.error(f"Error in model validation: {str(e)}")
            return {}

    def optimize_hyperparameters(self, df: pd.DataFrame) -> Dict:
        """Find optimal hyperparameters for Prophet model with improved error handling and data validation"""
        logging.info("Starting hyperparameter optimization...")
        
        # Validate input data
        if len(df) < 30:  # Minimum required data points
            logging.warning("Insufficient data for optimization, using default parameters")
            return {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'n_changepoints': min(25, max(1, len(df) // 4)),  # Scale with data size
                'changepoint_range': 0.8
            }
        
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'n_changepoints': [
                min(15, max(1, len(df) // 6)),
                min(25, max(1, len(df) // 4)),
                min(35, max(1, len(df) // 3))
            ],
            'changepoint_range': [0.8, 0.9],
        }
        
        best_params = {}
        best_mape = float('inf')
        
        # Generate all combinations of parameters
        import itertools
        all_params = [dict(zip(param_grid.keys(), v)) 
                    for v in itertools.product(*param_grid.values())]
        
        # Sample a subset if too many combinations
        if len(all_params) > 20:
            all_params = random.sample(all_params, 20)
        
        successful_fits = 0
        for params in all_params:
            try:
                # Adjust initial window based on data size
                initial = min('365 days', f"{len(df) // 2} days")
                horizon = min('30 days', f"{len(df) // 4} days")
                period = min('90 days', f"{len(df) // 3} days")
                
                m = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    n_changepoints=params['n_changepoints'],
                    changepoint_range=params['changepoint_range'],
                    weekly_seasonality=True,
                    yearly_seasonality=len(df) > 365,  # Only if enough data
                    daily_seasonality=False
                )
                
                # Only add holidays if enough data
                if len(df) > 90:
                    m.add_country_holidays(country_name='KE')
                
                # Add monthly seasonality only if enough data
                if len(df) > 60:
                    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                
                # Fit model with error handling
                with suppress_stdout_stderr():
                    m.fit(df)
                
                # Cross-validation with dynamic windows
                df_cv = cross_validation(
                    m,
                    initial=initial,
                    period=period,
                    horizon=horizon,
                    parallel="processes"
                )
                
                df_p = performance_metrics(df_cv)
                mape = df_p['mape'].mean()
                
                if mape < best_mape:
                    best_mape = mape
                    best_params = params
                    
                successful_fits += 1
                logging.info(f"Parameters: {params}, MAPE: {mape}")
                
            except Exception as e:
                logging.warning(f"Error with parameters {params}: {str(e)}")
                continue
        
        # If no successful fits or poor performance, use conservative defaults
        if not best_params or successful_fits == 0:
            logging.warning("Using default parameters as optimization failed")
            best_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'n_changepoints': min(25, max(1, len(df) // 4)),
                'changepoint_range': 0.8
            }
        elif best_mape > 1.0:  # If MAPE is too high, use more conservative parameters
            logging.warning(f"High MAPE ({best_mape}), adjusting parameters to be more conservative")
            best_params['changepoint_prior_scale'] = min(best_params['changepoint_prior_scale'], 0.05)
            best_params['n_changepoints'] = min(best_params['n_changepoints'], len(df) // 4)
        
        logging.info(f"Best parameters found: {best_params} with MAPE: {best_mape}")
        return best_params

    def calculate_metrics(self, actual_data: pd.DataFrame, forecast: pd.DataFrame) -> Dict:
        """Calculate model performance metrics"""
        # Merge actual and predicted values
        comparison_df = actual_data.merge(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds'
        )
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {
            'mae': mean_absolute_error(comparison_df['y'], comparison_df['yhat']),
            'rmse': np.sqrt(mean_squared_error(comparison_df['y'], comparison_df['yhat'])),
            'r2': r2_score(comparison_df['y'], comparison_df['yhat']),
            'mape': np.mean(np.abs((comparison_df['y'] - comparison_df['yhat']) / comparison_df['y'])) * 100
        }
        
        return metrics

    def save_training_results(self, sku: str, model: Prophet, metrics: Dict) -> bool:
        """Save model and training results to S3"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save model file
            model_path = f'/tmp/{sku}_model_{timestamp}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Upload model to S3
            model_key = f'models/{sku}/{timestamp}/model.pkl'
            self.s3.upload_file(model_path, self.bucket_name, model_key)
            
            # Save metrics
            metrics_key = f'models/{sku}/{timestamp}/metrics.json'
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=metrics_key,
                Body=json.dumps(metrics)
            )
            
            # Clean up
            os.remove(model_path)
            
            logging.info(f"Saved model and metrics to S3 for SKU {sku}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving training results: {e}")
            return False
        
    def make_multi_day_prediction(self, model: Prophet, sku: str, start_date: date, num_days: int) -> List[SimplePrediction]:
        """Make predictions for multiple days"""
        try:
            logging.info(f"Making {num_days} day prediction for SKU {sku} from {start_date}")
            
            # Create future dataframe
            future = pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=num_days, freq='D')
            })
            
            # Add required columns for logistic growth
            future['floor'] = 0  
            future['cap'] = model.history['y'].max() * 2  
            
            
            # Add required regressors
            future['y_ma7'] = model.history['y'].rolling(window=7).mean().iloc[-1]
            future['y_ma30'] = model.history['y'].rolling(window=30).mean().iloc[-1]
            
            # Make predictions
            forecast = model.predict(future)
            
            # Log raw predictions
            logging.info(f"Raw predictions:")
            logging.info(f"yhat: {forecast['yhat'].values}")
            
            # Format predictions
            predictions = []
            for _, row in forecast.iterrows():
                # Apply floor of 0 and add uncertainty bounds
                median_pred = max(0, float(row['yhat']))
                lower_bound = max(0, float(row['yhat_lower']))
                upper_bound = max(median_pred, float(row['yhat_upper']))
                
                prediction_value = Prediction(
                    median=median_pred,
                    low=lower_bound,
                    high=upper_bound
                )
                
                prediction = SimplePrediction(
                    product_sku=sku,
                    prediction_date=row['ds'].date(),
                    prediction_value=prediction_value
                )
                predictions.append(prediction)
            
            # Log prediction summary
            logging.info(f"\nPrediction Summary:")
            logging.info(f"Mean: {forecast['yhat'].mean():.2f}")
            logging.info(f"Min: {forecast['yhat'].min():.2f}")
            logging.info(f"Max: {forecast['yhat'].max():.2f}")
            logging.info(f"Std Dev: {forecast['yhat'].std():.2f}")
            
            return predictions            
        except Exception as e:
            logging.error(f"Error making predictions for SKU {sku}: {str(e)}")
            return None

async def train_from_s3(sku: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
    """Main function to train model using data from S3"""
    try:
        trainer = ModelTrainer()
        s3 = boto3.client('s3')
        bucket_name = os.getenv('S3_BUCKET_NAME')
        
        logging.info(f"Starting training for SKU: {sku}")
        if parameters:
            logging.info(f"Using custom parameters: {parameters}")
        
        # Get historical data
        historical_prefix = f'historical_data/{sku}/'
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=historical_prefix
        )
        
        all_data = []
        
        # Read historical data
        if 'Contents' in response:
            for obj in response['Contents']:
                s3_object = s3.get_object(
                    Bucket=bucket_name,
                    Key=obj['Key']
                )
                df = pd.read_csv(s3_object['Body'])
                all_data.append(df)
        
        # Get latest data from Elasticsearch
        latest_prefix = f'latest_data/{sku}/'
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=latest_prefix
        )
        
        if 'Contents' in response:
            latest_file = max(
                response['Contents'],
                key=lambda x: x['LastModified']
            )
            s3_object = s3.get_object(
                Bucket=bucket_name,
                Key=latest_file['Key']
            )
            df = pd.read_csv(s3_object['Body'])
            all_data.append(df)
        
        if not all_data:
            logging.error(f"No training data found for SKU {sku}")
            return False
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=['Date'])
        combined_data = combined_data.sort_values('Date')
        
        combined_data['Quantity'] = combined_data['Quantity'].astype(float)
        if combined_data['Quantity'].min() < 0:
            logging.warning(f"Negative values found in data for SKU {sku}")
        if combined_data['Quantity'].isna().any():
            logging.warning(f"Missing values found in data for SKU {sku}")

        
        # Train model
        logging.info(f"Training with parameters: {parameters}")
        model, metrics = trainer.train_model(sku, combined_data, parameters)
        
        if model is None or metrics is None:
            logging.error(f"Model training failed for SKU {sku}")
            return False
        
        if model is not None and metrics is not None:
            model_registry = ModelRegistry()
            success = model_registry.save_model(
                model=model,
                sku=sku,
                metrics=metrics,
                parameters=parameters or {}
            )
            if success:
                # Clean up old versions
                model_registry.cleanup_old_versions(sku)
                return True
        
        return False
    except Exception as e:
        logging.error(f"Error in train_from_s3 for SKU {sku}: {e}")
        return False
    
class suppress_stdout_stderr:
    """Context manager to suppress stdout/stderr during Prophet fitting"""
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)