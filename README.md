app/: Contains the FastAPI application code.
main.py: Entry point of the FastAPI app.
models.py: Contains functions for model training and predictions.
database.py: Handles data fetching from S3 and Elasticsearch.
schemas.py: Defines Pydantic models for request and response validation.
utils.py: Utility functions (e.g., logging setup).
ingestion.py: Functions for data ingestion from Elasticsearch to S3.
requirements.txt: Lists Python dependencies.
Dockerfile: Instructions to containerize the application.
.env: Stores environment variables.