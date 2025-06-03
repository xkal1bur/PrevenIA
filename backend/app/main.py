from fastapi import FastAPI, HTTPException, Depends
from .aws_clients import s3_client, rds_client, cognito_client, s3_bucket_name
from typing import List
import boto3

app = FastAPI(
    title="Patient Management API",
    description="API for managing patient data",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Welcome to Patient Management API"}

@app.get("/health")
async def health_check():
    try:
        # Check S3 access
        s3_client.list_objects_v2(Bucket=s3_bucket_name, MaxKeys=1)
        return {"status": "healthy", "services": {"s3": "connected"}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service health check failed: {str(e)}")

@app.get("/files")
async def list_files():
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name)
        files = [obj['Key'] for obj in response.get('Contents', [])]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}") 