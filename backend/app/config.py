from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # AWS Credentials
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"  # Change this to your preferred region
    
    # Service Endpoints
    s3_bucket_name: str = ""  # Will be populated from CDK outputs
    aurora_endpoint: str = ""  # Will be populated from CDK outputs
    cognito_user_pool_id: str = ""  # Will be populated from CDK outputs
    cognito_client_id: str = ""  # Will be populated from CDK outputs

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings() 