import boto3
from .config import get_settings

settings = get_settings()

# Initialize AWS session
session = boto3.Session(
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    region_name=settings.aws_region
)

# Initialize service clients
s3_client = session.client('s3')
rds_client = session.client('rds')
cognito_client = session.client('cognito-idp')

# Get the S3 bucket name
s3_bucket_name = settings.s3_bucket_name

# Get Aurora endpoint
aurora_endpoint = settings.aurora_endpoint

# Get Cognito settings
cognito_user_pool_id = settings.cognito_user_pool_id
cognito_client_id = settings.cognito_client_id 