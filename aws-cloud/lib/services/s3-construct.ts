import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface S3ConstructProps {
  bucketName: string;
  allowedOrigins?: string[]; // Add allowed origins for CORS
}

export class S3Construct extends Construct {
  public readonly bucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: S3ConstructProps) {
    super(scope, id);

    // Create S3 bucket
    this.bucket = new s3.Bucket(this, 'Bucket', {
      bucketName: props.bucketName,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      cors: [
        {
          allowedMethods: [
            s3.HttpMethods.GET,
            s3.HttpMethods.PUT,
            s3.HttpMethods.POST,
            s3.HttpMethods.DELETE,
          ],
          allowedOrigins: props.allowedOrigins || ['*'], // Allow all origins by default
          allowedHeaders: ['*'],
          exposedHeaders: [
            'ETag',
            'x-amz-server-side-encryption',
            'x-amz-request-id',
            'x-amz-id-2',
          ],
          maxAge: 3000,
        },
      ],
    });

    // Output the bucket name and ARN
    new cdk.CfnOutput(this, 'BucketName', {
      value: this.bucket.bucketName,
      description: 'The S3 bucket name',
    });

    new cdk.CfnOutput(this, 'BucketArn', {
      value: this.bucket.bucketArn,
      description: 'The S3 bucket ARN',
    });

    // Output the bucket URL
    new cdk.CfnOutput(this, 'BucketUrl', {
      value: `https://${this.bucket.bucketName}.s3.${cdk.Stack.of(this).region}.amazonaws.com`,
      description: 'The S3 bucket URL',
    });
  }
}
