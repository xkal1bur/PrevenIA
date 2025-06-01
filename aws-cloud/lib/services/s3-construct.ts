import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface S3ConstructProps {
  bucketName?: string;
  versioned?: boolean;
  removalPolicy?: cdk.RemovalPolicy;
  autoDeleteObjects?: boolean;
}

export class S3Construct extends Construct {
  public readonly bucket: s3.Bucket;

  constructor(scope: Construct, id: string, props?: S3ConstructProps) {
    super(scope, id);

    // Create an S3 bucket with configurable properties
    this.bucket = new s3.Bucket(this, 'Bucket', {
      bucketName: props?.bucketName || `prevenia-bucket-${cdk.Stack.of(this).account}`,
      versioned: props?.versioned ?? true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: props?.removalPolicy ?? cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: props?.autoDeleteObjects ?? true,
    });

    // Output the bucket name
    new cdk.CfnOutput(this, 'BucketName', {
      value: this.bucket.bucketName,
      description: 'The name of the S3 bucket',
    });
  }
}
