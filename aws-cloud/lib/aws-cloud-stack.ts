import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';
import { S3Construct } from './services/s3-construct';
import { AuroraConstruct } from './services/aurora-construct';
import { CognitoConstruct } from './services/cognito-construct';
// import * as sqs from 'aws-cdk-lib/aws-sqs';

export class AwsCloudStack extends cdk.Stack {
  public readonly s3Bucket: S3Construct;
  public readonly aurora: AuroraConstruct;
  public readonly cognito: CognitoConstruct;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create VPC
    const vpc = new ec2.Vpc(this, 'PreveniaVPC', {
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        {
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
        {
          name: 'Isolated',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
    });

    // Create Security Group for Aurora
    const dbSecurityGroup = new ec2.SecurityGroup(this, 'DatabaseSecurityGroup', {
      vpc,
      description: 'Security group for Aurora DB',
      allowAllOutbound: true,
    });

    // Initialize S3 service
    this.s3Bucket = new S3Construct(this, 'S3Service', {
      bucketName: `prevenia-bucket-${this.account}`,
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY, // For development
      autoDeleteObjects: true, // For development
    });

    // Initialize Aurora service
    this.aurora = new AuroraConstruct(this, 'AuroraService', {
      vpc,
      securityGroup: dbSecurityGroup,
    });

    // Initialize Cognito service
    this.cognito = new CognitoConstruct(this, 'CognitoService');

    // Grant Aurora access to S3 for data import/export
    this.s3Bucket.bucket.grantRead(this.aurora.role);
    this.s3Bucket.bucket.grantWrite(this.aurora.role);

    // Output VPC ID
    new cdk.CfnOutput(this, 'VpcId', {
      value: vpc.vpcId,
      description: 'The VPC ID',
    });

    // TODO: Add more services here
    // - Lambda functions
    // - API Gateway
    // - DynamoDB tables
    // - Cognito user pools
    // etc.

    // The code that defines your stack goes here

    // example resource
    // const queue = new sqs.Queue(this, 'AwsCloudQueue', {
    //   visibilityTimeout: cdk.Duration.seconds(300)
    // });
  }
}
