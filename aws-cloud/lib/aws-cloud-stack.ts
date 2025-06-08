import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';
import { S3Construct } from './services/s3-construct';
import { AuroraConstruct } from './services/aurora-construct';
import { DefaultStackSynthesizer } from 'aws-cdk-lib';

export class AwsCloudStack extends cdk.Stack {
  public readonly s3Bucket: S3Construct;
  public readonly aurora: AuroraConstruct;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    const defaultStackSynthesizer = new DefaultStackSynthesizer({
      // Name of the S3 bucket for file assets
      fileAssetsBucketName: "cdk-${Qualifier}-assets-${AWS::AccountId}-${AWS::Region}",
      bucketPrefix: "",

      // Name of the ECR repository for Docker image assets
      imageAssetsRepositoryName: "cdk-${Qualifier}-container-assets-${AWS::AccountId}-${AWS::Region}",

      // ARN of the role assumed by the CLI and Pipeline to deploy here
      deployRoleArn: "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/LabRole",
      deployRoleExternalId: "",

      // ARN of the role used for file asset publishing (assumed from the deploy role)
      fileAssetPublishingRoleArn: "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/LabRole",
      fileAssetPublishingExternalId: "",

      // ARN of the role used for Docker asset publishing (assumed from the deploy role)
      imageAssetPublishingRoleArn: "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/LabRole",
      imageAssetPublishingExternalId: "",

      // ARN of the role passed to CloudFormation to execute the deployments
      cloudFormationExecutionRole: "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/LabRole",

      // ARN of the role used to look up context information in an environment
      lookupRoleArn: "arn:${AWS::Partition}:iam::${AWS::AccountId}:role/LabRole",
      lookupRoleExternalId: "",

      // Name of the SSM parameter which describes the bootstrap stack version number
      bootstrapStackVersionSsmParameter: "/cdk-bootstrap/${Qualifier}/version",

      // Add a rule to every template which verifies the required bootstrap stack version
      generateBootstrapVersionRule: true,
    });

    super(scope, id, {
      ...props,
      synthesizer: defaultStackSynthesizer
    });

    // Create VPC with minimal configuration
    const vpc = new ec2.Vpc(this, 'PreveniaVPC', {
      maxAzs: 2,
      natGateways: 0, // Remove NAT Gateway to avoid IAM role creation
      subnetConfiguration: [
        {
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        }
      ],
      // Disable custom resource creation
      createInternetGateway: true,
      enableDnsHostnames: true,
      enableDnsSupport: true,
      restrictDefaultSecurityGroup: false,
    });

    // Create Security Group for Aurora
    const dbSecurityGroup = new ec2.SecurityGroup(this, 'DatabaseSecurityGroup', {
      vpc,
      description: 'Security group for Aurora DB',
      allowAllOutbound: true,
    });

    // Allow inbound PostgreSQL access
    dbSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(5432),
      'Allow PostgreSQL access'
    );

    // Initialize S3 service
    this.s3Bucket = new S3Construct(this, 'S3Service', {
      bucketName: `prevenia-bucket-${this.account}-${Date.now()}`,
    });

    // Initialize Aurora service
    this.aurora = new AuroraConstruct(this, 'AuroraService', {
      vpc,
      securityGroup: dbSecurityGroup,
    });

    // Output VPC ID
    new cdk.CfnOutput(this, 'VpcId', {
      value: vpc.vpcId,
      description: 'The VPC ID',
    });

    // Output the database endpoint
    new cdk.CfnOutput(this, 'DatabaseEndpoint', {
      value: this.aurora.cluster.clusterEndpoint.hostname,
      description: 'The database endpoint',
    });

    // Output the database port
    new cdk.CfnOutput(this, 'DatabasePort', {
      value: '5432',
      description: 'The database port',
    });

    // Output the S3 bucket name
    new cdk.CfnOutput(this, 'BucketName', {
      value: this.s3Bucket.bucket.bucketName,
      description: 'The S3 bucket name',
    });

    // TODO: Add more services here
    // - Lambda functions
    // - API Gateway
    // - DynamoDB tables
    // etc.
  }
}
