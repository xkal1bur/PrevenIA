import * as cdk from 'aws-cdk-lib';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface AuroraConstructProps {
  vpc: ec2.IVpc;
  securityGroup: ec2.ISecurityGroup;
}

export class AuroraConstruct extends Construct {
  public readonly cluster: rds.DatabaseCluster;

  constructor(scope: Construct, id: string, props: AuroraConstructProps) {
    super(scope, id);

    // Create Aurora PostgreSQL cluster
    this.cluster = new rds.DatabaseCluster(this, 'Database', {
      engine: rds.DatabaseClusterEngine.auroraPostgres({
        version: rds.AuroraPostgresEngineVersion.VER_15_3,
      }),
      credentials: rds.Credentials.fromGeneratedSecret('postgres'),
      instanceProps: {
        vpc: props.vpc,
        vpcSubnets: {
          subnetType: ec2.SubnetType.PUBLIC,
        },
        securityGroups: [props.securityGroup],
        instanceType: ec2.InstanceType.of(
          ec2.InstanceClass.T3,
          ec2.InstanceSize.MEDIUM
        ),
        publiclyAccessible: true,
      },
      defaultDatabaseName: 'prevenia',
    });

    // Output the cluster endpoint
    new cdk.CfnOutput(this, 'ClusterEndpoint', {
      value: this.cluster.clusterEndpoint.hostname,
      description: 'The cluster endpoint',
    });

    // Output the port
    new cdk.CfnOutput(this, 'ClusterPort', {
      value: '5432',
      description: 'The cluster port',
    });

    // Output the secret ARN for the database password
    new cdk.CfnOutput(this, 'SecretArn', {
      value: this.cluster.secret?.secretArn || 'No secret created',
      description: 'The ARN of the secret containing the database password',
    });
  }
} 