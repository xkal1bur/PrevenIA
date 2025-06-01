#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { AwsCloudStack } from '../lib/aws-cloud-stack';

const app = new cdk.App();

// Create the main stack that contains all our services
new AwsCloudStack(app, 'PrevenIAStack', {
  env: { 
    account: process.env.CDK_DEFAULT_ACCOUNT, 
    region: process.env.CDK_DEFAULT_REGION 
  },
});