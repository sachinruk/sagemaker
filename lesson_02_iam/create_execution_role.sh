#!/usr/bin/env bash

# This code was copied from: https://github.com/aletheia/mnist_pl_sagemaker/blob/master/create_execution_role.sh

# This script creates a role named SageMakerRole
# that can be used by SageMaker and has Full access to S3.

export AWS_PROFILE=sachin-personal
ROLE_NAME=SageMakerRole_MNIST

# WARNING: this policy gives full S3 access to container that
# is running in SageMaker. You can change this policy to a more
# restrictive one, or create your own policy.
POLICY=arn:aws:iam::aws:policy/AmazonS3FullAccess
POLICY2=arn:aws:iam::aws:policy/CloudWatchFullAccess

# Creates the role
aws iam create-role --role-name ${ROLE_NAME} --assume-role-policy-document ./lesson_02_iam/assume-role-policy-document.json

# attaches the S3 full access policy to the role
aws iam attach-role-policy --policy-arn ${POLICY}  --role-name ${ROLE_NAME}
aws iam attach-role-policy --policy-arn ${POLICY2}  --role-name ${ROLE_NAME}