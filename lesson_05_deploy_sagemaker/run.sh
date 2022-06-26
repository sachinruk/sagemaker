#! /usr/bin/env bash
export AWS_PROFILE=sachin-personal
ROLE_NAME=SageMakerRole_MNIST

POLICY=arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

aws iam attach-role-policy --policy-arn ${POLICY}  --role-name ${ROLE_NAME}