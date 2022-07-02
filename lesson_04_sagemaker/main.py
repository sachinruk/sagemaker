import pathlib

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

BUCKET = "sachin-mnist-2"
PREFIX = "data/mnist"
PROFILE_NAME = "sachin-personal"
ROLE_NAME = "SageMakerRole_MNIST"
JOB_NAME = "mnist-lightning"

session = boto3.session.Session(profile_name=PROFILE_NAME)
sts = session.client("sts")
account_number = sts.get_caller_identity().get("Account")
role = f"arn:aws:iam::{account_number}:role/{ROLE_NAME}"
sagemaker_session = sagemaker.Session(boto_session=session)

# Initializes SageMaker session which holds context data
estimator = PyTorch(
    entry_point="main.py",
    source_dir=str(pathlib.Path(__file__).parent / "training_scripts"),
    role=role,
    framework_version="1.10.0",
    py_version="py38",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    hyperparameters={
        "epochs": 2,
        "batch-size": 128,
    },
    sagemaker_session=sagemaker_session,
    base_job_name=JOB_NAME,
)

train_input = f"s3://{BUCKET}/{PREFIX}/train.npy"
valid_input = f"s3://{BUCKET}/{PREFIX}/valid.npy"

estimator.fit({"train": train_input, "valid": valid_input}, wait=True)
