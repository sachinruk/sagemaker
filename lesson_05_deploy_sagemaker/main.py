import pathlib

import boto3
import numpy as np
import torchvision
import sagemaker
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer

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
sagemaker_client = session.client("sagemaker")
response = sagemaker_client.list_training_jobs(NameContains=JOB_NAME, StatusEquals="Completed")
if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
    raise RuntimeError(f"Could not retrieve any jobs with {JOB_NAME} OR connection issue")
training_job_name = response["TrainingJobSummaries"][0]["TrainingJobName"]

model_artifacts = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)["ModelArtifacts"]["S3ModelArtifacts"]

model = PyTorchModel(
    entry_point="inference.py",
    source_dir=str(pathlib.Path(__file__).parent / "inference_scripts"),
    role=role,
    model_data=model_artifacts,
    framework_version="1.10.0",
    py_version="py38",
    sagemaker_session=sagemaker_session,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

dummy_data = {"inputs": np.random.randint(0, 255, (16, 1, 28, 28)).tolist()}

res = predictor.predict(dummy_data)

breakpoint()
mnist = torchvision.datasets.MNIST(root="/tmp", train=False, download=True)
data = {"inputs": [np.array(mnist[i][0]).tolist() for i in range(16)]}
actuals = [mnist[i][1] for i in range(16)]
res = predictor.predict(data)

breakpoint()

predictor.delete_endpoint()
# use this: 
# https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
# https://sagemaker.readthedocs.io/en/stable/overview.html#consuming-sagemaker-model-packages
