# This short example is based on the fastai GitHub
# Repository of vision examples
# https://github.com/fastai/fastai/blob/master/examples/vision.ipynb
# Modified here to show mlflow.fastai.autolog() capabilities
#
import argparse
from fastai.vision.all import *
import mlflow.fastai
import dvc.api
import tarfile
import boto3
import io
import numpy as np
import os
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec, ColSpec
from mlflow.types import DataType

MLFLOW_PROJECT = os.environ.get('MLFLOW_PROJECT')


def parse_args():
    parser = argparse.ArgumentParser(description="Fastai example")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate to update step size at each step (default: 0.01)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs (default: 5). "
             "Note it takes about 1 min per epoch"
    )
    return parser.parse_args()


def extract(extract_path=MLFLOW_PROJECT+'/data_test'):
    fobj = io.BytesIO(
        dvc.api.read(
            path='data/mnist_tiny.tgz',
            repo='https://github.com/Seebak-Tech/fastai_example.git',
            remote='aws-remote',
            mode='rb'
        )
    )

    tar = tarfile.open(
        mode='r:gz',
        fileobj=fobj
    )
    #  for item in tar:
    tar.extractall(extract_path)


def set_session_token():
    # create an STS client object that represents a live connection to the 
    # STS service
    sts_client = boto3.client('sts')

    # Call the assume_role method of the STSConnection object and pass the role
    # ARN and a role session name.
    assumed_role_object = sts_client.assume_role(
        RoleArn=os.environ.get('AWS_ROLE_ARN'),
        RoleSessionName="AssumeRoleSession1"
    )

    # From the response that contains the assumed role, get the temporary 
    # credentials that can be used to make subsequent API calls
    credentials = assumed_role_object['Credentials']

    # Use the temporary credentials that AssumeRole returns to make a 
    # connection to Amazon S3  
    os.environ['AWS_ACCESS_KEY_ID'] = credentials['AccessKeyId']
    os.environ['AWS_SECRET_ACCESS_KEY_ID'] = credentials['SecretAccessKey']
    os.environ['AWS_SESSION_TOKEN'] = credentials['SessionToken']
    os.environ['AWS_DEFAULT_REGION'] = "us-west-2"

    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken'],
    )
    # Use the Amazon S3 resource object that is now configured with the
    # credentials to access your S3 buckets.
    for bucket in s3_resource.buckets.all():
        print(bucket.name)


def main():
    # Set Session token
    # set_session_token()
    # Parse command-line ArgumentParser
    args = parse_args()

    # Download and untar the MNIST data set
    path = MLFLOW_PROJECT + '/data_test/mnist_tiny'
    extract()

    # Prepare, transform, and normalize the data
    data = ImageDataLoaders.from_folder(
        path,
        train='train',
        valid='valid',
        num_workers=0
    )
    #  data.normalize(imagenet_stats)

    # Train and fit the Learner model
    learn = vision_learner(data, models.resnet18, metrics=accuracy)

    # Enable auto logging
    mlflow.fastai.autolog(log_models=False, silent=True)

    # Start MLflow session
    with mlflow.start_run():
        # Train and fit with default or supplied command line arguments
        learn.fit(args.epochs, args.lr)

        input_schema = Schema(
            [
                TensorSpec(np.dtype(np.uint8), (-1, 28, 28, 3)),
            ]
        )
        output_schema = Schema(
            [
                ColSpec(DataType.integer, "class"),
                ColSpec(DataType.string, "label"),
            ]
        )
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        saved_model = path + '/model.fastai'
        learn.export(saved_model)
        model_info = mlflow.pyfunc.log_model(
            "classifier",
            data_path=saved_model,
            code_path=["./fastai_example/fastai_classifier_module.py"],
            loader_module="fastai_classifier_module",
            signature=signature
        )


if __name__ == "__main__":
    main()
