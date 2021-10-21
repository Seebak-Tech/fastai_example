# This short example is based on the fastai GitHub
# Repository of vision examples
# https://github.com/fastai/fastai/blob/master/examples/vision.ipynb
# Modified here to show mlflow.fastai.autolog() capabilities
#
import argparse
import fastai.vision as vis
import mlflow.fastai
import dvc.api
import tarfile
import boto3
import io
import os


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
        help="number of epochs (default: 5). " \
             "Note it takes about 1 min per epoch"
    )
    return parser.parse_args()


def extract(extract_path='./data_test'):
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
    os.environ['AWS_SESSION_TOKEN'] = credentials['SessionToken']


def main():
    # Set Session token
    set_session_token()
    # Parse command-line ArgumentParser
    args = parse_args()

    # Download and untar the MNIST data set
    path = '/workspace/fastai_example/data_test/mnist_tiny'
    extract()

    # Prepare, transform, and normalize the data
    data = vis.ImageDataBunch.from_folder(
        path,
        ds_tfms=(vis.rand_pad(2, 28), []),
        bs=64
    )
    data.normalize(vis.imagenet_stats)

    # Train and fit the Learner model
    learn = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy)

    # Enable auto logging
    mlflow.fastai.autolog()

    # Start MLflow session
    with mlflow.start_run():
        # Train and fit with default or supplied command line arguments
        learn.fit(args.epochs, args.lr)


if __name__ == "__main__":
    main()
