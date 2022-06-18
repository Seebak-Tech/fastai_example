import boto3
from fastai.vision.all import *
#  from fastai.data.external import *
import os
import numpy as np
import json

app_name = os.environ.get('APP_NAME')
region = os.environ.get('REGION')
mlflow_project = os.environ.get('MLFLOW_PROJECT')


def main():
    sm = boto3.client('sagemaker', region_name=region)
    smrt = boto3.client('runtime.sagemaker', region_name=region)

    # Check endpoint status
    endpoint = sm.describe_endpoint(EndpointName=app_name)
    print("Endpoint status:", endpoint["EndpointStatus"])

    # get all the image paths from testing folder
    path = mlflow_project + '/data_test/mnist_tiny/test'
    images = get_image_files(path)

    # Predict image
    #  img = load_image(images[3], mode='RGB')
    #  img = load_image(images[3], mode='RGB')
    #  print(img)
    #  img_arr = np.array(img)
    #  print(img_arr.shape, img_arr.ndim)
    #  img_batch = img_arr.reshape(
    #      1,
    #      28,
    #      28,
    #      3
    #  )
    #  #  print(img_batch.shape)
    #  img_json = json.dumps(
    #      {
    #          "instances": img_batch.tolist()
    #      }
    #  )
    #  images_arr = np.array(
    #      [np.array(load_image(img, mode='RGB')) for img in images]
    #  )
    images_arr = np.array(
        [np.array(load_image(img, mode='RGB')) for img in images[:3]]
    )

    img_json = json.dumps(
        {
            "instances": images_arr.tolist()
        }
    )
    query_input = img_json
    prediction = smrt.invoke_endpoint(
        EndpointName=app_name,
        Body=query_input,
        ContentType='application/json'
    )
    prediction = prediction['Body'].read().decode("ascii")
    print(prediction)

if __name__ == "__main__":
    main()
