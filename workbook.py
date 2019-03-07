import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role

# Define IAM role
import boto3
import re

import sagemaker as sage
from time import gmtime, strftime

# S3 prefix
prefix = 'tf-cpu-13'

role = get_execution_role()

sess = sage.Session()

WORK_DIRECTORY = 'test.txt'
data_location = sess.upload_data(path=WORK_DIRECTORY, bucket='tensorflow-output', key_prefix=prefix)

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/tensorflow-cpu'.format(account, region)

tree = sage.estimator.Estimator(image,
                       role, 1, 'ml.c5.2xlarge',
                       output_path="s3://{}/output".format(sess.default_bucket()),
                       sagemaker_session=sess)

tree.fit(data_location)