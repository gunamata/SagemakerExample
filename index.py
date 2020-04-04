import boto3
import numpy as np
import os.path
import json
import os
import pickle
import pandas as pd

strBucket = 'sagemaker-us-east-1-298573704325'
strBucketKey = 'up-lambda-iris-model/'
strModelFilename = 'model.pkl'
strLocalTempDir = '/tmp/model/'

def get_model(src, dest):
    global strBucket
    bucket = boto3.resource('s3').Bucket(strBucket)
    bucket.download_file(src, dest)
    with open(dest,'rb') as inp:
        model = pickle.load(inp)
    return model

def predict(input):
    global strBucketKey, strModelFilename, strLocalTempDir
    if not os.path.exists(strLocalTempDir):
        os.makedirs(strLocalTempDir)
    src = strBucketKey + strModelFilename
    dest = strLocalTempDir + strModelFilename
    model = get_model(src,dest)
    return model.predict(input)

def handler(event, context):
    data = event['body']
    data_dict = json.loads(data)
    data_df = pd.DataFrame.from_dict(data_dict)
    predictions =  predict(data_df.values)
    body = { "predictions": np.array2string(predictions) }
    ret =  { "headers": { "Content-Type": "application/json" }, "statusCode": 200, "body": json.dumps(body) } 
    return ret 
    

