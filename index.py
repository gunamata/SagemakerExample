import boto3
import numpy as np
import os.path
import json
import os
import pickle
import pandas as pd

strBucket = 'sagemaker-us-east-1-298573704325'

def handler(event, context):
    if isinstance(event['body'], (unicode, str)):
        sample = json.loads(event['body'])    
    result = predict(sample)
    return {'StatusCode':200,
    'body':result[0]}  

def predict(sample):
    global strBucket
    if not os.path.exists('/tmp/model/'):
        os.makedirs('/tmp/model/')
    dest = '/tmp/model/model.pkl'
    src = '/up-lambda-iris-model/model.pkl'
    model = get_model(strBucket,src,dest)
    result = model.predict(sample)
    return result
        
def get_model(strBucket,src,dest):
    bucket= boto3.resource('s3').Bucket(strBucket)
    bucket.download_file(src,dest)
    with open(os.path(dest), 'rb') as inp:
	    model = pickle.load(inp)
    return(model)      

