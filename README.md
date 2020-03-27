# Build and deploy a Machine Learning model using AWS Sagemaker & Lambda

This example shows how to train and deploy a Machine Learning model using 2 approaches 
1. Amazon SageMaker 
2. AWS Lambda

## 1. Amazon Sagemaker Approach

Prerequisites:

1. Python 3.6,  docker are installed
2. `mkvirtualenv python36-sagemaker`. Make sure the virtualenv is activated after you create it.
3. `pip install jupyter sagemaker numpy scipy scikit-learn pandas`
4. Create a new IAM User. You can use an existing IAM User as well but make sure you know the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY of the user account.
5. Add a profile with a name (Ex:up-sagemaker) in the .aws/credentials file as below.
	[up-sagemaker]
	aws_access_key_id = <your-access-key-id>
	aws_secret_access_key = <yout-secret-access-key>
   
5. Create an AWS role. For example, SagemakerRole
6. Add a configuration to the .aws/config file
	[profile up-sagemaker]
	region = <your-aws-region>
	role_arn = <arn of the role created in Step 5>
	source_profile = up-sagemaker
7. Attach below persmission policies to the IAM role created in Step 5
	AmazonEC2ContainerRegistryFullAccess
	AmazonS3FullAccess
	IAMReadOnlyAccess
	AmazonSageMakerFullAccess
	AmazonEC2FullAccess
	
That's all for the prerequisites and setup.

## How Amazon SageMaker Runs Training and Prediction

1. `docker run image train`: for training
2. `docker run image serve`: for prediction

## Training

### Data

The [iris](https://archive.ics.uci.edu/ml/datasets/iris) data set is used for training the model.

### Machine Learning Algorithm

[scikit-learn's Random Forest implementation](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is chosen in order to train the iris classifier.

### Machine Learning Model Accuracy Metrics

Precision and Recall are used to evaluate the trained ML model.

## Prediction

The following technologies are used in order to build a RESTful prediction service:

1. [nginx](https://www.nginx.com/): a high-performance web server to handle and serve HTTP requests and responses, respectively.
2. [gunicorn](https://gunicorn.org/): a Python WSGI HTTP server responsible to run multiple copies of your application and load balance between them.
3. [flask](http://flask.pocoo.org/): a Python micro web framework that lets you implement the controllers for the two SageMaker endpoints `/ping` and `/invocations`. 

REST Endpoints:

1. `GET /ping`: health endpoint
2. `POST /invocations`: predict endpoint that expects a JSON body with the required features

## Code

1. `train_and_deploy_your_first_model_on_sagemaker.ipynb`: Jupyter notebook to train/deploy your first ML model on SageMaker
