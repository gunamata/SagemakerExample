# Build and deploy a Machine Learning model using AWS Sagemaker & Lambda

This example shows how to train and deploy a Machine Learning model using 2 approaches 
1. Amazon SageMaker 
2. AWS Lambda

## Approach 1. Amazon Sagemaker Approach

Steps to get the example working:

1. Install Python 3.6,  docker if not avaialable on your machine already

2. `mkvirtualenv python36-sagemaker`. Make sure the virtualenv is activated after you create it.

3. `pip install jupyter sagemaker numpy scipy scikit-learn pandas`

4. Create a new IAM User. You can use an existing IAM User as well but make sure you know 
   the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY of the user account.
   
5. Add a profile with a name (Ex:up-sagemaker) in the .aws/credentials file as below.

	`[up-sagemaker]
	aws_access_key_id = <your-access-key-id>
	aws_secret_access_key = <yout-secret-access-key>`
   
5. Create an AWS role. For example, SagemakerRole

6. Add a configuration to the .aws/config file

	`[profile up-sagemaker]
	region = <your-aws-region>
	role_arn = <arn of the role created in Step 5>
	source_profile = up-sagemaker`
	
7. Attach below persmission policies to the IAM role created in Step 5

	`AmazonEC2ContainerRegistryFullAccess
	AmazonS3FullAccess
	IAMReadOnlyAccess
	AmazonSageMakerFullAccess
	AmazonEC2FullAccess`
	
8. Run \container\build_and_push.sh to build a docker image with all the software (Python, Libraries etc) and 
the source code (logic to train, serve, predict) included.

	`build_and_push.sh <image-name> <profile>
	Example: build_and_push.sh iris-model up-sagemaker.
	Note: This script needs to be run from "container" folder in the source code.`
	
9. Run cells in the juPyter notebook train_and_deploy_your_first_model_on_sagemaker.ipynb to train, deploy, test the model.
	
That's all for the prerequisites and setup.

## Approach 2. Amazon AWS, API Gateway Approach (serverless framework)

1. Install Python 3.6,  node, npm if not avaialable on your machine already
2. Install serverless framework

   `npm install -g serverless`
   
   Note: Always run npm commands as admin to avoid problems.
   
3. Install npm packages required for the serverless project. This command installs npm modules under "node_modules" folder.

   `npm install`
   
4. Set below environment variables to be able to deploy, debug your serverless app onto AWS

   `export AWS_ACCESS_KEY_ID=<your-key-here>
   
    export AWS_SECRET_ACCESS_KEY=<your-secret-key-here>
    
    export PIP_DEFAULT_TIMEOUT=100
    
    export SLS_DEBUG=*`
    
5. Deploy the app to AWS cloud

    `serverless deploy`
    

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
