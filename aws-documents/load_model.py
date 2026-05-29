import boto3
import pickle

s3 = boto3.client('s3')

s3.download_file(
    'employee-attrition-project',
    'model/random_forest.pkl',
    'random_forest.pkl'
)

model = pickle.load(open("random_forest.pkl", "rb"))