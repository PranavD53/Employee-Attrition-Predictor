import json
import boto3

runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):

    response = runtime.invoke_endpoint(
        EndpointName='employee-attrition-predictor',
        ContentType='application/json',
        Body=json.dumps(event)
    )

    result = json.loads(response['Body'].read())

    return {
        'statusCode': 200,
        'body': result
    }