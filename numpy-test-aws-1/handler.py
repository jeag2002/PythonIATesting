import json

import numpy as np

def main(event, context):
    print("parameter_1")
    print(event['parameter_1'])
    print("parameter_2")
    print(event['parameter_2'])
    a = np.matrix(event['parameter_1'])
    b = np.matrix(event['parameter_2'])
    print("input value a")
    print(a)
    print("input value b")
    print(b)
    c = np.add(a,b)
    print("response")
    print(c)
    response = {"statusCode": 200,"body": json.dumps(c.tolist())}
    return response

if __name__ == "__main__":
    dict = {'parameter_1':'1,2;3,4','parameter_2':'5,6;7,8'}
    main(dict, '')


"""
def hello(event, context):
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
"""
