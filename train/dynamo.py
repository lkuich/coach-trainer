import boto3
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('user-settings')

def get_model(accountId, modelName):
    account = table.get_item(
        Key={
            'id': accountId
        }
    )
    return account['Item']['models']['modelName']

def update_status(accountId, modelName, status):
    account = table.get_item(
        Key={
            'id': accountId
        }
    )

    models = account['Item']['models']
    if modelName in models: # Model exists, update status/version
        model = models[modelName]

        version = model['version']
        if status == 3: # Model is complete, increment version
            version = version + 1

        # Update version and status
        table.update_item(
            Key={
                'id': accountId
            },
            UpdateExpression='SET models.' + modelName + ' = :body',
            ExpressionAttributeValues={
                ':body': {
                    'version': version,
                    'status': status
                }
            }
        )
    else:
        raise ValueError('Model does not exist!')