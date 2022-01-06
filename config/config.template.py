import os

CWD = os.getcwd()
MODELS_DIR = os.path.join(CWD, "models")

s3 = {
    "aws_access_key_id": "",
    "aws_secret_access_key": "",
    "region": "",
    "bucket": ""
}

port = 8964