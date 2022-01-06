import boto3
from boto3.s3.transfer import S3Transfer
import logging
import sys
import os
logging.basicConfig(level=logging.INFO)

class AWSHandler():

    def __init__(self, accessKey, secretKey, region, bucket):
        self.accessKey = accessKey
        self.secretKey = secretKey
        self.region = region
        self.bucket = bucket

    def upload_2S3(self, s3_path, local_path):
        # connect to s3
        s3 = boto3.client('s3',
                            aws_access_key_id = self.accessKey,
                            aws_secret_access_key = self.secretKey,
                            region_name = self.region
                        )
        transfer = S3Transfer(s3)
        #transfer.upload_file(mdir+"/"+zip_file, bucket, zip_file)
        transfer.upload_file(local_path, self.bucket, s3_path)

    def download_fromS3(self, s3_path, local_path):
        # connect to s3
        s3 = boto3.client('s3',
                        aws_access_key_id = self.accessKey,
                        aws_secret_access_key = self.secretKey,
                        region_name = self.region)
        s3.download_file(self.bucket, s3_path, local_path)