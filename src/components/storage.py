import os, sys

from boto3 import Session

from src.entity.config import s3Config
from src.logger import logging
from src.exception import CustomException



class StorageConnection:
    '''
    Created connection with S3 bucket using boto3 api to fetch the model from Repository.
    '''
    def __init__(self):
        self.config: s3Config = s3Config()
        self.session = Session(aws_access_key_id=self.config.ACCESS_KEY,
                               aws_secret_access_key=self.config.SECRET_KEY,
                               region_name=self.config.REGION)
        self.s3 = self.session.resource("s3")
        self.bucket = self.s3.Bucket(self.config.BUCKET)
    
    def get_package(self):
        try:
            logging.info("Fetching Artifacts From S3 Bucket .....")

            for name in self.config.ZIP_PATHS:
                download_dir = os.path.join(self.config.ARTIFACTS_ROOT, name)
                if os.path.exists(download_dir):
                    os.remove(download_dir)
                
                self.bucket.download_file(f'{self.config.KEY}/{name}', download_dir)
            logging.info("Fetching Completed !")

        except Exception as e:
            error = CustomException(e, sys)
            logging.error(error.error_message)
            raise error

if __name__ == "__main__":
    connection = StorageConnection()
    connection.get_package()