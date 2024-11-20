from google.cloud import storage
from dotenv import dotenv_values
import pandas as pd
import re
import os

#travels up two levels to find the .env
dotenv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', '.env'))
print(dotenv_path)
config = dotenv_values(dotenv_path)

class StorageHelper():
    
    client = None
    
    def __init__(self):
        gcloud_service_account_key_path = config["GCLOUD_SERVICE_ACCOUNT_KEY"]
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config["GCLOUD_SERVICE_ACCOUNT_KEY"]
        self.client = storage.Client.from_service_account_json(gcloud_service_account_key_path)
        self.bucket_name = config["GCLOUD_INPUT_BUCKET"]
    
    def upload_file(self, input_root_path, file_name, destination_root_path):
        """Uploads a file to the bucket."""
        input_file_path = os.path.join(input_root_path, file_name)
        destination_file_path = os.path.join(destination_root_path, file_name)
        full_gcs_path = os.path.join(self.bucket_name, destination_file_path)
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(destination_file_path)
        if not blob.exists():
            blob.upload_from_filename(input_file_path)
            print(f"File {input_file_path} uploaded to {destination_file_path}.")
        return f"gs://{full_gcs_path}" 
    
    def download_file(self, bucket_name, remote_file_name, remote_file_path, destination_root_path):
        """Downloads a blob from the bucket."""  

        storage_client = self.client
        download_file_path = os.path.join(destination_root_path, remote_file_name)
        full_gcs_path = os.path.join(remote_file_path, remote_file_name)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(full_gcs_path)
        os.makedirs(destination_root_path, exist_ok=True)
        # Check if the file already exists locally
        if os.path.exists(download_file_path):
            print(f"File {download_file_path} already exists. Skipping download.")
            return  # return without downloading

        blob.download_to_filename(download_file_path)  
        print(
            "Blob {} downloaded to {}.".format(
                remote_file_name, download_file_path
            )
        )
        # decompress gz file
        df = pd.read_csv(download_file_path, compression='gzip')
        uncompressed_file_path = re.sub(r"\.gz$", ".csv", download_file_path)
        df.to_csv(uncompressed_file_path, index=False, encoding="utf-8")
        print(
            "Decompressed {} to {}.".format(
                remote_file_name, uncompressed_file_path
            )
        )

    def download_files_matching_prefix(self, bucket_name, remote_file_name_prefix, remote_file_path, destination_root_path):
        """Downloads blobs from the bucket that match the given prefix."""
        storage_client = self.client
        bucket = storage_client.bucket(bucket_name)
        full_path_prefix = os.path.join(remote_file_path, remote_file_name_prefix)
        blobs = bucket.list_blobs(prefix=full_path_prefix)  # List blobs with the specified prefix
        for blob in blobs:
            # Extract the filename from the blob name
            filename = blob.name.split('/')[-1]  
            self.download_file(bucket_name, filename, remote_file_path, destination_root_path)
