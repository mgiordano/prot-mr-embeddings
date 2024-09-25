# thesis_exp

1. Create Python Virtual Environment venv

2. Install requirements

3. Create and download Service account key to use with Google Cloud project. Store the file in project root folder as "gcloud_saccount_key.json"

4. Create .env file in project root folder with the following variables:

INPUT_DATA_ROOT_PATH
GCLOUD_SERVICE_ACCOUNT_KEY
GCLOUD_INPUT_BUCKET
GCLOUD_PROJECT_ID

5. Make sure the processed protein input Maximal Repeats are stored under INPUT_DATA_ROOT_PATH. These should come from the output of mr-generator.

6. Run start_prefect.sh to setup Docker PostgreSQL container for Prefect database, a new profile with project config settings and finally start Prefect server. You may always run this script as a replacement for prefect server start as it will ensure the right settings are active.