#!/bin/bash

# https://docs.prefect.io/3.0/manage/self-host
# https://docs.prefect.io/3.0/manage/settings-and-profiles

# run docker postgresql container
docker-compose up -d

# set prefect config

PROFILE_NAME="bioword_profile"  # Replace with your desired profile name

if ! prefect profile ls | grep $PROFILE_NAME; then
    echo "Prefect profile $PROFILE_NAME does not exist. Creating it..."
    prefect profile create $PROFILE_NAME

    # Add your configuration 
    # use local server
    prefect -p $PROFILE_NAME config set PREFECT_API_URL=http://127.0.0.1:4200/api

    # disable result persist to prevent concurrency errors
    prefect -p $PROFILE_NAME config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=false

    # set PostgreSQL db from docker file 
    # supports concurrency better than default Sqlite
    prefect -p $PROFILE_NAME config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopS3cretPassw0rd!@localhost:5432/prefect"

    # ... other config settings
fi

echo "Activating profile $PROFILE_NAME..."
prefect profile use $PROFILE_NAME

# Check if a Prefect server is already running
if pgrep -f "prefect server start" > /dev/null; then
    echo "Prefect server is already running."
else
    echo "Starting Prefect server..."
    prefect server start
fi