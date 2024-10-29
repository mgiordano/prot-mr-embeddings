from google.cloud import storage
from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud.exceptions import NotFound
from dotenv import dotenv_values
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
        blob.download_to_filename(download_file_path)  


        print(
            "Blob {} downloaded to {}.".format(
                remote_file_name, download_file_path
            )
        )


class DatabaseHelper():
    # Input data group
    dataset_name = ""
    input_data_root_path = ""
    run_id = ""

    # GCloud
    project_id = ""
    location = "US"

    # Storage #
    storage_helper = StorageHelper()

    # BigQuery #
    client = None
    bqstorage_client = None
    read_session = None
    # Dataset names
    BQ_INPUT_DATASET_NAME = "protein_input"
    BQ_CORPUS_DATASET_NAME = "protein_corpus"
    BQ_STAGE_DATASET_NAME = 'stage_results'

    # Table suffixes
    SEQUENCES_TABLE_SUFFIX = "_sequences"
    PATTERNS_TABLE_SUFFIX = "_patterns"
    POSITIONS_TABLE_SUFFIX = "_positions"
    PATTERN_POSITION_VIEW_SUFFIX = "_pattern_positions"

    # Table data
    sequences_table_name = ""
    sequences_table_cluster_columns = ["family_name"]
    patterns_table_name = ""
    patterns_table_cluster_columns = ["type"]
    positions_table_name = ""
    positions_table_cluster_columns = ["pattern_id", "protein_id"]
    family_types_table_name = "family_types"
    family_types_table_schema = [
        {
            'column_name' : 'family_type',
            'column_type' : 'STRING'
        },
        {
            'column_name' : 'family_name',
            'column_type' : 'STRING'
        },
    ]
    family_types_table_cluster_columns = []

    pattern_position_view_name = ""

    # Queries
    query = ""
    udf_query = ""
    CHAIN_QUERY_WILDCARD = "<CHAIN_QUERY>"
    # indicate wether stage results should be materialized
    dry_run = False

    def __init__(self, dataset_name, input_data_root_path="", run_id="", dry_run=False):
        self.input_data_root_path = input_data_root_path
        self.dataset_name = dataset_name
        gcloud_service_account_key_path = config["GCLOUD_SERVICE_ACCOUNT_KEY"]
        self.project_id = config["GCLOUD_PROJECT_ID"]
        self.client = bigquery.Client.from_service_account_json(gcloud_service_account_key_path, location=self.location)
        self.bqstorage_client = bigquery_storage.BigQueryReadClient()
        self.sequences_table_name = self.dataset_name + self.SEQUENCES_TABLE_SUFFIX
        self.patterns_table_name = self.dataset_name + self.PATTERNS_TABLE_SUFFIX
        self.positions_table_name = self.dataset_name + self.POSITIONS_TABLE_SUFFIX
        self.run_id = run_id
        self.dry_run = dry_run
        if dry_run:
            self.BQ_CORPUS_DATASET_NAME += "_test"

    def get_init_params(self):
        return {
            "input_data_root_path" : self.input_data_root_path,
            "dataset_name" : self.dataset_name,
            "run_id" : self.run_id,
            "dry_run" : self.dry_run
        }

    def load_sequences_dataset(self, input_root_path, file_name):
        self.load_or_create_table(input_root_path, "family_types.csv", self.dataset_name, self.BQ_INPUT_DATASET_NAME, 
                                self.family_types_table_name, cluster_columns=self.family_types_table_cluster_columns,
                                table_schema=self.family_types_table_schema)
        self.load_or_create_table(input_root_path, file_name, self.dataset_name, self.BQ_INPUT_DATASET_NAME, 
                                  self.sequences_table_name, cluster_columns=self.sequences_table_cluster_columns)
        return self
    
    def load_patterns_dataset(self, input_dataset_path, file_name):
        self.load_or_create_table(input_dataset_path, file_name, self.dataset_name, self.BQ_INPUT_DATASET_NAME, 
                                  self.patterns_table_name, cluster_columns=self.patterns_table_cluster_columns)
        return self
    
    def load_positions_dataset(self, input_dataset_path, file_name):
        self.load_or_create_table(input_dataset_path, file_name, self.dataset_name, self.BQ_INPUT_DATASET_NAME, 
                                  self.positions_table_name, cluster_columns=self.positions_table_cluster_columns)
        return self

    def check_table_existence(self, bq_dataset_name, table_name):
        table_id = f"{self.project_id}.{bq_dataset_name}.{table_name}" 
        table_exists = False
        try:
            self.client.get_table(table_id)  # Make an API request.
            table_exists = True
        except NotFound:
            print(f"Table {table_id} does not exist.")
        return table_exists
    
    def load_or_create_table(self, input_root_path, file_name, gcs_folder, bq_dataset_name, table_name, cluster_columns=[], gcs_uri="", table_schema=[]):
        # load existing or create table
        table_exists = self.check_table_existence(bq_dataset_name, table_name)
        
        if not table_exists:
            print(f"Loading data...")
            dataset_id = f"{self.project_id}.{bq_dataset_name}"
            dataset = bigquery.Dataset(dataset_id)
            dataset = self.client.create_dataset(dataset, exists_ok=True)

            if not len(gcs_uri) > 0:
                # load input data into cloud storage
                gcs_uri = self.storage_helper.upload_file(input_root_path, file_name, gcs_folder)

            # Configure the job and schema options to load gcs data into bq
            # https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#jobconfigurationquery
            job_config = bigquery.LoadJobConfig()
            if len(table_schema) > 0:
                schema = []
                for column_schema in table_schema:
                    schema.append(bigquery.SchemaField(column_schema["column_name"], column_schema["column_type"]))
                job_config.schema = schema
            else:
                job_config.autodetect = True
            job_config.skip_leading_rows = 1
            job_config.max_bad_records = 1000000
            job_config.source_format = bigquery.SourceFormat.CSV
            if len(cluster_columns) > 0:
                job_config.clustering_fields = cluster_columns

            bq_uri = f"{dataset_id}.{table_name}"
            # Create the load job
            load_job = self.client.load_table_from_uri(
                gcs_uri, bq_uri, job_config=job_config
            )

            # Wait for the job to complete
            load_job.result()
            # Print confirmation
            destination_table = self.client.get_table(bq_uri)
            print("Loaded {} rows into ".format(destination_table.num_rows) + bq_uri)

        return self

    def add_options_to_query(self, query, options):
        if len(options) > 0:
            query += " OPTIONS("
            option_definitions = []
            for option in options:
                option_definition = option["key"] + " = " + str(option["value"])
                option_definitions.append(option_definition)
            query += ",".join(option_definitions) + ")"
        return query

    def save_to_table(self, output_table_name, cluster_columns=[], options=[]):
        """Executes chained queries up to calling point and materializes result to table"""
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#create_table_statement

        table_id = f"{self.project_id}.{self.BQ_STAGE_DATASET_NAME}.{output_table_name}"
        dataset_id = f"{self.project_id}.{self.BQ_STAGE_DATASET_NAME}"
        dataset = bigquery.Dataset(dataset_id)
        dataset = self.client.create_dataset(dataset, exists_ok=True)
        query = f"CREATE OR REPLACE TABLE `{table_id}`"
        if len(cluster_columns) > 0:
            query += " CLUSTER BY " + ",".join(cluster_columns)    
        query = self.add_options_to_query(query, options)
        query += f" AS {self.query}"
        return self.execute(query_override=query)

    def export_table_to_gcs_as_csv(self, export_table_name, gcs_root_path, export_columns=["*"], file_name_suffix="", extra_options=[]):
        """Executes chained queries up to calling point and materializes result to table"""
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/export-statements

        table_id = f"{self.project_id}.{self.BQ_STAGE_DATASET_NAME}.{export_table_name}"
        self.select_from_table(table_id, column_names=export_columns)
        file_name = export_table_name + file_name_suffix + ".gz"
        full_gcs_path = os.path.join("gs://", self.storage_helper.bucket_name, gcs_root_path, file_name)
        options = [
            {
                "key" : "uri",
                "value": f'"{full_gcs_path}"'
             },
             {
                 "key" : "compression",
                 "value" : '"GZIP"'
             },
             {
                 "key" : "format",
                 "value" : '"CSV"'
             },
             {
                 "key" : "overwrite",
                 "value" : 'TRUE'
             }
        ]
        for extra_option in extra_options:
            options.append(extra_option)
        query = f"EXPORT DATA"
        query = self.add_options_to_query(query, options)
        query += f" AS {self.query}"
        return self.execute(query_override=query)
    
    def create_stream_session(self, table_name, dataset_name="", max_stream_count=1):
        """Streams tables that are very big to read in one pass"""
        # Configure the read session
        dataset_name = self.BQ_STAGE_DATASET_NAME if len(dataset_name) == 0 else dataset_name
        requested_session = bigquery_storage.types.ReadSession(
            data_format=bigquery_storage.types.DataFormat.AVRO, # Use Avro for DataFrame conversion
            table=f"projects/{self.project_id}/datasets/{dataset_name}/tables/{table_name}",
        )
        self.read_session = self.bqstorage_client.create_read_session(
            parent=f"projects/{self.project_id}",
            read_session=requested_session,
            max_stream_count=max_stream_count,
        )
        return self.read_session
    
    def read_rows_from_stream(self, stream_name):
        return self.bqstorage_client.read_rows(stream_name)
        
    def execute(self, query_override=""):
        query = query_override if len(query_override) > 0 else self.query
        query = self.udf_query + '\n' + query if len(self.udf_query) > 0 else query
        df_result = self.client.query_and_wait(query).to_dataframe()
        # cleanup
        self.query = ""
        self.udf_query = ""
        return df_result

    def set_query(self, query, limit=0, chain_query=False):
        if limit > 0:
            query += " LIMIT " + str(limit)
        if chain_query and len(self.query) > 0:
            wildcard = re.escape(self.CHAIN_QUERY_WILDCARD)
            self.query = re.sub(wildcard, self.query, query)
        else:
            self.query = query
        return self
    
    def set_udf_query(self, udf_query):
        self.udf_query = udf_query
        return self

    def select_from_table(self, table_id, filter="", limit=0, chain_query=False, column_names=["*"]):
        column_list = ",".join(column_names)
        query = f"SELECT {column_list} FROM `{table_id}`"
        if len(filter) > 0:
            query += f" WHERE {filter}"
        return self.set_query(query, limit, chain_query)
    
    def preview_table(self, table_id, limit):
        # Construct a TableReference object 
        table_ref = self.client.get_table(table_id)

        # Fetch a limited number of rows (default is 100)
        rows_iterator = self.client.list_rows(table_ref, max_results=limit)  # Adjust max_results as needed

        # Convert the rows to a DataFrame
        rows_df = rows_iterator.to_dataframe()
        return rows_df

    def select_sequences(self, filter="", limit=0, chain_query=False):
        table_id = f"{self.BQ_INPUT_DATASET_NAME}.{self.sequences_table_name}"
        return self.select_from_table(table_id, filter, limit, chain_query)
    
    def preview_sequences(self, limit=10):
        table_id = f"{self.BQ_INPUT_DATASET_NAME}.{self.sequences_table_name}"
        return self.preview_table(table_id, limit)

    def select_patterns(self, filter="", limit=0, chain_query=False):
        table_id = f"{self.BQ_INPUT_DATASET_NAME}.{self.patterns_table_name}"
        return self.select_from_table(table_id, filter, limit, chain_query)
    
    def preview_patterns(self, limit=10):
        table_id = f"{self.BQ_INPUT_DATASET_NAME}.{self.patterns_table_name}"
        return self.preview_table(table_id, limit)
    
    def select_positions(self, filter="", limit=0, chain_query=False):
        table_id = f"{self.BQ_INPUT_DATASET_NAME}.{self.positions_table_name}"
        return self.select_from_table(table_id, filter, limit, chain_query)
    
    def preview_positions(self, limit=10):
        table_id = f"{self.BQ_INPUT_DATASET_NAME}.{self.positions_table_name}"
        return self.preview_table(table_id, limit)
    
    def select_all_patterns_positions(self, source_override="", limit=0, chain_query=False):
        patterns_source = f"`{self.BQ_INPUT_DATASET_NAME}.{self.patterns_table_name}`"
        if chain_query:
            patterns_source = f"({self.CHAIN_QUERY_WILDCARD})"
        if len(source_override) > 0:
            patterns_source = f"`{self.BQ_STAGE_DATASET_NAME}.{source_override}`"
        positions_source = f"{self.BQ_INPUT_DATASET_NAME}.{self.positions_table_name}"
        agg_positions_query = f'''
            SELECT
                protein_id,
                pattern_id,
                ARRAY_AGG(position) AS starting_positions
            FROM
                `{positions_source}`
            GROUP BY
            1,
            2
        '''
        query = f'''
            SELECT 
                patterns.pattern_id,
                patterns.type,
                patterns.pattern,
                patterns.instances,
                positions.protein_id,
                positions.starting_positions
            FROM {patterns_source} as patterns 
            INNER JOIN ({agg_positions_query}) as positions 
            ON patterns.pattern_id = positions.pattern_id
        '''
        self.set_query(query, limit, chain_query)
        return self

    def select_all_sequence_mrs(self, patterns_source_override="", limit=0, chain_query=False):
        self.select_all_patterns_positions(patterns_source_override, chain_query)
        query = f'''
            WITH full_seq_mrs AS (
                SELECT 
                    sequences.family_name as family_name,
                    sequences.name as name,
                    sequences.sequence as sequence,
                    pattern_positions.pattern as pattern,
                    pattern_positions.starting_positions
                FROM 
                    `{self.BQ_INPUT_DATASET_NAME}.{self.sequences_table_name}` as sequences
                LEFT JOIN
                    ({self.CHAIN_QUERY_WILDCARD}) as pattern_positions
                ON sequences.id = pattern_positions.protein_id
            )

            SELECT
                family_name AS sequence_family_name,
                name AS sequence_name,
                sequence,
                ARRAY_AGG(STRUCT(pattern AS pattern, starting_positions AS starting_positions)) AS pattern_positions
            FROM
                full_seq_mrs
            GROUP BY 1,2,3
        '''
        self.set_query(query, limit, chain_query=True)
        print(self.query)
        return self
    
    def select_bioword_partition(self, joined_mrs_source_table_name, limit=0, row_size_limit=99000000):
        udf_file_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'bq_udf_compute_partition.sql'))
        with open(udf_file_path, 'r') as f:
            udf_query = f.read()
            
        query = f'''
            SELECT
            joined_patterns.sequence_family_name AS sequence_family_name,
            family_types.family_type AS sequence_family_type,
            joined_patterns.sequence_name AS sequence_name,
            joined_patterns.sequence AS sequence,
            SUBSTR(ARRAY_TO_STRING(compute_partition(sequence,
                pattern_positions), " "), 1, {str(row_size_limit)}) as word_partition
            FROM `{self.BQ_STAGE_DATASET_NAME}.{joined_mrs_source_table_name}` AS joined_patterns
            INNER JOIN
                `{self.BQ_INPUT_DATASET_NAME}.{self.family_types_table_name}` as family_types
            ON joined_patterns.sequence_family_name = family_types.family_name
            '''
        self.set_udf_query(udf_query).set_query(query, limit, chain_query=False)
        return self