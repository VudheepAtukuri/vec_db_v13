"""
milvus_db_csv.py

A Milvus-based CSV ingestion module that reads CSV files via Pandas,
creates an embedding from each row (using a customizable strategy),
and stores the complete row (as a JSON string) in the text field.

This module is built on top of the Milvus Collection API and a base VectorDatabase class.
"""

import json
import pandas as pd
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from pymilvus.exceptions import MilvusException
from src.vec_db.base import VectorDatabase


class MilvusCSVDatabase(VectorDatabase):
    """
    A Milvus-based database wrapper for ingesting CSV data.
    Reads a CSV file via Pandas, generates an embedding for each row,
    and stores the row (serialized as JSON) in the 'text' field.
    """

    def __init__(self, logger: object, cfg: dict, embedding_function):
        """
        Initialize the CSV database instance.

        Args:
            logger (object): Logger instance.
            cfg (dict): Configuration dictionary.
            embedding_function (callable): Function that takes a text string and returns a vector.
        """
        super().__init__(logger, cfg)
        self.collection_name = cfg.get("collection_name", "csv_data_collection")
        self.dimension = cfg.get("dimension", 768)
        self.unique_key_columns = cfg.get(
            "unique_key_columns", ["StateAbbr", "CountyName"]
        )
        self.embedding_function = embedding_function
        self.cfg = cfg
        self.connect_db()
        self.init_db()
        self.logger.info(
            f"Connected to Milvus at {cfg.get('milvus', {}).get('host', 'localhost')}:{cfg.get('milvus', {}).get('port', '19530')}"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from pymilvus import connections
        connections.disconnect(alias="default")
        self.logger.info("Milvus connection disconnected (alias 'default').")


    def connect_db(self):
        """
        Establish a connection to the Milvus server using configuration parameters.
        """
        host = self.cfg.get("milvus", {}).get("host", "localhost")
        port = self.cfg.get("milvus", {}).get("port", "19530")
        connections.connect(alias="default", host=host, port=port)
        self.logger.info("Milvus connection established (alias 'default').")

    def init_db(self) -> None:
        """
        Initialize the Milvus collection for CSV data.
        If the collection does not exist, it is created with the appropriate schema.
        Otherwise, it is loaded.
        Then, create an index on the 'vector' field if it does not exist.
        """
        try:
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(
                        name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
                    ),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                ]
                schema = CollectionSchema(fields, description="CSV collection")
                collection = Collection(name=self.collection_name, schema=schema)
                self.logger.info(
                    f"Collection '{self.collection_name}' created in Milvus."
                )
            else:
                collection = Collection(self.collection_name)
                self.logger.info(
                    f"Collection '{self.collection_name}' loaded from Milvus."
                )
            self.client = collection

            if not self.client.has_index(field_name="vector"):
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                }
                self.client.create_index(field_name="vector", index_params=index_params)
                self.logger.info("Index on vector field created.")
            else:
                self.logger.info("Index on vector field already exists.")
        except MilvusException as e:
            self.logger.error(f"Milvus collection initialization failed: {e}")
            raise RuntimeError("Error initializing Milvus collection") from e
        except Exception as e:
            self.logger.error(f"Unexpected error initializing collection: {e}")
            raise RuntimeError("Unexpected error initializing collection") from e

    def is_collection_empty(self) -> bool:
        """
        Check if the Milvus collection is empty.
        """
        try:
            num = self.client.num_entities
            self.logger.info(f"Collection '{self.collection_name}' has {num} entities.")
            return num == 0
        except MilvusException as e:
            self.logger.error(f"Error checking collection stats: {e}")
            raise RuntimeError("Error checking collection stats") from e

    def ingest_from_file(
        self, csv_file_path: str, embedding_columns: list = None
    ) -> None:
        """
        Ingest a CSV file into Milvus.

        Args:
            csv_file_path (str): Path to the CSV file.
            embedding_columns (list): Optional list of column names to be used for generating
                                      the embedding text. If not provided, the entire row is used.
        """
        if not self.is_collection_empty():
            self.logger.info(
                "CSV collection already contains data. Skipping CSV ingestion."
            )
            return
        try:
            df = pd.read_csv(csv_file_path)
            self.logger.info(f"Loaded CSV file: {csv_file_path} with {len(df)} rows.")
        except Exception as e:
            self.logger.error(f"Failed to load CSV file {csv_file_path}: {e}")
            raise RuntimeError(f"Failed to load CSV file '{csv_file_path}'") from e

        try:
            for idx, row in df.iterrows():
                record_id = idx + 1  # Generate a unique ID
                if embedding_columns:
                    embed_text_input = " ".join(
                        [str(row[col]) for col in embedding_columns if col in row]
                    )
                else:
                    embed_text_input = ", ".join(
                        [f"{col}: {row[col]}" for col in df.columns]
                    )
                vector = self.embedding_function(embed_text_input)
                metadata = row.to_dict()
                metadata["id"] = record_id
                unique_key = "-".join(
                    [str(row[col]) for col in self.unique_key_columns]
                )
                metadata["unique_key"] = unique_key
                self.add_vector(vector, metadata)
            self.logger.info("CSV ingestion complete.")
        except Exception as e:
            self.logger.error(f"Error during CSV ingestion: {e}")
            raise RuntimeError("Error during CSV ingestion") from e

    def add_vector(self, vector: list, metadata: dict) -> None:
        """
        Insert a vector with its metadata into Milvus.
        The metadata is serialized as a JSON string and stored in the 'text' field.

        Args:
            vector (list): The embedding vector.
            metadata (dict): The full row data.
        """
        try:
            json_metadata = json.dumps(metadata)
            data = [{"id": metadata["id"], "vector": vector, "text": json_metadata}]
            self.client.insert(data=data)
            self.logger.info(f"Inserted vector with ID: {metadata['id']}")
        except MilvusException as e:
            self.logger.error(f"Milvus error inserting vector: {e}")
            raise RuntimeError("Milvus error inserting vector") from e

    def query_loop(self, top_k: int = 2) -> None:
        """
        Start an interactive query loop for CSV data.

        Args:
            top_k (int): Number of top similar records to retrieve.
        """
        while True:
            user_query = input("\nEnter a CSV query (or type 'exit' to quit): ").strip()
            if user_query.lower() == "exit":
                self.logger.info("Exiting CSV query loop.")
                break
            results = self.retrieve_similar_records(query=user_query, top_k=top_k)
            if results:
                print("\nTop similar CSV records:")
                for idx, rec in enumerate(results, start=1):
                    print(f"{idx}. Unique Key: {rec.get('unique_key')}")
                    print(f"    Full Metadata: {rec}")
            else:
                print("\nNo similar CSV records found.")

    def retrieve_similar_records(self, query: str, top_k: int = 2) -> list:
        """
        Retrieve similar records from Milvus given a query.

        Args:
            query (str): The query text.
            top_k (int): Number of similar records to return.

        Returns:
            list: A list of metadata dictionaries for the similar records.
        """
        try:
            query_vector = self.embedding_function(query)
            results = self.client.search(
                data=[query_vector],
                anns_field="vector",  # specify the field containing vectors
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["text"],
            )
            similar_records = []
            for hit in results[0]:
                metadata = json.loads(hit.entity.get("text"))
                similar_records.append(metadata)
            return similar_records
        except MilvusException as e:
            self.logger.error(f"Milvus error retrieving similar records: {e}")
            raise RuntimeError("Error retrieving similar records") from e

    def get_max_id(self) -> int:
        """
        Get the maximum existing record ID from the collection.

        Returns:
            int: The maximum record ID, or 0 if the collection is empty.
        """
        try:
            res = self.client.query(
                "id >= 0",
                output_fields=["id"],
            )
            self.logger.info(f"Query result for max ID: {res}")
            if res and len(res) > 0:
                max_id = max(item["id"] for item in res)
            else:
                max_id = 0
            return max_id
        except MilvusException as e:
            self.logger.error(f"Error getting max ID: {e}")
            raise RuntimeError("Error retrieving max ID") from e

    def insert_data(
        self, csv_file_path: str = None, embedding_columns: list = None
    ) -> None:
        """
        Insert new data (additional rows) into the existing CSV collection.

        Args:
            csv_file_path (str): Path to the new CSV file.
            embedding_columns (list): Optional list of columns
                 to use for generating the embedding text.
        """
        if csv_file_path is None:
            csv_file_path = input(
                "Enter the path to the new CSV file (or type 'exit' to cancel): "
            ).strip()
            if csv_file_path.lower() == "exit":
                self.logger.info("Insert new data cancelled by user.")
                return

        try:
            new_df = pd.read_csv(csv_file_path)
            self.logger.info(
                f"Loaded new CSV data from {csv_file_path} with {len(new_df)} rows."
            )
        except Exception as e:
            self.logger.error(f"Failed to load new CSV data: {e}")
            raise RuntimeError(
                f"Failed to load new CSV data from '{csv_file_path}'"
            ) from e

        for col in self.unique_key_columns:
            if col not in new_df.columns:
                self.logger.error(
                    f"Column '{col}' is missing in the new CSV file. Aborting insertion."
                )
                return

        current_max_id = self.get_max_id()
        self.logger.info(f"Current max ID in collection: {current_max_id}")

        for idx, row in new_df.iterrows():
            new_id = current_max_id + idx + 1
            if embedding_columns:
                embed_text_input = " ".join(
                    [str(row[col]) for col in embedding_columns if col in row]
                )
            else:
                embed_text_input = ", ".join(
                    [f"{col}: {row[col]}" for col in new_df.columns]
                )
            vector = self.embedding_function(embed_text_input)
            metadata = row.to_dict()
            metadata["id"] = new_id
            unique_key = "-".join([str(row[col]) for col in self.unique_key_columns])
            metadata["unique_key"] = unique_key
            self.add_vector(vector, metadata)

        self.logger.info("New CSV data inserted successfully.")
        self.log_collection_state()

    def delete_data(self, record_ids: str = None) -> None:
        """
        Delete one or more records from the CSV collection based on record IDs.

        Args:
            record_ids (str): A comma-separated string of record IDs to delete.
        """
        if record_ids is None:
            record_ids = input(
                "Enter the record IDs to delete (comma separated, or type 0 to cancel): "
            ).strip()
            if record_ids == "0":
                self.logger.info("Deletion cancelled by user.")
                return

        try:
            id_list = [int(rid.strip()) for rid in record_ids.split(",") if rid.strip()]
        except ValueError:
            self.logger.error(
                "Invalid input. Record IDs must be integers, separated by commas."
            )
            return

        try:
            expr = f"id in [{','.join(str(x) for x in id_list)}]"
            self.client.delete(expr=expr)
            self.logger.info(f"Deleted records with IDs: {id_list}")
        except Exception as e:
            self.logger.error(f"Failed to delete records with IDs {id_list}: {e}")
            raise RuntimeError(f"Failed to delete records with IDs {id_list}") from e

        self.log_collection_state()

    def update_from_csv(
        self, csv_file_path: str, embedding_columns: list = None
    ) -> None:
        """
        Update existing CSV records using data from a CSV file.

        The CSV file must have an 'ID' column identifying which record to update.
        For each row in the CSV, the existing record with that ID will be deleted and
        replaced with the updated data from the CSV row.
        """
        try:
            updated_df = pd.read_csv(csv_file_path)
            self.logger.info(
                f"Loaded updated CSV data from {csv_file_path} with {len(updated_df)} rows."
            )
        except Exception as e:
            self.logger.error(f"Failed to load updated CSV data: {e}")
            raise RuntimeError(
                f"Failed to load updated CSV data from '{csv_file_path}'"
            ) from e

        for idx, row in updated_df.iterrows():
            try:
                record_id = int(row["ID"])
            except Exception:
                self.logger.error(
                    "Missing or invalid 'ID' in updated CSV row. Skipping row."
                )
                raise RuntimeError(
                    "Missing or invalid 'ID' in updated CSV row."
                ) from None

            updated_row = row.to_dict()
            if embedding_columns:
                embed_text_input = " ".join(
                    [str(row[col]) for col in embedding_columns if col in row]
                )
            else:
                embed_text_input = ", ".join([f"{col}: {row[col]}" for col in row])
            vector = self.embedding_function(embed_text_input)
            updated_row["id"] = record_id
            unique_key = "-".join(
                [str(row[col]) for col in self.unique_key_columns if col in row]
            )
            updated_row["unique_key"] = unique_key

            try:
                self.client.delete(expr=f"id == {record_id}")
                self.logger.info(f"Deleted record with ID {record_id} for update.")
            except Exception as e:
                self.logger.error(
                    f"Failed to delete record {record_id} for update: {e}"
                )
                raise RuntimeError("Failed to delete record for update") from e

            try:
                self.add_vector(vector, updated_row)
                self.logger.info(f"Updated record with ID {record_id} successfully.")
            except Exception as e:
                self.logger.error(f"Failed to insert updated record {record_id}: {e}")
                raise RuntimeError("Failed to insert updated record") from e

        self.log_collection_state()

    def log_collection_state(self) -> None:
        """
        Log the current collection statistics.
        """
        try:
            records = self.client.query(
                "id >= 0",
                output_fields=["id", "text"],
            )
            if records and len(records) > 0:
                self.logger.info(
                    f"Current collection state: {len(records)} records found."
                )
                for record in records:
                    self.logger.info(f"ID: {record['id']}, Text: {record['text']}")
            else:
                self.logger.info("Collection is empty.")
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            raise RuntimeError("Error getting collection stats") from e

    def export_to_csv(self, output_file: str) -> None:
        """
        Export the entire CSV collection to a text file.
        Each line in the output file will contain the record's ID
          and its metadata as stored in the 'text' field.

        Args:
            output_file (str): Path to the output text file.
        """
        try:
            records = self.client.query(
                "id >= 0",
                output_fields=["id", "text"],
            )
            with open(output_file, "w", encoding="utf-8") as f:
                for record in records:
                    line = f"ID: {record['id']}, Metadata: {record['text']}\n\n"
                    f.write(line)
            self.logger.info(f"Exported CSV collection to {output_file}")
        except Exception as e:
            self.logger.error(f"Error exporting CSV collection to txt: {e}")
            raise RuntimeError("Error exporting CSV collection") from e
