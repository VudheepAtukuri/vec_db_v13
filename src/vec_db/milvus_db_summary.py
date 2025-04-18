"""
milvus_db_summary.py

Milvus-based vector database implementation for chapter summaries.
It implements methods to ingest summaries from a TXT file (by chapter),
retrieve similar summaries (top 2), and perform insert, delete, and update operations.
"""

import yaml
import re
import os
from typing import List, Dict, Any, Union
import pandas as pd
from pymilvus import (
    Collection,
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from pymilvus.exceptions import MilvusException
from src.vec_db.base import VectorDatabase

# Load configuration from YAML file
with open("src/config/config.yaml", "r", encoding="utf-8") as file_handle:
    config = yaml.safe_load(file_handle)


class MilvusSummaryDatabase(VectorDatabase):
    """Summary database class for Milvus.
    This class handles the ingestion, retrieval, and management of chapter summaries
    in a Milvus vector database."""
    
    def __init__(
        self, logger: object, cfg: Dict[str, Any], embedding_function, client=None
    ) -> None:
        """
        Initialize the MilvusSummaryDatabase.

        Args:
            logger (object): Logger instance.
            cfg (dict): Configuration dictionary.
            embedding_function: Function to generate embeddings.
            client: Optional injected client.
        """
        super().__init__(logger, cfg)
        # Use collection name from config; if summary_collection_name not provided,
        # default to collection_name from config appended with '_summary'
        default_collection = cfg.get("collection_name", "default_collection")
        self.collection_name = cfg.get(
            "summary_collection_name", f"{default_collection}_summary"
        )
        self.dimension = cfg.get("vector_dim", 768)
        self.embedding_function = embedding_function
        self.cfg = cfg
        # Use injected client if provided; otherwise, establish our own connection.
        if client:
            self.client = client
        else:
            self.connect_db()
        self.init_db()
        self.logger.info(
            f"MilvusSummaryDatabase initialized with collection '{self.collection_name}'."
        )

    def connect_db(self):
        """Establish a connection to the Milvus database."""
        host = self.cfg.get("milvus", {}).get("host", "localhost")
        port = self.cfg.get("milvus", {}).get("port", "19530")
        connections.connect(alias="default", host=host, port=port)
        self.logger.info("Milvus connection established (alias 'default') for summary.")

    def init_db(self) -> None:
        """Initialize the summary collection in the Milvus database."""
        try:
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        description="Unique identifier for each summary record",
                    ),
                    FieldSchema(
                        name="vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.dimension,
                        description="Embedding vector generated from the summary text",
                    ),
                    FieldSchema(
                        name="text",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                        description="The chapter summary text",
                    ),
                ]
                schema = CollectionSchema(
                    fields, description="Schema for summary collection"
                )
                collection = Collection(name=self.collection_name, schema=schema)
                self.logger.info(
                    f"Collection '{self.collection_name}' created for summary."
                )
            else:
                collection = Collection(self.collection_name)
                self.logger.info(
                    f"Collection '{self.collection_name}' loaded for summary."
                )
            self.client = collection
            if not self.client.has_index(field_name="vector"):
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                }
                self.client.create_index(field_name="vector", index_params=index_params)
                self.logger.info("Index on vector field created for summary.")
            else:
                self.logger.info("Index on vector field already exists for summary.")
        except MilvusException as e:
            self.logger.error(f"Summary collection initialization failed: {e}")
            raise RuntimeError(
                f"Error initializing summary collection '{self.collection_name}'."
            ) from e

    def is_collection_empty(self) -> bool:
        """
        Check if the summary collection is empty.

        Returns:
            bool: True if empty, False otherwise.
        """
        try:
            num = self.client.num_entities
            self.logger.info(
                f"Summary collection '{self.collection_name}' has {num} entities."
            )
            return num == 0
        except MilvusException as e:
            self.logger.error(f"Error checking summary collection stats: {e}")
            raise RuntimeError(
                f"Failed to retrieve summary collection stats for '{self.collection_name}'."
            ) from e

    def _process_summary_row(self, row: pd.Series, chapter_pattern: re.Pattern) -> None:
        """
        Process a single row from the summary DataFrame and insert vector if pattern matches.

        Args:
            row (pd.Series): A row from the DataFrame.
            chapter_pattern (re.Pattern): Regular expression pattern to match chapter lines.
        """
        match = chapter_pattern.search(row["line"])
        if match:
            # Instead of capturing and storing the chapter number, only the summary text is stored.
            summary_text = match.group(2).strip()
            vector = self.embedding_function(summary_text)
            metadata = {
                "id": row["record_id"],
                "text": summary_text,
            }
            self.add_vector(vector, metadata)
        else:
            self.logger.info("Line did not match summary pattern; skipping.")

    def ingest_by_chapter(self, summary_file_path: str) -> None:
        """
        Ingest chapter summaries from a text file into the summary collection.

        Args:
            summary_file_path (str): Path to the summary text file.
        """
        if not self.is_collection_empty():
            self.logger.info(
                "Summary collection already contains data; skipping ingestion."
            )
            return
        try:
            with open(summary_file_path, "r", encoding="utf-8") as s_file:
                summary_lines = s_file.readlines()
            summary_df = pd.DataFrame({"line": summary_lines})
            summary_df["record_id"] = range(1, len(summary_df) + 1)
            chapter_pattern = re.compile(r"CHAPTER\s+(\d+)\s*-\s*(.+)", re.IGNORECASE)
            summary_df.apply(
                lambda row: self._process_summary_row(row, chapter_pattern), axis=1
            )
            self.logger.info(f"Ingested chapter summaries from {summary_file_path}")
        except MilvusException as e:
            self.logger.exception("Error ingesting summary file.")
            raise RuntimeError("Failed to ingest summary file") from e

    def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        Insert a summary vector and its metadata into the summary collection.

        Args:
            vector (List[float]): The embedding vector.
            metadata (dict): Dictionary containing 'id' and 'text'.
        """
        try:
            self.logger.info(f"Inserting summary vector with metadata: {metadata}")
            data = [
                {
                    "id": metadata["id"],
                    "vector": vector,
                    "text": metadata["text"],
                }
            ]
            self.client.insert(data=data)
            self.logger.info(
                f"Summary vector with ID {metadata['id']} added successfully."
            )
        except MilvusException as e:
            self.logger.error(f"Failed to add summary vector: {e}")
            raise RuntimeError(
                f"Error adding summary vector with ID {metadata['id']} to '{self.collection_name}'"
            ) from e

    def retrieve_similar_facts(
        self, query: str, top_k: int = 2
    ) -> Union[List[Dict[str, Any]], None]:
        """
        Retrieve similar chapter summaries.
        Returns a list of dicts with keys 'id' and 'text' so that the unique summary record ID
        can be used for mapping with paragraph-level operations.
        """
        try:
            self.client.load()
            query_vector = self.embedding_function(query)
            results = self.client.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["id", "text"],
            )
            if not results or len(results[0]) == 0:
                self.logger.info("No matching summaries found.")
                return None
            final_results = [
                {"id": hit.entity.get("id"), "text": hit.entity.get("text")}
                for hit in results[0]
            ]
            return final_results
        except MilvusException as e:
            self.logger.exception(f"Summary query failed: {e}")
            raise RuntimeError(
                f"Error retrieving similar summaries for query: {query}"
            ) from e

    def export_to_txt(self, output_file: str) -> None:
        """Export summary collection data to a text file."""

        try:
            self.client.load()
            records = self.client.query(
                expr="id >= 0",
                output_fields=["id", "text"],
            )
            if not records:
                self.logger.info("No records found in the summary collection.")
                return
            with open(output_file, "w", encoding="utf-8") as f:
                for record in records:
                    line = f"ID: {record['id']}, Text: {record['text'][:10000]}...\n"
                    f.write(line)
            self.logger.info(f"Summary data exported to {output_file}.")
        except Exception as e:
            self.logger.error(f"Error exporting summary data: {e}")
            raise RuntimeError("Failed to export summary data") from e

    def insert_data(self, summary_file_path: str) -> None:
        """Insert new summary data from a text file into the summary collection."""
        if not os.path.exists(summary_file_path):
            self.logger.error(f"File {summary_file_path} does not exist.")
            return
        current_max_id = self.get_max_id()
        try:
            with open(summary_file_path, "r", encoding="utf-8") as s_file:
                summary_lines = s_file.readlines()
            summary_df = pd.DataFrame({"line": summary_lines})
            summary_df["record_id"] = range(
                current_max_id + 1, current_max_id + 1 + len(summary_df)
            )
            chapter_pattern = re.compile(r"CHAPTER\s+(\d+)\s*-\s*(.+)", re.IGNORECASE)
            summary_df.apply(
                lambda row: self._process_summary_row(row, chapter_pattern), axis=1
            )
            self.logger.info(f"Inserted new summaries from '{summary_file_path}'")
        except Exception as e:
            self.logger.exception("Error inserting new summary data.")
            raise RuntimeError("Failed to insert summary data") from e

    def get_max_id(self) -> int:       
        """Retrieve the maximum summary record ID in the collection."""

        try:
            self.client.load()
            res = self.client.query(expr="id >= 0", output_fields=["id"])
            max_id = max(item["id"] for item in res) if res and len(res) > 0 else 0
            self.logger.info(f"Current max summary record ID is {max_id}")
            return max_id
        except MilvusException as e:
            self.logger.error(f"Error retrieving max summary ID: {e}")
            raise RuntimeError("Failed to get the max summary ID") from e

    def delete_chapter(self, chapter_id: int) -> None:
        """Delete the chapter summary record from the collection"""
        try:
            self.client.load()
            expr = f"id == {chapter_id}"
            self.client.delete(expr=expr)
            self.logger.info(f"Deleted chapter summary record with ID {chapter_id}.")
            print(f"Deleted chapter summary record {chapter_id}.")
        except Exception as e:
            self.logger.error(
                f"Failed to delete chapter summary record {chapter_id}: {e}"
            )
            raise RuntimeError("Error deleting chapter summary record") from e

    def update_data(self) -> None:
        """
        Update an existing summary record. The user is prompted for the record ID and new text.
        The old record is deleted and a new record is inserted with the same ID.
        """
        try:
            record_id_str = input("Enter the summary record ID to update (or '0' to cancel): ").strip()
            if record_id_str == "0":
                self.logger.info("Summary update cancelled by user.")
                raise RuntimeError("Update cancelled by user.")
            record_id = int(record_id_str)
        except ValueError as exception:
            self.logger.error("Summary update cancelled: non-integer record ID.")
            raise RuntimeError("Invalid input. Record ID must be an integer.") from exception
        try:
            self.client.load()
            records = self.client.query(
                expr=f"id == {record_id}",
                output_fields=["id", "text"],
            )
            if not records:
                self.logger.info(f"No summary record found with ID {record_id}. Update aborted.")
                raise RuntimeError(f"No summary record found with ID {record_id}.")
            old_record = records[0]
            old_text = old_record["text"]
            self.logger.info(f"Found summary record {record_id}")
        except MilvusException as exception:
            self.logger.error(f"Query to find summary record {record_id} failed: {exception}")
            raise RuntimeError("Error retrieving the summary record to update.") from exception
        print(f"Current text:\n{old_text}\n")
        new_text = input("Enter the UPDATED summary text: ").strip()
        if not new_text:
            self.logger.info("Summary update aborted: empty text.")
            raise RuntimeError("No new text given. Update cancelled.")
        try:
            self.client.delete(expr=f"id == {record_id}")
            self.logger.info(f"Deleted summary record {record_id} for update.")
        except Exception as exception:
            self.logger.error(f"Failed to delete summary record {record_id} for update: {exception}")
            raise RuntimeError("Error deleting old summary record. Update aborted.") from exception
        try:
            new_vector = self.embedding_function(new_text)
            metadata = {"id": record_id, "text": new_text}
            self.add_vector(new_vector, metadata)
            self.logger.info(f"Inserted updated summary record with ID {record_id}.")
            print(f"Summary record {record_id} updated successfully!")
        except Exception as exception:
            self.logger.error(f"Failed to insert updated summary record {record_id}: {exception}")
            raise RuntimeError("Error inserting updated summary record. Update incomplete.") from exception