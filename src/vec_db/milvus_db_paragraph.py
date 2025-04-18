"""
milvus_db_paragraph.py

Milvus-based vector database implementation for paragraphs.
This module creates a separate collection (table) for paragraph records.
It divides data into partitions (one per unique summary record ID) for efficient searches.
It includes methods to ingest paragraphs from an ebook TXT file,
retrieve similar paragraphs, and perform insert, delete, and update operations.
"""

import re
import os
import yaml
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


class MilvusParagraphDatabase(VectorDatabase):
    """
    Milvus-based vector database for paragraph records.
    This class manages paragraph records, allowing for ingestion,
    retrieval, and management of paragraph data."""

    def __init__(
        self, logger: object, cfg: Dict[str, Any], embedding_function, client=None
    ) -> None:
        """
        Initialize the MilvusParagraphDatabase.

        Args:
            logger (object): Logger instance.
            cfg (dict): Configuration dictionary.
            embedding_function: Function to generate embeddings.
            client: Optional injected client.
        """
        super().__init__(logger, cfg)
        # Use collection name from config; if paragraph_collection_name not provided,
        # default to collection_name from config appended with '_paragraph'
        default_collection = cfg.get("collection_name", "default_collection")
        self.collection_name = cfg.get(
            "paragraph_collection_name", f"{default_collection}_paragraph"
        )
        self.dimension = cfg.get("vector_dim", 768)
        self.embedding_function = embedding_function
        self.cfg = cfg
        if client:
            self.client = client
        else:
            self.connect_db()
        self.init_db()
        self.logger.info(
            f"MilvusParagraphDatabase initialized with collection '{self.collection_name}'."
        )

    def connect_db(self):
        """
        Establish connection to Milvus for the paragraph collection.
        """
        host = self.cfg.get("milvus", {}).get("host", "localhost")
        port = self.cfg.get("milvus", {}).get("port", "19530")
        connections.connect(alias="default", host=host, port=port)
        self.logger.info(
            "Milvus connection established (alias 'default') for paragraph."
        )

    def init_db(self) -> None:
        """
        Initialize the paragraph collection in Milvus.
        """
        try:
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.INT64,
                        is_primary=True,
                        description="Unique identifier for each paragraph record",
                    ),
                    FieldSchema(
                        name="vector",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.dimension,
                        description="Embedding vector generated from the paragraph text",
                    ),
                    FieldSchema(
                        name="text",
                        dtype=DataType.VARCHAR,
                        max_length=65535,
                        description="The paragraph content",
                    ),
                    FieldSchema(
                        name="chapter",
                        dtype=DataType.VARCHAR,
                        max_length=50,
                        description="Unique summary record ID associated with the paragraph",
                    ),
                ]
                schema = CollectionSchema(
                    fields, description="Schema for paragraph collection"
                )
                collection = Collection(name=self.collection_name, schema=schema)
                self.logger.info(
                    f"Collection '{self.collection_name}' created for paragraph."
                )
            else:
                collection = Collection(self.collection_name)
                self.logger.info(
                    f"Collection '{self.collection_name}' loaded for paragraph."
                )
            self.client = collection
            if not self.client.has_index(field_name="vector"):
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                }
                self.client.create_index(field_name="vector", index_params=index_params)
                self.logger.info("Index on vector field created for paragraph.")
            else:
                self.logger.info("Index on vector field already exists for paragraph.")
        except MilvusException as e:
            self.logger.error(f"Paragraph collection initialization failed: {e}")
            raise RuntimeError(
                f"Error initializing paragraph collection '{self.collection_name}'."
            ) from e

    def is_collection_empty(self) -> bool:
        """
        Check if the paragraph collection is empty.

        Returns:
            bool: True if the collection is empty, False otherwise.
        """

        try:
            num = self.client.num_entities
            self.logger.info(
                f"Paragraph collection '{self.collection_name}' has {num} entities."
            )
            return num == 0
        except MilvusException as e:
            self.logger.error(f"Error checking paragraph collection stats: {e}")
            raise RuntimeError(
                f"Failed to retrieve paragraph collection stats for '{self.collection_name}'."
            ) from e

    def _get_partition_name(self, summary_id: str) -> str:
        """
        Determine and return the partition name for a given summary ID.

        Args:
            summary_id (str): The summary record ID.
        Returns:
            str: The partition name.
        """
        # Use a prefix that clearly indicates it's a summary partition.
        partition_prefix = self.cfg.get("partition_prefix", "summary_")
        partition_name = f"{partition_prefix}{summary_id}"
        if not utility.has_partition(self.collection_name, partition_name):
            self.client.create_partition(partition_name)
            self.logger.info(f"Created partition: {partition_name}")
        return partition_name

    def ingest_by_paragraph(self, ebook_file_path: str) -> None:
        """Ingest paragraphs from a file into the Milvus collection."""
        if not self.is_collection_empty():
            self.logger.info(
                "Paragraph collection already contains data; skipping ingestion."
            )
            return
        try:
            with open(ebook_file_path, "r", encoding="utf-8") as ebook_file:
                ebook_content = ebook_file.read().strip()
            paragraphs = [p.strip() for p in ebook_content.split("\n\n") if p.strip()]
        except Exception as e:
            self.logger.error("Error reading ebook file.")
            raise RuntimeError("Failed to read ebook file") from e

        # Instead of using a default chapter "0", we start with None.
        current_summary_id = None
        # Pattern to detect chapter header (e.g., "CHAPTER 1")
        chapter_header_pattern = re.compile(r"CHAPTER\s+(\d+)", re.IGNORECASE)
        annotated_paragraphs = []
        for para in paragraphs:
            match = chapter_header_pattern.search(para)
            if match:
                # The matched number is used as the unique summary record ID.
                current_summary_id = match.group(1)
                self.logger.info(
                    f"Detected chapter header; setting summary record ID to {current_summary_id}"
                )
                continue
            # Only annotate paragraphs when a summary ID has been set.
            if current_summary_id is not None:
                annotated_paragraphs.append(
                    {"paragraph": para, "chapter": current_summary_id}
                )
            else:
                # If no chapter header was encountered yet, assign a default ID (e.g., "0")
                annotated_paragraphs.append({"paragraph": para, "chapter": "0"})
        try:
            paragraphs_df = pd.DataFrame(annotated_paragraphs)
            start_id = self.get_max_id() + 1
            paragraphs_df["record_id"] = range(start_id, start_id + len(paragraphs_df))
        except Exception as e:
            self.logger.error("Error preparing paragraph data for ingestion.")
            raise RuntimeError("Failed to prepare paragraph data for ingestion.") from e

        paragraphs_df.apply(lambda row: self._process_paragraph_row(row), axis=1)
        self.logger.info("Ingestion complete: paragraphs ingested.")

    def get_max_id(self) -> int:
        """Retrieve the maximum paragraph record ID from the collection."""
        try:
            self.client.load()
            res = self.client.query(expr="id >= 0", output_fields=["id"])
            max_id = max(item["id"] for item in res) if res and len(res) > 0 else 0
            self.logger.info(f"Current max paragraph record ID is {max_id}")
            return max_id
        except MilvusException as e:
            self.logger.error(f"Error retrieving max paragraph ID: {e}")
            raise RuntimeError("Failed to get the max paragraph ID") from e

    def _process_paragraph_row(self, row: pd.Series) -> None:
        """Process a single row of the DataFrame to extract paragraph and metadata."""
        para = row["paragraph"]
        # Here, the 'chapter' field from the annotated data now represents the unique summary record ID.
        summary_id = row["chapter"]
        vector = self.embedding_function(para)
        metadata = {"id": row["record_id"], "text": para, "chapter": str(summary_id)}
        self.add_vector(vector, metadata)

    def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> None:
        try:
            self.logger.info(f"Inserting paragraph vector with metadata: {metadata}")
            summary_id = metadata.get("chapter", "")
            partition_name = None
            if summary_id:
                partition_name = self._get_partition_name(summary_id)
            data = [
                {
                    "id": metadata["id"],
                    "vector": vector,
                    "text": metadata["text"],
                    "chapter": summary_id,
                }
            ]
            if partition_name:
                self.client.insert(data=data, partition_name=partition_name)
            else:
                self.client.insert(data=data)
            self.logger.info(
                f"Paragraph vector with ID {metadata['id']} added successfully."
            )
        except MilvusException as e:
            self.logger.error(f"Failed to add paragraph vector: {e}")
            raise RuntimeError(
                f"Error adding paragraph vector with ID {metadata['id']} to '{self.collection_name}'"
            ) from e

    def retrieve_naive_search(
        self, query: str, top_k: int = 2
    ) -> Union[List[Dict[str, Any]], None]:
        """Naive search through all paragraphs without filtering."""
        try:
            self.client.load()
            query_vector = self.embedding_function(query)
            results = self.client.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["id", "chapter", "text"],
            )
            if not results or len(results[0]) == 0:
                self.logger.info("No matching paragraphs found.")
                return None

            final_paragraphs = []
            for hit in results[0]:
                record_id = hit.entity.get("id")
                summary_id = hit.entity.get("chapter")
                text = hit.entity.get("text")
                # self.logger.info(f"Retrieved paragraph: ID={record_id}, SummaryID={summary_id}")
                final_paragraphs.append(
                    {"id": record_id, "chapter": summary_id, "text": text}
                )
            return final_paragraphs
        except MilvusException as e:
            self.logger.exception(f"Naive search failed: {e}")
            raise RuntimeError(f"Error retrieving paragraphs for query: {query}") from e

    def retrieve_abstracted_search(
        self, query: str, summary_ids: List[str], top_k: int = 2
    ) -> Union[List[Dict[str, Any]], None]:
        """
        Search using partitions corresponding to provided summary record IDs.
        This version ensures we skip invalid partition names (e.g., negative numbers).
        """
        try:
            self.client.load()
            query_vector = self.embedding_function(query)
            partition_prefix = self.cfg.get("partition_prefix", "summary_")

            valid_partition_names = []
            for sid in summary_ids:
                try:
                    # Convert the summary ID (1-based) to the paragraph's chapter ID (0-based)
                    converted_id = int(sid) - 1
                    if converted_id < 0:
                        continue  # Skip invalid IDs
                    partition_name = f"{partition_prefix}{converted_id}"
                    if utility.has_partition(self.collection_name, partition_name):
                        valid_partition_names.append(partition_name)
                    else:
                        self.logger.warning(
                            f"Partition {partition_name} not found, skipping it."
                        )
                except ValueError:
                    self.logger.warning(
                        f"Invalid summary ID '{sid}' during abstracted search."
                    )

            if not valid_partition_names:
                self.logger.info("No valid partitions found for abstracted search.")
                return None

            results = self.client.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                partition_names=valid_partition_names,
                limit=top_k,
                output_fields=["id", "chapter", "text"],
            )

            if not results or len(results[0]) == 0:
                self.logger.info("No matching paragraphs found.")
                return None

            final_paragraphs = []
            for hit in results[0]:
                record_id = hit.entity.get("id")
                chap = hit.entity.get("chapter")
                text = hit.entity.get("text")
                # self.logger.info(f"Retrieved paragraph: ID={record_id}, Chapter={chap}")
                final_paragraphs.append(
                    {"id": record_id, "chapter": chap, "text": text}
                )
            return final_paragraphs

        except MilvusException as e:
            self.logger.exception(f"Abstracted search failed: {e}")
            raise RuntimeError(f"Error retrieving paragraphs for query: {query}") from e

    def export_to_txt(self, output_file: str) -> None:
        """
        Export all paragraph records to a text file.
        """
        try:
            self.client.load()
            records = self.client.query(
                expr="id >= 0", output_fields=["id", "text", "chapter"]
            )
            if not records:
                self.logger.info("No records found in the paragraph collection.")
                return
            with open(output_file, "w", encoding="utf-8") as f:
                for record in records:
                    line = f"ID: {record['id']}, SummaryID: {record.get('chapter','')}, Text: {record['text'][:10000]}...\n"
                    f.write(line)
            self.logger.info(f"Paragraph data exported to {output_file}.")
        except Exception as e:
            self.logger.error(f"Error exporting paragraph data: {e}")
            raise RuntimeError("Failed to export paragraph data") from e

    def insert_data(self, ebook_file_path: str) -> None:
        """
        Insert paragraphs from a file into the Milvus collection."""
        if not os.path.exists(ebook_file_path):
            self.logger.error(f"File {ebook_file_path} does not exist.")
            return
        current_max_id = self.get_max_id()
        try:
            with open(ebook_file_path, "r", encoding="utf-8") as ebook_file:
                ebook_content = ebook_file.read().strip()
            paragraphs = [p.strip() for p in ebook_content.split("\n\n") if p.strip()]
        except Exception as e:
            self.logger.exception("Error reading ebook file.")
            raise RuntimeError("Failed to read ebook file") from e

        current_summary_id = None
        chapter_header_pattern = re.compile(r"CHAPTER\s+(\d+)", re.IGNORECASE)
        annotated_paragraphs = []
        for para in paragraphs:
            match = chapter_header_pattern.search(para)
            if match:
                # When a chapter header is encountered, set the current summary ID.
                current_summary_id = match.group(1)
                self.logger.info(
                    f"Detected chapter header; setting summary record ID to {current_summary_id}"
                )
                continue
            # Use the current summary ID if available; otherwise assign "0"
            if current_summary_id is not None:
                annotated_paragraphs.append(
                    {"paragraph": para, "chapter": current_summary_id}
                )
            else:
                annotated_paragraphs.append({"paragraph": para, "chapter": "0"})
        try:
            paragraphs_df = pd.DataFrame(annotated_paragraphs)
            paragraphs_df["record_id"] = range(
                current_max_id + 1, current_max_id + 1 + len(paragraphs_df)
            )
        except Exception as e:
            self.logger.exception("Error preparing paragraph data for insertion.")
            raise RuntimeError("Failed to prepare paragraph data for insertion.") from e

        paragraphs_df.apply(lambda row: self._process_paragraph_row(row), axis=1)
        self.logger.info(f"Inserted new paragraphs from '{ebook_file_path}'")

    def delete_paragraphs_by_summary(self, summary_id: str) -> None:
        """Delete all paragraph records associated with a given summary ID.
        The stored chapter value is assumed to be (summary_id - 1)."""
        try:
            # Convert the summary ID (as stored in the summaries, starting at 1) to the value stored in paragraphs.
            converted_id = str(int(summary_id) - 1)
            self.client.load()
            expr = f"chapter == '{converted_id}'"
            self.client.delete(expr=expr)
            self.logger.info(
                f"Deleted all paragraph records with chapter (converted SummaryID) {converted_id} (from summary ID {summary_id})."
            )
            print(
                f"Deleted all paragraph records with chapter {converted_id} (converted from summary ID {summary_id})."
            )
        except Exception as e:
            self.logger.error(
                f"Error deleting paragraphs for summary ID {summary_id}: {e}"
            )
            raise RuntimeError("Error deleting paragraphs by summary ID") from e

    def delete_data(self) -> None:
        """
        Prompt the user to delete either an entire chapter's paragraphs or specific paragraphs.
        """
        mode = (
            input(
                "Do you want to delete an entire chapter's paragraphs or specific paragraphs? (Enter 'chapter' or 'paragraph'): "
            )
            .strip()
            .lower()
        )
        if mode == "chapter":
            summary_id = input(
                "Enter the summary (chapter) ID to delete paragraphs for: "
            ).strip()
            try:
                self.delete_paragraphs_by_summary(summary_id)
            except Exception as e:
                self.logger.error(f"Error during chapter deletion: {e}")
        elif mode == "paragraph":
            record_ids = input(
                "Enter the paragraph record IDs to delete (comma separated): "
            ).strip()
            try:
                id_list = [
                    int(rid.strip()) for rid in record_ids.split(",") if rid.strip()
                ]
            except ValueError:
                print("Invalid input. Record IDs must be integers.")
                self.logger.error("Invalid input for paragraph deletion.")
                return
            try:
                self.client.load()
                expr = f"id in [{','.join(str(x) for x in id_list)}]"
                self.client.delete(expr=expr)
                self.logger.info(f"Deleted paragraph records with IDs: {id_list}")
                print(f"Deleted paragraph records with IDs: {id_list}")
            except Exception as e:
                self.logger.error(f"Error deleting paragraph records: {e}")
                raise RuntimeError("Error deleting paragraph records") from e
        else:
            print("Invalid deletion mode. Deletion cancelled.")
            self.logger.info("Deletion cancelled due to invalid mode.")

    def update_data(self) -> None:
        """
        Update an existing paragraph record.
        The user is prompted for the record ID and new text.
        The old record is deleted and a new record is inserted with the same ID.
        """
        record_id_str = input(
            "Enter the paragraph record ID to update (or '0' to cancel): "
        ).strip()
        if record_id_str == "0":
            self.logger.info("Paragraph update cancelled by user.")
            return
        try:
            record_id = int(record_id_str)
        except ValueError:
            print("Invalid input. Record ID must be an integer.")
            self.logger.error(
                "Paragraph update cancelled due to non-integer record ID."
            )
            return
        try:
            self.client.load()
            records = self.client.query(
                expr=f"id == {record_id}", output_fields=["id", "text", "chapter"]
            )
            if not records:
                print(f"No paragraph record found with ID {record_id}.")
                self.logger.info(
                    f"No paragraph record found with ID {record_id}. Update aborted."
                )
                return
            old_record = records[0]
            old_text = old_record["text"]
            summary_id = old_record["chapter"]
            self.logger.info(
                f"Found paragraph record {record_id} with chapter {summary_id}."
            )
        except Exception as e:
            self.logger.error(f"Error retrieving paragraph record {record_id}: {e}")
            print("Error retrieving the paragraph record. Update aborted.")
            return
        print(f"Current text:\n{old_text}\n")
        new_text = input("Enter the UPDATED paragraph text: ").strip()
        if not new_text:
            print("No new text provided. Update cancelled.")
            self.logger.info("Paragraph update cancelled due to empty text.")
            return
        try:
            self.client.delete(expr=f"id == {record_id}")
            self.logger.info(f"Deleted paragraph record {record_id} for update.")
        except Exception as e:
            self.logger.error(
                f"Error deleting paragraph record {record_id} for update: {e}"
            )
            print("Error deleting old paragraph record. Update aborted.")
            return
        try:
            new_vector = self.embedding_function(new_text)
            metadata = {"id": record_id, "text": new_text, "chapter": summary_id}
            self.add_vector(new_vector, metadata)
            self.logger.info(f"Inserted updated paragraph record with ID {record_id}.")
            print(f"Paragraph record {record_id} updated successfully!")
        except Exception as e:
            self.logger.error(
                f"Error inserting updated paragraph record {record_id}: {e}"
            )
            print("Error inserting updated paragraph record. Update incomplete.")
            return
