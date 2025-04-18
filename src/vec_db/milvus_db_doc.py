"""
milvus_db_doc.py

Milvus-based implementation of the VectorDatabase for TXT data.
This version is adapted to work with a Milvus instance running in Docker,
and implements chapter-based partitioning to restrict operations to relevant chapters.
"""

import re
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


class MilvusDocumentDatabase(VectorDatabase):
    """
    Milvus-based vector database implementation for text-based data with chapter-level metadata.
    Stores both summary and paragraph records, distinguished by the "record_type" field.
    Uses partitions (one per chapter) for efficient search.
    """

    def __init__(self, logger: object, cfg: Dict[str, Any], embedding_function) -> None:
        """
        Initialize the Milvus TXT database instance.

        Args:
            logger (object): Logger instance.
            cfg (dict): Configuration dictionary.
            embedding_function (callable): Function to generate an embedding from text.
        """
        super().__init__(logger, cfg)
        self.collection_name = cfg.get("collection_name", "ebook_collection")
        self.dimension = cfg.get("vector_dim", 768)
        self.embedding_function = embedding_function
        # Mapping: chapter number -> chapter summary text
        self.chapter_summaries: Dict[str, str] = {}
        self.cfg = cfg
        # Establish connection and initialize the collection
        self.connect_db()
        self.init_db()
        self.logger.info(
            f"Connected to Milvus at {cfg.get('milvus', {}).get('host', 'localhost')}:{cfg.get('milvus', {}).get('port', '19530')}"
        )

    def __enter__(self):
        """ Enter method for context manager. """
        return self

    def __exit__(self, *_):
        """ Exit method for context manager. """
        connections.disconnect(alias="default")
        self.logger.info("Milvus connection disconnected (alias 'default').")

    def connect_db(self):
        """Establish a connection to the Milvus database using configuration parameters."""
        host = self.cfg.get("milvus", {}).get("host", "localhost")
        port = self.cfg.get("milvus", {}).get("port", "19530")
        # Create the connection (alias "default" will be used by all subsequent operations)
        connections.connect(alias="default", host=host, port=port)
        self.logger.info("Milvus connection established (alias 'default').")

    def init_db(self) -> None:
        """
        Initialize the Milvus collection for TXT data.
        If the collection does not exist, it is created with the appropriate schema.
        Otherwise, it is loaded.
        Then, create an index on the "vector" field if it does not exist.
        """
        try:
            if not utility.has_collection(self.collection_name):
                # Define collection fields (important_info removed)
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(
                        name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
                    ),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(
                        name="record_type", dtype=DataType.VARCHAR, max_length=20, is_filterable=True
                    ),
                ]
                schema = CollectionSchema(fields, description="eBook collection")
                collection = Collection(name=self.collection_name, schema=schema)
                self.logger.info(
                    f"Collection '{self.collection_name}' created in Milvus."
                )
            else:
                collection = Collection(self.collection_name)
                self.logger.info(
                    f"Collection '{self.collection_name}' loaded from Milvus."
                )
            # Set the collection as our client for subsequent operations
            self.client = collection

            # Check if an index exists on the vector field; if not, create one.
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
            self.logger.error(f"Collection initialization failed: {e}")
            raise RuntimeError(
                f"Error initializing collection '{self.collection_name}' in Milvus."
            ) from e
        except KeyError as e:
            self.logger.error(f"Configuration key missing: {e}")
            raise KeyError(f"Missing required configuration key: {e}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error during collection initialization: {e}")
            raise RuntimeError(
                "Unexpected error during Milvus collection initialization."
            ) from e

    def is_collection_empty(self) -> bool:
        """Check if the collection is empty and log the number of entities."""
        try:
            num = self.client.num_entities
            self.logger.info(f"Collection '{self.collection_name}' has {num} entities.")
            return num == 0
        except MilvusException as e:
            self.logger.error(f"Error checking collection stats: {e}")
            raise RuntimeError(
                f"Failed to retrieve collection stats for '{self.collection_name}'"
            ) from e

    def _get_partition_name(self, chapter: str) -> str:
        """
        Compute the partition name using the chapter number and configuration prefix.
        Also, create the partition if it does not exist.
        """
        partition_prefix = self.cfg.get("partition_prefix", "chapter_")
        partition_name = f"{partition_prefix}{chapter}"
        if not utility.has_partition(self.collection_name, partition_name):
            self.client.create_partition(partition_name)
            self.logger.info(f"Created partition: {partition_name}")
        return partition_name

    def add_vector(self, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        Add a vector to the Milvus collection in the partition corresponding to its chapter.

        Args:
            vector (List[float]): The vector representation.
            metadata (Dict[str, Any]): Dictionary containing:
                - "id": unique record id.
                - "text": the actual text (paragraph or summary).
                - "chapter": the chapter number.
                - "record_type": either "summary" or "paragraph".
        """
        try:
            self.logger.info(f"Inserting vector with metadata: {metadata}")
            chapter = metadata.get("chapter", "")
            partition_name = None
            if chapter:
                partition_name = self._get_partition_name(chapter)
            data = [
                {
                    "id": metadata["id"],
                    "vector": vector,
                    "text": metadata["text"],
                    "chapter": chapter,
                    "record_type": metadata.get("record_type", ""),
                }
            ]
            if partition_name:
                self.client.insert(data=data, partition_name=partition_name)
            else:
                self.client.insert(data=data)
            self.logger.info(f"Vector with ID {metadata['id']} added successfully.")
        except MilvusException as e:
            self.logger.error(f"Failed to add vector: {e}")
            raise RuntimeError(
                f"Error adding vector with ID {metadata['id']} to '{self.collection_name}'"
            ) from e

    def _process_summary_row(self, row: pd.Series, chapter_pattern: re.Pattern) -> None:
        """
        Process a summary row from a DataFrame.

        Args:
            row (pandas.Series): Row containing the summary line and record_id.
            chapter_pattern (re.Pattern): Compiled regex pattern for chapter summaries.
        """
        match = chapter_pattern.search(row["line"])
        if match:
            chap_num = match.group(1)
            chap_summary = match.group(2).strip()
            self.chapter_summaries[chap_num] = chap_summary
            vector = self.embedding_function(chap_summary)
            metadata = {
                "id": row["record_id"],
                "text": chap_summary,
                "chapter": chap_num,
                "record_type": "summary",
                "index": row["record_id"],
            }
            self.add_vector(vector, metadata)
        else:
            self.logger.info("Line did not match summary pattern; skipping.")

    def _process_paragraph_row(self, row: pd.Series) -> None:
        """
        Process a paragraph row from a DataFrame.

        Args:
            row (pandas.Series): Row containing the paragraph, chapter, and record_id.
        """
        para = row["paragraph"]
        chap = row["chapter"]
        vector = self.embedding_function(para)
        metadata = {
            "id": row["record_id"],
            "text": para,
            "chapter": chap,
            "record_type": "paragraph",
            "index": row["record_id"],
        }
        self.add_vector(vector, metadata)

    def ingest_by_paragraph(self, ebook_file_path: str, summary_file_path: str) -> None:
        """
        Ingest the ebook and summary data into Milvus.
        First, parse the summary file to store chapter summaries 
            as records with record_type "summary".
        Then, read the ebook file, split it into paragraphs,
          assign chapter numbers (defaulting to "0" for text
        before the first chapter header), and store each paragraph with record_type "paragraph".

        Args:
            ebook_file_path (str): Path to the ebook TXT file.
            summary_file_path (str): Path to the summary TXT file.
        """
        if not self.is_collection_empty():
            self.logger.info("Collection already contains data; skipping ingestion.")
            return

        # Ingest Summary Records
        try:
            with open(summary_file_path, "r", encoding="utf-8") as s_file:
                summary_lines = s_file.readlines()
            summary_df = pd.DataFrame({"line": summary_lines})
            summary_df["record_id"] = range(1, len(summary_df) + 1)
            chapter_pattern = re.compile(r"CHAPTER\s+(\d+)\s*-\s*(.+)", re.IGNORECASE)
            summary_df.apply(
                self._process_summary_row, args=(chapter_pattern,), axis=1
            )
            self.logger.info(
                f"Parsed and ingested chapter summaries from {summary_file_path}"
            )
        except Exception as e:
            self.logger.exception("Error parsing or ingesting summary file.")
            raise RuntimeError("Failed to ingest summary file") from e

        # Ingest Paragraph Records
        try:
            with open(ebook_file_path, "r", encoding="utf-8") as ebook_file:
                ebook_content = ebook_file.read().strip()
            paragraphs = [p.strip() for p in ebook_content.split("\n\n") if p.strip()]
        except Exception as e:
            self.logger.exception("Error reading ebook file.")
            raise RuntimeError("Failed to read ebook file") from e

        current_chapter = "0"  # Default for introductory paragraphs.
        chapter_header_pattern = re.compile(r"CHAPTER\s+(\d+)", re.IGNORECASE)
        annotated_paragraphs = []
        for para in paragraphs:
            match = chapter_header_pattern.search(para)
            if match:
                current_chapter = match.group(1)
                self.logger.info(
                    f"Detected chapter header; setting current chapter to {current_chapter}"
                )
                continue  # Skip the header line.
            annotated_paragraphs.append({"paragraph": para, "chapter": current_chapter})
        try:
            paragraphs_df = pd.DataFrame(annotated_paragraphs)
            start_id = summary_df["record_id"].max() + 1 if not summary_df.empty else 1
            paragraphs_df["record_id"] = range(start_id, start_id + len(paragraphs_df))
        except Exception as e:
            self.logger.exception("Error creating DataFrame for paragraphs.")
            raise RuntimeError("Failed to prepare paragraph data for ingestion.") from e

        paragraphs_df.apply(self._process_paragraph_row, axis=1)
        self.logger.info("TXT ingestion complete: summaries and paragraphs ingested.")

    def retrieve_similar_facts(
        self, query: str, top_k: int = 2
    ) -> Union[List[str], None]:
        """
        Retrieve the most similar paragraphs for a given query.
        First, search summary records to get the top 2 relevant chapters.
        Then, restrict the paragraph search to the partitions corresponding to those chapters.

        Args:
            query (str): The query text.
            top_k (int): Number of top similar paragraphs to retrieve (overall).

        Returns:
            Union[List[str], None]: List of matching paragraph texts, sorted by similarity.
        """
        try:
            # Ensure the collection is loaded before search
            self.client.load()
            query_vector = self.embedding_function(query)
            summary_results = self.client.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                filter="record_type == 'summary'",
                limit=2,
                output_fields=["chapter"],
            )
            self.logger.info(f"Summary results: {summary_results}")
            if not summary_results or len(summary_results[0]) == 0:
                self.logger.info("No matching chapter summary found.")
                return None

            selected_chapters = []
            for hit in summary_results[0]:
                chapter = hit.entity.get("chapter")
                distance = hit.distance  # Using attribute access for distance
                self.logger.info(
                    f"Found summary for chapter {chapter} with distance {distance}"
                )
                selected_chapters.append(chapter)

            if not selected_chapters:
                self.logger.info("No valid chapters found from summary search.")
                return None

            partition_names = [
                f"{self.cfg.get('partition_prefix', 'chapter_')}{chap}"
                for chap in selected_chapters
            ]

            paragraph_results = self.client.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                partition_names=partition_names,
                filter='record_type like "paragraph"f',
                limit=top_k,
                output_fields=["text"],
            )

            if not paragraph_results or len(paragraph_results[0]) == 0:
                self.logger.info("No matching paragraphs found.")
                return None

            final_paragraphs = [hit.entity.get("text") for hit in paragraph_results[0]]
            return final_paragraphs

        except MilvusException as e:
            self.logger.exception(f"Query failed: {e}")
            raise RuntimeError(
                f"Error retrieving similar facts for query: {query}"
            ) from e

    def query_loop(self, top_k: int) -> None:
        """
        Start an interactive query loop.

        Args:
            top_k (int): Number of top similar paragraphs to retrieve.
        """
        while True:
            query = input("Enter a TXT query (or type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                self.logger.info("Exiting TXT query loop.")
                break
            if not query:
                print("Query cannot be empty. Please enter valid text.")
                continue
            results = self.retrieve_similar_facts(query, top_k=top_k)
            if results:
                print("Top similar TXT records:")
                for idx, rec in enumerate(results, start=1):
                    print(f"{idx}. {rec[:100000]}")
            else:
                print("No similar TXT records found.")

    def get_max_id(self) -> int:
        """
        Get the maximum existing record ID from the collection.

        Returns:
            int: The maximum record ID, or 0 if the collection is empty.
        """
        try:
            self.client.load()
            res = self.client.query(
                expr="id >= 0",
                output_fields=["id"],
            )
            if res and len(res) > 0:
                max_id = max(item["id"] for item in res)
            else:
                max_id = 0
            self.logger.info(f"Current max record ID is {max_id}")
            return max_id
        except MilvusException as e:
            self.logger.error(f"Error retrieving max ID: {e}")
            raise RuntimeError("Failed to get the max ID from the collection.") from e

    def insert_data(
        self, ebook_file_path: str = None, summary_file_path: str = None
    ) -> None:
        """
        Insert new TXT data from files by processing them into paragraphs and summaries.
        If no file paths are provided, prompt the user.

        Args:
            ebook_file_path (str, optional): Path to the ebook TXT file.
            summary_file_path (str, optional): Path to the summary TXT file.
        """
        if ebook_file_path is None or summary_file_path is None:
            ebook_file_path = input(
                "Enter the path to the new ebook TXT file (or type 'exit' to cancel): "
            ).strip()
            if ebook_file_path.lower() == "exit":
                self.logger.info("Insert new TXT data cancelled by user.")
                return
            summary_file_path = input(
                "Enter the path to the new summary TXT file: "
            ).strip()

        current_max_id = self.get_max_id()
        record_id = current_max_id + 1
        self.logger.info(f"Starting record ID for new data: {record_id}")

        try:
            with open(summary_file_path, "r", encoding="utf-8") as s_file:
                summary_lines = s_file.readlines()
            summary_df = pd.DataFrame({"line": summary_lines})
            summary_df["record_id"] = range(record_id, record_id + len(summary_df))
            chapter_pattern = re.compile(r"CHAPTER\s+(\d+)\s*-\s*(.+)", re.IGNORECASE)
            summary_df.apply(
                lambda row: self._process_summary_row(row, chapter_pattern), axis=1
            )
            record_id += len(summary_df)
            self.logger.info(
                f"Parsed and ingested chapter summaries from {summary_file_path}"
            )
        except Exception as e:
            self.logger.exception("Error parsing or ingesting summary file.")
            raise RuntimeError("Failed to ingest summary file") from e

        try:
            with open(ebook_file_path, "r", encoding="utf-8") as ebook_file:
                ebook_content = ebook_file.read().strip()
            paragraphs = [p.strip() for p in ebook_content.split("\n\n") if p.strip()]
        except Exception as e:
            self.logger.exception(f"Error reading ebook file '{ebook_file_path}'")
            raise RuntimeError("Failed to read ebook file") from e

        current_chapter = "0"
        chapter_header_pattern = re.compile(r"CHAPTER\s+(\d+)", re.IGNORECASE)
        annotated_paragraphs = []
        for para in paragraphs:
            match = chapter_header_pattern.search(para)
            if match:
                current_chapter = match.group(1)
                self.logger.info(
                    f"Detected chapter header; setting current chapter to {current_chapter}"
                )
                continue
            annotated_paragraphs.append({"paragraph": para, "chapter": current_chapter})
        try:
            paragraphs_df = pd.DataFrame(annotated_paragraphs)
            paragraphs_df["record_id"] = range(
                record_id, record_id + len(paragraphs_df)
            )
        except Exception as e:
            self.logger.exception("Error creating DataFrame for paragraphs.")
            raise RuntimeError("Failed to prepare paragraph data for ingestion.") from e

        paragraphs_df.apply(self._process_paragraph_row, axis=1)
        self.logger.info(f"Inserted new paragraphs from '{ebook_file_path}'")
        self.log_collection_state()

    def delete_data(self) -> None:
        """
        Delete data from the collection.

        Supports two modes:
        1) Deleting an entire chapter (summary + paragraphs).
        2) Deleting specific paragraph records by ID.
        """
        mode = (
            input(
                "Do you want to delete an entire chapter or specific paragraphs? (Enter 'chapter' or 'paragraph'): "
            )
            .strip()
            .lower()
        )
        if mode == "chapter":
            chapter_input = input(
                "Enter the chapter number to delete (or type '0' to cancel): "
            ).strip()
            if chapter_input == "0":
                self.logger.info("Deletion cancelled by user.")
                return

            try:
                self.client.load()
                # Instead of manually finding IDs, delete by expression:
                expr = f"chapter == '{chapter_input}'"
                self.client.delete(expr=expr)
                self.logger.info(f"Deleted all records for chapter {chapter_input}.")
                print(f"Deleted chapter {chapter_input} (all records).")
            except MilvusException as e:
                self.logger.error(f"Failed to delete chapter {chapter_input}: {e}")
                raise RuntimeError("Error deleting records from the collection.") from e

        elif mode == "paragraph":
            record_ids = input(
                "Enter the paragraph record IDs to delete (comma separated, or type '0' to cancel): "
            ).strip()
            if record_ids == "0":
                self.logger.info("Deletion cancelled by user.")
                return
            try:
                id_list = [
                    int(rid.strip()) for rid in record_ids.split(",") if rid.strip()
                ]
                expr = f"id in [{','.join(str(x) for x in id_list)}]"
                self.client.delete(expr=expr)
                self.logger.info(f"Deleted TXT records with IDs: {id_list}")
            except ValueError:
                self.logger.error(
                    "Invalid input. Record IDs must be integers, separated by commas."
                )
                print(
                    "Invalid input. Record IDs must be integers, separated by commas."
                )
            except MilvusException as e:
                self.logger.error(f"Failed to delete TXT records: {e}")
                raise RuntimeError("Error deleting records from the collection.") from e

        else:
            print("Invalid choice. Please enter 'chapter' or 'paragraph'.")
            self.logger.info("Deletion cancelled: invalid choice.")
            return

        self.log_collection_state()

    def update_data(self) -> None:
        """
        Update an existing TXT record (summary or paragraph).
        For summary records, updates all paragraphs in the same chapter.
        """
        record_id_str = input(
            "Enter the TXT record ID to update (or '0' to cancel): "
        ).strip()
        if record_id_str == "0":
            self.logger.info("Update cancelled by user.")
            return
        try:
            record_id = int(record_id_str)
        except ValueError:
            print("Invalid input. Record ID must be an integer.")
            self.logger.error("Update cancelled: non-integer record ID.")
            return

        try:
            self.client.load()
            records = self.client.query(
                expr=f"id == {record_id}",
                output_fields=["id", "text", "chapter", "record_type"],
            )
            if not records:
                print(f"No record found with ID {record_id}.")
                self.logger.info(
                    f"No record found with ID {record_id}. Update aborted."
                )
                return
            old_record = records[0]
            old_text = old_record["text"]
            record_type = old_record["record_type"]
            chapter = old_record["chapter"]
            self.logger.info(
                f"Found record {record_id} -> Type={record_type}, Chapter={chapter}"
            )
        except MilvusException as e:
            self.logger.error(f"Query to find record {record_id} failed: {e}")
            print("Error finding the record to update. Please try again.")
            return

        print(f"Current text:\n{old_text}\n")
        new_text = input(
            "Enter the UPDATED summary text: "
            if record_type == "summary"
            else "Enter the UPDATED paragraph text: "
        ).strip()
        if not new_text:
            print("No new text given. Update cancelled.")
            self.logger.info("User provided empty text. Update aborted.")
            return

        try:
            self.client.delete(expr=f"id == {record_id}")
            self.logger.info(f"Deleted record ID {record_id} for update.")
        except Exception as e:
            self.logger.error(f"Failed to delete record {record_id} for update: {e}")
            print("Error deleting old record. Update aborted.")
            return

        try:
            new_vector = self.embedding_function(new_text)
            metadata = {
                "id": record_id,
                "text": new_text,
                "chapter": chapter,
                "record_type": record_type,
                "index": record_id,
            }
            self.add_vector(new_vector, metadata)
            self.logger.info(f"Inserted updated record with ID {record_id}.")
            print(f"Record {record_id} updated successfully!")
        except Exception as e:
            self.logger.error(f"Failed to insert updated record {record_id}: {e}")
            print("Error inserting the updated record. Update incomplete.")
            return

        if record_type == "summary":
            self.logger.info(
                f"Updating all paragraph records for chapter {chapter} with the new summary."
            )
            try:
                paragraphs = self.client.query(
                    expr=f"record_type == 'paragraph' AND chapter == '{chapter}'",
                    output_fields=["id", "text", "chapter", "record_type"],
                )
                for para in paragraphs:
                    pid = para["id"]
                    try:
                        self.client.delete(expr=f"id == {pid}")
                        new_metadata = {
                            "id": pid,
                            "text": para["text"],
                            "chapter": chapter,
                            "record_type": "paragraph",
                            "index": pid,
                        }
                        new_vector = self.embedding_function(para["text"])
                        self.add_vector(new_vector, new_metadata)
                        self.logger.info(
                            f"Updated paragraph record {pid} with new summary info."
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to update paragraph record {pid}: {e}"
                        )
            except MilvusException as e:
                self.logger.error(
                    f"Failed to query paragraph records for chapter {chapter}: {e}"
                )
                print("Error updating associated paragraph records.")

        self.log_collection_state()

    def log_collection_state(self) -> None:
        """
        Log the state of the collection, listing each record's ID, a snippet of its text,
        its chapter, and record type.
        """
        try:
            self.client.load()
            records = self.client.query(
                expr="id >= 0",
                output_fields=["id", "text", "chapter", "record_type"],
            )
            if records and len(records) > 0:
                self.logger.info(
                    f"Current collection state: {len(records)} records found."
                )
                for record in records:
                    snippet = record["text"][:50]
                    self.logger.info(
                        f"ID: {record['id']}, Chapter: {record.get('chapter','')}, Type: {record.get('record_type','')}, Text snippet: {snippet}..."
                    )
            else:
                self.logger.info("Collection is empty.")
        except MilvusException as e:
            self.logger.error(f"Error retrieving collection state: {e}")
            raise RuntimeError("Failed to log collection state.") from e

    def export_to_txt(self, output_file: str) -> None:
        """
        Export the TXT data from the collection to a text file.
        Each line in the output contains the record's ID, chapter, record type,
        and a snippet of its text.

        Args:
            output_file (str): Path to the output text file.
        """
        try:
            self.client.load()
            records = self.client.query(
                expr="id >= 0",
                output_fields=["id", "text", "chapter", "record_type"],
            )
            if not records:
                self.logger.info("No records found in the collection.")
                return
            with open(output_file, "w", encoding="utf-8") as f:
                for record in records:
                    line = (
                        f"ID: {record['id']}, Chapter: {record.get('chapter','')}, Type: {record.get('record_type','')}, "
                        f"Text: {record['text'][:10000]}...\n"
                    )
                    f.write(line)
            self.logger.info(f"TXT data exported to {output_file}.")
        except FileNotFoundError as e:
            self.logger.error(f"Export failed: File path '{output_file}' not found.")
            raise FileNotFoundError(
                f"File path '{output_file}' is invalid or inaccessible."
            ) from e
        except MilvusException as e:
            self.logger.error(f"Error querying Milvus for export: {e}")
            raise RuntimeError("Failed to retrieve data from Milvus for export.") from e
        except Exception as e:
            self.logger.error(f"Unexpected error exporting TXT data: {e}")
            raise RuntimeError(
                "An unexpected error occurred while exporting TXT data."
            ) from e
    
    def naive_retrieve_similar_facts(self, query: str, top_k: int = 5) -> Union[List[dict], None]:
        """
        Perform a naive search over all paragraph records without restricting to specific partitions.
        Returns a list of dictionaries containing the record id, text, and chapter.
        """
        try:
            self.client.load()
            query_vector = self.embedding_function(query)
            results = self.client.search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                filter="record_type == 'paragraph'",
                limit=top_k,
                output_fields=["id", "text", "chapter"],
            )
            if not results or len(results[0]) == 0:
                self.logger.info("No matching paragraphs found in naive search.")
                return None
            final_paragraphs = []
            for hit in results[0]:
                final_paragraphs.append({
                    "id": hit.entity.get("id"),
                    "text": hit.entity.get("text"),
                    "chapter": hit.entity.get("chapter")
                })
            return final_paragraphs
        except Exception as e:
            self.logger.exception(f"Naive query failed: {e}")
            raise RuntimeError(f"Error retrieving similar facts (naively) for query: {query}") from e
