"""
test_milvus_db_paragraph.py

Unit tests for the MilvusParagraphDatabase implementation in milvus_db_paragraph.py.
These tests ensure that methods execute without errors and that failures are raised as expected,
without verifying the precise output details.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from pymilvus.exceptions import MilvusException
from src.vec_db.milvus_db_paragraph import MilvusParagraphDatabase

# Create a dummy logger for testing.
import logging
logger = logging.getLogger("test_logger")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TestMilvusParagraphDatabase:
    # Dummy embedding function that returns a fixed vector.
    @staticmethod
    def __embed_test(text: str):
        return [0.1] * 768

    @pytest.fixture(autouse=True)
    def setup(self):
        # Test configuration for the paragraph database.
        test_config = {
            "collection_name": "test_collection",
            "paragraph_collection_name": "test_collection_paragraph",
            "vector_dim": 768,
            "milvus": {"host": "localhost", "port": "19530"},
            "partition_prefix": "summary_",
        }
        # Create a dummy client to inject.
        self.dummy_client = MagicMock()
        self.dummy_client.num_entities = 0
        self.dummy_client.has_index.return_value = True
        self.dummy_client.query.return_value = []

        # Patch out utility.has_collection and Collection so no real Milvus connection is made.
        with patch("src.vec_db.milvus_db_paragraph.utility.has_collection", return_value=True), \
             patch("src.vec_db.milvus_db_paragraph.Collection", return_value=self.dummy_client):
            self.paragraph_db = MilvusParagraphDatabase(
                logger=logger,
                cfg=test_config,
                embedding_function=self.__embed_test,
                client=self.dummy_client,
            )
        yield
        self.dummy_client.reset_mock()

    def test_connect_db_success(self):
        """
        Test that connect_db establishes a connection successfully.
        (We patch connections.connect to avoid real connection attempts.)
        """
        with patch("src.vec_db.milvus_db_paragraph.connections.connect") as mock_connect:
            self.paragraph_db.connect_db()
            mock_connect.assert_called_once_with(alias="default", host="localhost", port="19530")

    def test_init_db_success(self):
        """
        Test that init_db loads an existing collection and creates an index if needed.
        """
        with patch("src.vec_db.milvus_db_paragraph.utility.has_collection", return_value=True), \
             patch("src.vec_db.milvus_db_paragraph.Collection", return_value=self.dummy_client):
            self.paragraph_db.init_db()
            self.dummy_client.has_index.assert_called_with(field_name="vector")

    def test_init_db_failure(self):
        """
        Test that init_db raises RuntimeError when index creation fails.
        """
        with patch("src.vec_db.milvus_db_paragraph.utility.has_collection", return_value=True), \
             patch("src.vec_db.milvus_db_paragraph.Collection", return_value=self.dummy_client):
            self.dummy_client.has_index.return_value = False
            self.dummy_client.create_index.side_effect = MilvusException("Test exception")
            with pytest.raises(RuntimeError) as excinfo:
                self.paragraph_db.init_db()
            assert "Error initializing paragraph collection" in str(excinfo.value)

    def test_is_collection_empty_success(self):
        """
        Test that is_collection_empty returns True when num_entities is 0.
        """
        self.dummy_client.num_entities = 0
        empty = self.paragraph_db.is_collection_empty()
        assert empty is True, "Collection should be empty when num_entities is 0."

    def test_get_max_id_empty(self):
        """
        Test that get_max_id returns 0 when query returns no records.
        """
        self.dummy_client.query.return_value = []
        max_id = self.paragraph_db.get_max_id()
        assert max_id == 0, "Max ID should be 0 if collection is empty."

    def test_get_max_id_after_insert(self):
        """
        Test that get_max_id returns the highest id from queried records.
        """
        self.dummy_client.query.return_value = [{"id": 2}, {"id": 5}, {"id": 3}]
        max_id = self.paragraph_db.get_max_id()
        assert max_id == 5, "Max ID should be the highest id among queried records."

    def test_add_vector_success(self):
        """
        Test that add_vector executes successfully with valid metadata.
        """
        test_metadata = {"id": 1, "text": "Test paragraph text", "chapter": "1"}
        test_vector = self.__embed_test("Test paragraph text")
        # Patch _get_partition_name to return a dummy partition.
        with patch.object(self.paragraph_db, "_get_partition_name", return_value="summary_1") as mock_get_partition:
            try:
                self.paragraph_db.add_vector(test_vector, test_metadata)
                mock_get_partition.assert_called_with("1")
                self.dummy_client.insert.assert_called_once_with(
                    data=[{
                        "id": test_metadata["id"],
                        "vector": test_vector,
                        "text": test_metadata["text"],
                        "chapter": "1",
                    }],
                    partition_name="summary_1"
                )
            except Exception as e:
                pytest.fail(f"add_vector raised an unexpected exception: {e}")

    def test_ingest_by_paragraph_skip_if_not_empty(self):
        """
        Test that ingest_by_paragraph does not proceed if the collection is not empty.
        """
        with patch.object(self.paragraph_db, "is_collection_empty", return_value=False):
            self.paragraph_db.ingest_by_paragraph("dummy_ebook.txt")
        self.dummy_client.insert.assert_not_called()

    def test_ingest_by_paragraph_failure(self, tmp_path):
        """
        Test that ingest_by_paragraph raises RuntimeError when reading the ebook file fails.
        """
        fake_path = tmp_path / "fake_ebook.txt"
        fake_path.write_text("CHAPTER 1\nParagraph one.")
        with patch("builtins.open", side_effect=Exception("Test read error")):
            with pytest.raises(RuntimeError) as excinfo:
                self.paragraph_db.ingest_by_paragraph(str(fake_path))
            assert "Failed to read ebook file" in str(excinfo.value)

    def test_export_to_txt_success(self, tmp_path):
        """
        Test that export_to_txt writes paragraph data to a file successfully.
        """
        self.dummy_client.query.return_value = [
            {"id": 1, "text": "Paragraph text 1", "chapter": "1"},
            {"id": 2, "text": "Paragraph text 2", "chapter": "1"},
        ]
        output_file = tmp_path / "export.txt"
        try:
            self.paragraph_db.export_to_txt(str(output_file))
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Paragraph text 1" in content, "Exported file should contain paragraph text 1."
            assert "Paragraph text 2" in content, "Exported file should contain paragraph text 2."
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_insert_data_failure_file_not_found(self, tmp_path):
        """
        Test that insert_data returns without insertion when the ebook file does not exist.
        """
        non_existent_path = str(tmp_path / "nonexistent.txt")
        if os.path.exists(non_existent_path):
            os.remove(non_existent_path)
        self.paragraph_db.insert_data(non_existent_path)
        self.dummy_client.insert.assert_not_called()

    def test_delete_paragraphs_by_summary_success(self):
        """
        Test that delete_paragraphs_by_summary calls client.delete with the proper expression.
        """
        # For summary_id "2", the stored chapter is expected to be "1" (2-1).
        with patch("src.vec_db.milvus_db_paragraph.utility.has_partition", return_value=True):
            self.paragraph_db.delete_paragraphs_by_summary("2")
            self.dummy_client.load.assert_called()
            self.dummy_client.delete.assert_called_once_with(expr="chapter == '1'")

    def test_delete_data_chapter_mode(self):
        """
        Test delete_data in chapter mode: simulate input to delete paragraphs by summary.
        """
        with patch("builtins.input", side_effect=["chapter", "3"]):
            with patch.object(self.paragraph_db, "delete_paragraphs_by_summary") as mock_del:
                self.paragraph_db.delete_data()
                mock_del.assert_called_once_with("3")

    def test_delete_data_paragraph_mode(self):
        """
        Test delete_data in paragraph mode: simulate input to delete specific paragraph IDs.
        """
        with patch("builtins.input", side_effect=["paragraph", "10, 20"]):
            with patch.object(self.dummy_client, "load") as mock_load:
                self.paragraph_db.delete_data()
                mock_load.assert_called()
                self.dummy_client.delete.assert_called_once()
                args, kwargs = self.dummy_client.delete.call_args
                expr = kwargs.get("expr", "")
                assert "10" in expr and "20" in expr

    def test_update_data_invalid_record(self):
        """
        Test that update_data prints error and returns when no record is found.
        """
        self.dummy_client.query.return_value = []
        with patch("builtins.input", side_effect=["9999"]):
            self.paragraph_db.update_data()
            self.dummy_client.delete.assert_not_called()

    def test_update_data_empty_text(self):
        """
        Test that update_data cancels update when new text is empty.
        """
        self.dummy_client.query.return_value = [{"id": 15, "text": "Old text", "chapter": "2"}]
        with patch("builtins.input", side_effect=["15", ""]):
            self.paragraph_db.update_data()
            self.dummy_client.delete.assert_not_called()

    def test_update_data_success(self):
        """
        Test that update_data deletes the old record and calls add_vector for the updated text.
        """
        self.dummy_client.query.return_value = [{"id": 15, "text": "Old text", "chapter": "2"}]
        self.dummy_client.delete.return_value = None
        with patch.object(self.paragraph_db, "_get_partition_name", return_value="dummy_partition"):
            with patch("builtins.input", side_effect=["15", "New updated text"]):
                try:
                    self.paragraph_db.update_data()
                except Exception as e:
                    pytest.fail(f"update_data raised an unexpected exception: {e}")
                self.dummy_client.delete.assert_called()
                self.dummy_client.insert.assert_called()

    def test_retrieve_naive_search_no_match(self):
        """
        Test that retrieve_naive_search returns None when search yields no results.
        """
        self.dummy_client.search.return_value = [[]]
        result = self.paragraph_db.retrieve_naive_search("query", top_k=2)
        assert result is None, "retrieve_naive_search should return None if no matches are found."

    def test_retrieve_abstracted_search_no_valid_partitions(self):
        """
        Test that retrieve_abstracted_search returns None when no valid partitions are found.
        """
        with patch("src.vec_db.milvus_db_paragraph.utility.has_partition", return_value=False):
            result = self.paragraph_db.retrieve_abstracted_search("query", ["1", "2"], top_k=2)
            assert result is None, "retrieve_abstracted_search should return None if no valid partitions exist."

    def test_retrieve_abstracted_search_success(self):
        """
        Test that retrieve_abstracted_search returns results when valid partitions are found.
        """
        with patch("src.vec_db.milvus_db_paragraph.utility.has_partition", return_value=True):
            dummy_hit = MagicMock()
            dummy_hit.entity = {"id": 5, "chapter": "1", "text": "Found paragraph"}
            self.dummy_client.search.return_value = [[dummy_hit]]
            result = self.paragraph_db.retrieve_abstracted_search("query", ["2"], top_k=2)
            assert isinstance(result, list), "retrieve_abstracted_search should return a list of results."
            assert result[0]["text"] == "Found paragraph", "Result text should match expected value."
