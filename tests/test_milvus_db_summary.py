"""
test_milvus_db_summary.py

Unit tests for the MilvusSummaryDatabase implementation in milvus_db_summary.py.
These tests check that methods execute without errors and that failures are raised appropriately,
without verifying the precise output values.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from pymilvus.exceptions import MilvusException
from src.vec_db.milvus_db_summary import MilvusSummaryDatabase

# Create a dummy logger for testing.
import logging
logger = logging.getLogger("test_logger")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TestMilvusSummaryDatabase:
    # Dummy embedding function returning a fixed vector.
    @staticmethod
    def __embed_test(text: str):
        return [0.1] * 768

    @pytest.fixture(autouse=True)
    def setup(self):
        # Test configuration for the summary database.
        test_config = {
            "collection_name": "test_collection",
            "summary_collection_name": "test_collection_summary",
            "vector_dim": 768,
            "milvus": {"host": "localhost", "port": "19530"},
        }
        # Create a dummy client to inject.
        self.dummy_client = MagicMock()
        self.dummy_client.num_entities = 0
        self.dummy_client.has_index.return_value = True
        self.dummy_client.query.return_value = []

        # Patch out calls to utility.has_collection and Collection so that no real Milvus connection is made.
        with patch("src.vec_db.milvus_db_summary.utility.has_collection", return_value=True), \
             patch("src.vec_db.milvus_db_summary.Collection", return_value=self.dummy_client):
            self.summary_db = MilvusSummaryDatabase(
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
        with patch("src.vec_db.milvus_db_summary.connections.connect") as mock_connect:
            self.summary_db.connect_db()
            mock_connect.assert_called_once_with(alias="default", host="localhost", port="19530")

    def test_init_db_success(self):
        """
        Test that init_db loads an existing collection and creates an index if needed.
        """
        with patch("src.vec_db.milvus_db_summary.utility.has_collection", return_value=True), \
             patch("src.vec_db.milvus_db_summary.Collection", return_value=self.dummy_client):
            self.summary_db.init_db()
            # The dummy client's has_index method should have been called.
            self.dummy_client.has_index.assert_called_with(field_name="vector")

    def test_init_db_failure(self):
        """
        Test that init_db raises RuntimeError when an exception occurs during index creation.
        """
        with patch("src.vec_db.milvus_db_summary.utility.has_collection", return_value=True), \
             patch("src.vec_db.milvus_db_summary.Collection", return_value=self.dummy_client):
            self.dummy_client.has_index.return_value = False
            self.dummy_client.create_index.side_effect = MilvusException("Test exception")
            with pytest.raises(RuntimeError) as excinfo:
                self.summary_db.init_db()
            assert "Error initializing summary collection" in str(excinfo.value)

    def test_is_collection_empty_success(self):
        """
        Test that is_collection_empty returns True when num_entities is 0.
        """
        self.dummy_client.num_entities = 0
        empty = self.summary_db.is_collection_empty()
        assert empty is True, "Collection should be empty when num_entities is 0."

    def test_add_vector_success(self):
        """
        Test that add_vector executes successfully with proper metadata.
        """
        test_metadata = {"id": 1, "text": "Test summary text"}
        test_vector = self.__embed_test("Test summary text")
        try:
            self.summary_db.add_vector(test_vector, test_metadata)
            self.dummy_client.insert.assert_called_once()
        except Exception as e:
            pytest.fail(f"add_vector raised an unexpected exception: {e}")

    def test_get_max_id_empty(self):
        """
        Test that get_max_id returns 0 when query returns no records.
        """
        self.dummy_client.query.return_value = []
        max_id = self.summary_db.get_max_id()
        assert max_id == 0, "Max ID should be 0 if collection is empty."

    def test_get_max_id_after_insert(self):
        """
        Test that get_max_id returns the highest id from the queried records.
        """
        self.dummy_client.query.return_value = [{"id": 5}, {"id": 3}, {"id": 7}]
        max_id = self.summary_db.get_max_id()
        assert max_id == 7, "Max ID should be the highest id among queried records."

    def test_retrieve_similar_facts_no_match(self):
        """
        Test that retrieve_similar_facts returns None when search yields no results.
        """
        self.dummy_client.search.return_value = [[]]
        result = self.summary_db.retrieve_similar_facts("Random query", top_k=2)
        assert result is None, "retrieve_similar_facts should return None if no matches are found."

    def test_ingest_by_chapter_skip_if_not_empty(self):
        """
        Test that ingest_by_chapter does not proceed if the collection is not empty.
        """
        with patch.object(self.summary_db, "is_collection_empty", return_value=False):
            self.summary_db.ingest_by_chapter("dummy_summary.txt")
        self.dummy_client.insert.assert_not_called()

    def test_ingest_by_chapter_failure(self, tmp_path):
        """
        Test that ingest_by_chapter raises an exception when file reading fails.
        """
        fake_path = tmp_path / "fake_summary.txt"
        fake_path.write_text("CHAPTER 1 - This is a summary\n")
        with patch("builtins.open", side_effect=Exception("Test read error")):
            with pytest.raises(Exception) as excinfo:
                self.summary_db.ingest_by_chapter(str(fake_path))
            assert "Test read error" in str(excinfo.value)

    def test_export_to_txt_success(self, tmp_path):
        """
        Test that export_to_txt writes content to a file successfully.
        """
        self.dummy_client.query.return_value = [
            {"id": 1, "text": "Summary text 1"},
            {"id": 2, "text": "Summary text 2"},
        ]
        output_file = tmp_path / "export.txt"
        try:
            self.summary_db.export_to_txt(str(output_file))
            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            assert "Summary text 1" in content, "Exported file should contain summary text 1."
            assert "Summary text 2" in content, "Exported file should contain summary text 2."
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

    def test_insert_data_failure_file_not_found(self, tmp_path):
        """
        Test that insert_data logs an error and returns when the summary file does not exist.
        """
        non_existent_path = str(tmp_path / "nonexistent.txt")
        if os.path.exists(non_existent_path):
            os.remove(non_existent_path)
        self.summary_db.insert_data(non_existent_path)
        self.dummy_client.insert.assert_not_called()

    def test_delete_chapter_success(self):
        """
        Test that delete_chapter calls client.delete with the proper expression.
        """
        self.summary_db.delete_chapter(3)
        self.dummy_client.delete.assert_called_once_with(expr="id == 3")

    def test_update_data_invalid_record(self):
        """
        Test that update_data raises a RuntimeError when the provided record id does not exist.
        """
        self.dummy_client.query.return_value = []
        with patch("builtins.input", side_effect=["9999"]):
            with pytest.raises(RuntimeError) as excinfo:
                self.summary_db.update_data()
            assert "No summary record found" in str(excinfo.value)

    def test_update_data_empty_text(self):
        """
        Test that update_data raises RuntimeError when new summary text is empty.
        """
        self.dummy_client.query.return_value = [{"id": 10, "text": "Old summary"}]
        with patch("builtins.input", side_effect=["10", ""]):
            with pytest.raises(RuntimeError) as excinfo:
                self.summary_db.update_data()
            assert "No new text given" in str(excinfo.value)

    def test_update_data_cancel(self):
        """
        Test that update_data raises RuntimeError if the user cancels the update by entering '0'.
        """
        with patch("builtins.input", side_effect=["0"]):
            with pytest.raises(RuntimeError) as excinfo:
                self.summary_db.update_data()
            assert "cancelled by user" in str(excinfo.value)

    def test_update_data_success(self):
        """
        Test that update_data deletes the old record and calls add_vector for updated text.
        """
        self.dummy_client.query.return_value = [{"id": 20, "text": "Old summary"}]
        self.dummy_client.delete.return_value = None
        with patch("builtins.input", side_effect=["20", "New summary text"]):
            try:
                self.summary_db.update_data()
            except Exception as e:
                pytest.fail(f"update_data raised an unexpected exception: {e}")
            self.dummy_client.delete.assert_called()
            self.dummy_client.insert.assert_called()
