"""
main.py - Implements Milvus vector database operations
for summarizing, querying, inserting, deleting, and updating document data.
"""

import os
import json
import yaml
import cProfile
import pstats
from loguru import logger
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections
from src.vec_db.milvus_db_summary import MilvusSummaryDatabase
from src.vec_db.milvus_db_paragraph import MilvusParagraphDatabase


# Load configuration from YAML file
with open("src/config/config.yaml", "r", encoding="utf-8") as file_handle:
    config = yaml.safe_load(file_handle)

# Load embedding model
logger.info(f"Loading embedding model: {config['embedding_model']}")
tokenizer = AutoTokenizer.from_pretrained(config["embedding_model"])
model = AutoModel.from_pretrained(config["embedding_model"])
logger.info("Embedding model loaded successfully.")


def embed_text(text):
    """
    Generates embedding vector for the given text.

    Args:
        text (str): The input text to embed.
    Returns:
        List[float]: The generated embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy().tolist()


def load_user_inputs():
    """
    Loads user input from a JSON file.
    Returns:
        dict: User input data; defaults to {"txt_query": "exit"} on error.
    """
    try:
        with open("data/user_inputs.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load user inputs: {e}")
        return {"txt_query": "exit"}


def process_data_ingestion(summary_db, paragraph_db):
    """
    Ingests base data from summary and ebook text files into the databases.

    Args:
        summary_db (MilvusSummaryDatabase): The summary database instance.
        paragraph_db (MilvusParagraphDatabase): The paragraph database instance.
        config (dict): Configuration dictionary.
    """
    summary_path = os.path.join("data", "summary.txt")
    ebook_path = os.path.join("data", "ebook.txt")
    summary_db.ingest_by_chapter(summary_path)
    paragraph_db.ingest_by_paragraph(ebook_path)


def perform_query(summary_db, paragraph_db, config):
    """
    Processes a user query to search through the vector databases.

    Args:
        summary_db (MilvusSummaryDatabase): The summary database instance.
        paragraph_db (MilvusParagraphDatabase): The paragraph database instance.
        config (dict): Configuration dictionary.
    """
    user_inputs = load_user_inputs()
    query = user_inputs.get("txt_query", "").strip()
    if not query:
        query = input("Enter your query (or type 'exit' to quit): ").strip()

    if query.lower() != "exit":
        search_mode = config["vector_db_doc"].get("search_mode", "naive")
        if search_mode == "naive":
            paragraph_results = paragraph_db.retrieve_naive_search(
                query, top_k=config["vector_db_doc"]["top_k_retrieve"]
            )
            print("\nTop similar paragraphs:")
            if paragraph_results:
                for idx, res in enumerate(paragraph_results, start=1):
                    print(
                        f"{idx}. (ID: {res['id']}, "
                        f"Chapter: {res['chapter']}) "
                        f"{res['text'][:200]}..."
                    )
            else:
                print("No matching paragraphs found.")
        elif search_mode == "abstracted":
            summary_results = summary_db.retrieve_similar_facts(
                query, top_k=config["vector_db_doc"]["top_k_retrieve"]
            )
            print("\nTop similar summaries:")
            if summary_results:
                for idx, res in enumerate(summary_results, start=1):
                    print(f"{idx}. (Summary ID: {res['id']}) {res['text'][:200]}...")
            else:
                print("No matching summaries found.")

            if summary_results:
                summary_ids = [str(res["id"]) for res in summary_results]
                paragraph_results = paragraph_db.retrieve_abstracted_search(
                    query, summary_ids, top_k=config["vector_db_doc"]["top_k_retrieve"]
                )
                print("\nTop similar paragraphs:")
                if paragraph_results:
                    for idx, res in enumerate(paragraph_results, start=1):
                        print(
                            f"{idx}. (ID: {res['id']}, Chapter: {res['chapter']}) "
                            f"{res['text'][:200]}..."
                        )
                else:
                    print("No matching paragraphs found.")
        else:
            logger.error(f"Invalid search mode: {search_mode}")


def export_data(summary_db, paragraph_db):
    """
    Exports the contents of the summary and paragraph collections to text files.

    Args:
        summary_db (MilvusSummaryDatabase): The summary database instance.
        paragraph_db (MilvusParagraphDatabase): The paragraph database instance.
    """
    summary_export_path = os.path.join("data", "exported_data", "exported_summary.txt")
    paragraph_export_path = os.path.join(
        "data", "exported_data", "exported_paragraph.txt"
    )
    summary_db.export_to_txt(summary_export_path)
    paragraph_db.export_to_txt(paragraph_export_path)


def handle_insertion(summary_db, paragraph_db, config):
    """
    Handles the insertion of new data based on user input and configuration."""

    if config["vector_db_doc"].get("allow_insert_new_data", False):
        insert_choice = (
            input("Do you want to insert new data? (yes/no): ").strip().lower()
        )
        if insert_choice == "yes":
            summary_file = input(
                "Enter the summary file name (e.g., insertsummary.txt): "
            ).strip()
            paragraph_file = input(
                "Enter the paragraph file name (e.g., insertdata.txt): "
            ).strip()
            summary_file_path = os.path.join("data", summary_file)
            paragraph_file_path = os.path.join("data", paragraph_file)
            try:
                summary_db.insert_data(summary_file_path)
                paragraph_db.insert_data(paragraph_file_path)
            except Exception as e:
                logger.error(f"Error inserting new data: {e}")
                raise RuntimeError("Error inserting new data") from e
        else:
            logger.info("Skipping insertion of new data as per user input.")
    else:
        logger.info("Insertion of new data is disabled in config.")


def handle_deletion(summary_db, paragraph_db, config):
    """
    Handles the deletion of data based on user input and configuration."""

    if not config["vector_db_doc"].get("allow_delete_data", False):
        logger.info("Deletion of data is disabled in config.")
        return

    delete_choice = input("Do you want to delete data? (yes/no): ").strip().lower()
    if delete_choice != "yes":
        print("Skipping deletion.")
        return

    deletion_mode = (
        input(
            "Delete an entire chapter or specific paragraphs? (Enter 'chapter' or 'paragraph'): "
        )
        .strip()
        .lower()
    )
    if deletion_mode == "chapter":
        summary_id_input = input("Enter the summary (chapter) ID to delete: ").strip()
        try:
            summary_id = int(summary_id_input)  # validate numeric
        except ValueError:
            print("Invalid input. Summary ID must be an integer.")
            return
        # Delete the summary record.
        summary_db.delete_chapter(summary_id)
        # Delete all paragraphs associated with this summary.
        paragraph_db.delete_paragraphs_by_summary(str(summary_id))
    elif deletion_mode == "paragraph":
        record_ids = input(
            "Enter the paragraph record IDs to delete (comma separated): "
        ).strip()
        try:
            id_list = [int(rid.strip()) for rid in record_ids.split(",") if rid.strip()]
        except ValueError:
            print("Invalid input. Record IDs must be integers.")
            return
        try:
            paragraph_db.client.load()
            expr = f"id in [{','.join(str(x) for x in id_list)}]"
            paragraph_db.client.delete(expr=expr)
            paragraph_db.logger.info(f"Deleted paragraph records with IDs: {id_list}")
            print(f"Deleted paragraph records with IDs: {id_list}")
        except Exception as e:
            paragraph_db.logger.error(f"Error deleting paragraph records: {e}")
            raise RuntimeError("Error deleting paragraph records") from e
    else:
        print("Invalid deletion choice. Skipping deletion.")


def handle_update(summary_db, paragraph_db, config):
    """
    Handles the update of data based on user input and configuration."""

    if config["vector_db_doc"].get("allow_update_data", False):
        update_choice = input("Do you want to update data? (yes/no): ").strip().lower()
        if update_choice == "yes":
            summary_db.update_data()
            paragraph_db.update_data()
        else:
            logger.info("Skipping update as per user input.")
    else:
        logger.info("Update is disabled in config.")


def handle_data_operations(summary_db, paragraph_db, config):
    handle_insertion(summary_db, paragraph_db, config)
    handle_deletion(summary_db, paragraph_db, config)
    handle_update(summary_db, paragraph_db, config)


def main():
    """
    Main function to connect to Milvus, initialize databases, ingest base data (if needed),
    handle data operations, perform queries, export data, and disconnect from Milvus.
    """
    # Connect to Milvus
    milvus_cfg = config.get("milvus", {})
    host = milvus_cfg.get("host", "localhost")
    port = milvus_cfg.get("port", 19530)
    connections.connect(alias="default", host=host, port=port)
    logger.info(f"Milvus connection established at {host}:{port} (alias 'default')")

    # Initialize databases
    summary_db = MilvusSummaryDatabase(logger, config["vector_db_doc"], embed_text)
    paragraph_db = MilvusParagraphDatabase(logger, config["vector_db_doc"], embed_text)

    # Ingest base data if collections are empty
    if summary_db.is_collection_empty():
        process_data_ingestion(summary_db, paragraph_db)
    else:
        logger.info("Collections already contain data. Skipping base ingestion.")

    # Handle insertion, deletion, and update operations
    handle_data_operations(summary_db, paragraph_db, config)

    # Process user query
    perform_query(summary_db, paragraph_db, config)

    # Export data
    export_data(summary_db, paragraph_db)

    # Disconnect from Milvus
    connections.disconnect(alias="default")
    logger.info("Milvus connection disconnected (alias 'default').")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.dump_stats("main.out")
