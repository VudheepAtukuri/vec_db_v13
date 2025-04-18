"""
evaluation.py - Evaluates retrieval performance on
chapter and paragraph levels using provided evaluation queries,
and logs precision, recall, and F1 metrics.
"""

import json
import yaml
from loguru import logger
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections
from src.vec_db.milvus_db_summary import MilvusSummaryDatabase
from src.vec_db.milvus_db_paragraph import MilvusParagraphDatabase

# Load configuration from YAML file
with open("src/config/config.yaml", "r", encoding="utf-8") as file_handle:
    config = yaml.safe_load(file_handle)

# Load embedding model (same as in main.py)
embedding_model = config["embedding_model"]
logger.info(f"Loading embedding model: {embedding_model}")
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
model = AutoModel.from_pretrained(embedding_model)
logger.info("Embedding model loaded successfully.")


def embed_text(text: str) -> list:
    """
    Generate embedding vector for the given text.

    Args:
        text (str): The input text.
    Returns:
        list: The embedding vector as a list of floats.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy().tolist()


class EvaluateDB:
    """
    Evaluation class that loads evaluation queries and calculates precision, recall, and F1 score
    based on the TP, FP, FN, and TN counts.
    For each query, retrieval is performed using the summary and paragraph databases.
    The evaluation on the paragraph retrieval can be done using either "naive" or "abstracted" search,
    based on search_type.
    """

    def __init__(self, evaluation_file: str, summary_db: MilvusSummaryDatabase, paragraph_db: MilvusParagraphDatabase) -> None:
        """
        Initialize the evaluator with evaluation file and database instances.

        Args:
            evaluation_file (str): Path to the evaluation JSON file.
            summary_db (MilvusSummaryDatabase): The summary database instance.
            paragraph_db (MilvusParagraphDatabase): The paragraph database instance.
        """
        self.logger = logger
        self.config = config
        self.summary_db = summary_db
        self.paragraph_db = paragraph_db
        self.evaluation_data = self._load_evaluation_data(evaluation_file)
        self.tokenizer = tokenizer
        self.model = model

    def _load_evaluation_data(self, file_path: str) -> list:
        """
        Load evaluation queries from a JSON file.

        Args:
            file_path (str): Path to the evaluation file.
        Returns:
            list: A list of evaluation queries.
        """
        with open(file_path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        return data["queries"]

    def get_retrieved_chapters(self, query: str, top_k: int = 2) -> list:
        """
        Retrieve top_k chapters using the summary database.
        Returns a list of summary IDs (as strings).

        Args:
            query (str): The query text.
            top_k (int, optional): Number of top results to retrieve. Defaults to 2.
        Returns:
            list: List of summary IDs as strings.
        """
        try:
            results = self.summary_db.retrieve_similar_facts(query, top_k=top_k)
            if results:
                return [str(hit["id"]) for hit in results]
            return []
        except Exception as exception:
            self.logger.error(f"Error retrieving chapters for query '{query}': {exception}")
            return []

    def get_retrieved_paragraphs(self, query: str, top_k: int = 2, search_type: str = "naive") -> list:
        """
        Retrieve top_k paragraphs using the paragraph database.
        The retrieval method is determined by search_type:
         - "naive": calls the naive search function.
         - "abstracted": first retrieves similar summaries and then uses their IDs for abstracted paragraph search.
        Returns a list of paragraph IDs (as strings).

        Args:
            query (str): The query text.
            top_k (int, optional): Number of top results to retrieve. Defaults to 2.
            search_type (str, optional): Retrieval method ("naive" or "abstracted"). Defaults to "naive".
        Returns:
            list: List of paragraph IDs as strings.
        """
        try:
            if search_type == "naive":
                results = self.paragraph_db.retrieve_naive_search(query, top_k=top_k)
            elif search_type == "abstracted":
                sum_results = self.summary_db.retrieve_similar_facts(query, top_k=top_k)
                if not sum_results:
                    return []
                summary_ids = [str(hit["id"]) for hit in sum_results]
                results = self.paragraph_db.retrieve_abstracted_search(query, summary_ids, top_k=top_k)
            else:
                self.logger.error(f"Invalid search_type: {search_type}")
                return []
            if results:
                return [str(hit["id"]) for hit in results]
            return []
        except Exception as exception:
            self.logger.error(f"Error retrieving paragraphs for query '{query}' with search_type '{search_type}': {exception}")
            return []

    def evaluate_prediction(self, gt_list: list, pred_list: list) -> dict:
        """
        For a single query, compare the ordered predicted list to the ground truth list.
        Special handling: if gt_list == ["0"], then the answer is considered not in the vector space.

        Args:
            gt_list (list): Ground truth list of IDs.
            pred_list (list): Predicted list of IDs.
        Returns:
            dict: A dictionary with tp, fp, fn, and tn counts.
        """
        if gt_list == ["0"]:
            if "0" in pred_list:
                return {"tp": 0, "fp": 0, "fn": 0, "tn": 1}
            return {"tp": 0, "fp": len(pred_list), "fn": 0, "tn": 0}

        for i, pred in enumerate(pred_list):
            if pred in gt_list:
                tp = 1
                fp = i  # all items before the first correct prediction are false positives
                fn = 0
                return {"tp": tp, "fp": fp, "fn": fn, "tn": 0}
        return {"tp": 0, "fp": len(pred_list), "fn": 1, "tn": 0}

    def calculate_metrics(self, counts: dict) -> tuple:
        """
        Compute precision, recall, and F1 score from tp, fp, fn, and tn.

        Args:
            counts (dict): Dictionary containing tp, fp, fn, tn counts.
        Returns:
            tuple: (precision, recall, f1_score)
        """
        tp = counts.get("tp", 0)
        fp = counts.get("fp", 0)
        fn = counts.get("fn", 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1_score

    def evaluate_chapters(self, top_k: int = 2) -> tuple:
        """
        Evaluate chapter-level predictions over all queries.

        Args:
            top_k (int, optional): Number of top results to retrieve per query. Defaults to 2.
        Returns:
            tuple: A list of per-query results and an overall metrics dictionary.
        """
        results = []
        total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for query in self.evaluation_data:
            query_text = query["query"]
            pred_chapters = self.get_retrieved_chapters(query_text, top_k=top_k)
            raw_gt = query.get("relevant_chapter", [])
            if not isinstance(raw_gt, list):
                raw_gt = [raw_gt]
            # Adjust ground truth IDs if needed (e.g., adding 1)
            gt_chapters = [str(int(cid) + 1) for cid in raw_gt]
            counts = self.evaluate_prediction(gt_chapters, pred_chapters)
            precision, recall, f1_score = self.calculate_metrics(counts)
            results.append({
                "query": query_text,
                "gt_chapters": gt_chapters,
                "pred_chapters": pred_chapters,
                "counts": counts,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            })
            for key in total_counts:
                total_counts[key] += counts.get(key, 0)
        overall_precision = sum(r["precision"] for r in results) / len(results) if results else 0.0
        overall_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
        overall_f1 = sum(r["f1_score"] for r in results) / len(results) if results else 0.0
        overall = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
            "total_counts": total_counts
        }
        return results, overall

    def evaluate_paragraphs(self, top_k: int = 2, search_type: str = "naive") -> tuple:
        """
        Evaluate paragraph-level predictions over all queries using the specified search_type.

        Args:
            top_k (int, optional): Number of top results to retrieve per query. Defaults to 2.
            search_type (str, optional): "naive" or "abstracted" retrieval method. Defaults to "naive".
        Returns:
            tuple: A list of per-query results and an overall metrics dictionary.
        """
        results = []
        total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for query in self.evaluation_data:
            query_text = query["query"]
            pred_paragraphs = self.get_retrieved_paragraphs(query_text, top_k=top_k, search_type=search_type)
            raw_gt = query.get("relevant_paragraphs", [])
            if not isinstance(raw_gt, list):
                raw_gt = [raw_gt]
            gt_paragraphs = [str(x) for x in raw_gt]
            counts = self.evaluate_prediction(gt_paragraphs, pred_paragraphs)
            precision, recall, f1_score = self.calculate_metrics(counts)
            results.append({
                "query": query_text,
                "gt_paragraphs": gt_paragraphs,
                "pred_paragraphs": pred_paragraphs,
                "counts": counts,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            })
            for key in total_counts:
                total_counts[key] += counts.get(key, 0)
        overall_precision = sum(r["precision"] for r in results) / len(results) if results else 0.0
        overall_recall = sum(r["recall"] for r in results) / len(results) if results else 0.0
        overall_f1 = sum(r["f1_score"] for r in results) / len(results) if results else 0.0
        overall = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
            "total_counts": total_counts
        }
        return results, overall

    def evaluate_all(self, top_k: int = 2, search_type: str = "abstracted") -> dict:
        """
        Run evaluation for both chapter-level and paragraph-level predictions.
        The paragraph evaluation uses the specified search_type ("naive" or "abstracted").

        Args:
            top_k (int, optional): Number of top results to retrieve per query. Defaults to 2.
            search_type (str, optional): Retrieval method for paragraphs. Defaults to "abstracted".
        Returns:
            dict: Contains evaluation results and overall metrics for both chapters and paragraphs.
        """
        chapter_results, chapter_overall = self.evaluate_chapters(top_k=top_k)
        paragraph_results, paragraph_overall = self.evaluate_paragraphs(top_k=top_k, search_type=search_type)
        return {
            "chapter": {"results": chapter_results, "overall": chapter_overall},
            "paragraph": {"results": paragraph_results, "overall": paragraph_overall},
        }


if __name__ == "__main__":
    # Constant for evaluation file path
    EVAL_FILE = "data/evaluation_data1.json"
    # Establish a Milvus connection using the configuration
    milvus_cfg = config.get("milvus", {})
    host = milvus_cfg.get("host", "localhost")
    port = milvus_cfg.get("port", 19530)
    connections.connect(alias="default", host=host, port=port)
    logger.info(f"Milvus connection established at {host}:{port} (alias 'default')")

    # Instantiate summary and paragraph database objects using embed_text
    summary_db = MilvusSummaryDatabase(logger, config["vector_db_doc"], embed_text)
    paragraph_db = MilvusParagraphDatabase(logger, config["vector_db_doc"], embed_text)

    # Read search_type from config (should be either "naive" or "abstracted")
    search_type_value = config["vector_db_doc"].get("search_mode", "naive")
    logger.info(f"Using search_type: {search_type_value}")

    # Create an evaluator instance
    evaluator = EvaluateDB(EVAL_FILE, summary_db, paragraph_db)

    # Constant for top_k value
    TOP_K_VALUE = 2
    evaluation_results = evaluator.evaluate_all(top_k=TOP_K_VALUE, search_type=search_type_value)

    # Log overall metrics for chapters and paragraphs
    logger.info("Chapter-Level Overall Metrics:")
    logger.info(evaluation_results["chapter"]["overall"])
    logger.info("Paragraph-Level Overall Metrics:")
    logger.info(evaluation_results["paragraph"]["overall"])

    # Disconnect from Milvus after evaluation
    connections.disconnect(alias="default")
    logger.info("Milvus connection disconnected (alias 'default').")
