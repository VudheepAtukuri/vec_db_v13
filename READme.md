**Milvus Vector Database Project**:
This project implements a vector database using Milvus for storing and retrieving text and CSV-based embeddings. The system supports operations such as inserting, deleting, updating, and querying vectorized data.

**Prerequisites**:
Python 3.12
Conda (for environment management)
Docker and Docker Compose

**Create a Virtual environment**:
conda create -n milvus_vecdb python=3.12 -y
conda activate milvus_vecdb

**Dependencies** -
This project requires the following dependencies:
transformers (for generating embeddings)
pymilvus (Milvus client for managing vector storage)
pandas (for handling CSV data processing)
loguru (for logging system messages)

**Install Dependicies using** -
pip install -r requirements.txt

Alternatively, install the project in editable mode:
pip install -e .
this allows to make changes to the project without reinstalling dependencies. 


**Installing Milvus Standalone** -
This project requires a running Milvus instance. You can install Milvus Standalone using Docker Compose directly from Milvus without including a docker-compose.yml file in the repository. Follow these steps:

Prerequisites:

Ensure your system meets the hardware/software requirements.
For macOS 10.14 or later, configure your Docker VM to use at least 2 virtual CPUs (vCPUs) and 8 GB of memory.

Download the YAML File:
Download the Milvus standalone Docker Compose file manually or with:

wget https://github.com/milvus-io/milvus/releases/download/v2.2.16/milvus-standalone-docker-compose.yml -O docker-compose.yml

Start Milvus:
In the same directory as the downloaded docker-compose.yml, start Milvus with:
sudo docker-compose up -d

Verify Running Containers:
Check that the following containers are running:
sudo docker-compose ps

You should see three containers:
milvus-etcd
milvus-minio
milvus-standalone (with ports mapped to 19530 and 9091)

Connect to Milvus:
Verify which local port Milvus is listening on:
sudo docker port milvus-standalone 19530/tcp
Use the returned IP and port in your configuration if needed.

Stop Milvus:
To stop the Milvus instance, run:
sudo docker-compose down

To delete data after stopping, run:
sudo rm -rf volumes



**Configuration** -
The config.py file allows customization of database settings. Key parameters include:
data_source: Choose txt, csv, or both
vector_db_doc: Settings for text data storage
csv_vector_db Settings for CSV data storage

**Ingesting Data**-
Document Data (TXT)
The module milvus_db_doc.py handles text data ingestion. When you run the project:
It loads the ebook and summary files.
Splits the ebook into paragraphs, assigns chapter numbers (using chapter headers), and generates embeddings.
Data is stored in Milvus using chapter-based partitions.

CSV Data
The module milvus_db_csv.py handles CSV ingestion:
Reads CSV files via Pandas.
Generates embeddings from selected columns (or the entire row).
Stores the row data (serialized as JSON) in the text field of Milvus.

**Operations**-
After ingesting data, you can perform:
Insert: Add new records (TXT or CSV) to the collection.
Delete: Remove records by chapter (TXT) or by record ID (CSV).
Update: Modify existing records.
Query: Perform similarity search queries interactively.

Run the script with:
python main.py

This script will:
Load the embedding model
Process either TXT or CSV data, depending on config.py
Ingest data into Milvus
Allow insert, delete, and update operations
Perform similarity search queries

Querying Similar Data-
When prompted, enter a query string to retrieve similar records.
Results will be displayed based on vector similarity.

**Evaluation**-
The project includes an evaluation module that computes retrieval metrics:

Chapter-Level Metrics:
Precision@k: 1/k if the top retrieved chapter is relevant; otherwise 0.
Recall@k: 1 if the top retrieved chapter is relevant; otherwise 0.
F1 Score: Harmonic mean of precision and recall.

Paragraph-Level Metrics:
Precision@k: Fraction of top-k retrieved paragraphs that are relevant.
Recall@k: Fraction of total relevant paragraphs retrieved in top-k.
F1 Score: Harmonic mean of precision and recall.
MRR: Reciprocal rank of the first relevant paragraph.

The evaluation is implemented in evaluation.py using the class EvaluateDocumentDB. Ground truth data is stored in evaluation_data.json (with both chapter and paragraph fields as lists).

Run the evaluation with:
python evaluation.py


**Database Structure**
For Text Data (milvus_db_doc.py)
Each record consists of:
id: Unique identifier.
vector: Embedding representation.
text: Original text data.
chapter: Chapter number.
record_type: Either "summary" or "paragraph".

For CSV Data (milvus_db_csv.py)
Each record consists of:
id: Unique identifier.
vector: Embedding representation.
text: JSON-formatted string of the CSV row metadata.

Logging
Logs are stored in the logs/ directory, capturing operations, errors, and queries.
