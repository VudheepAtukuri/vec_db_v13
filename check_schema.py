from pymilvus import connections, Collection

# Connect to Milvus using the alias "default"
connections.connect(alias="default", host="localhost", port="19530")

# Define the summary and paragraph collections
summary_collection_name = "summary_collection27"
paragraph_collection_name = "paragraph_collection27"

# Retrieve and print the schema for your collections
summary_collection = Collection(summary_collection_name)
print(f"Summary Collection Schema: {summary_collection.schema}")

paragraph_collection = Collection(paragraph_collection_name)
print(f"Paragraph Collection Schema: {paragraph_collection.schema}")


summary_records = summary_collection.query(expr="id>=0", output_fields=["id", "text"])
if summary_records:
    for record in summary_records:
        print(f"Summary ID: {record['id']}, Text (first 100 chars): {record['text'][:100]}")
else:
    print("No records found in the summary collection.")

# Query paragraph collection to get IDs associated with Chapter 1
chapter = "1"
query_expr = f"chapter == '{chapter}'"
paragraph_results = paragraph_collection.query(expr=query_expr, output_fields=["id", "chapter", "text"])

if paragraph_results:
    paragraph_ids = [record["id"] for record in paragraph_results]
    print(f"\nParagraph IDs for Chapter (Summary ID) {chapter}: {paragraph_ids}")
    for rec in paragraph_results:
        print(f"Record: ID: {rec['id']}, Chapter: {rec['chapter']}, Text (first 100 chars): {rec['text'][:100]}")
else:
    print(f"\nNo paragraphs found for Chapter (Summary ID) {chapter}.")

# List all partitions in the paragraph collection
partitions = paragraph_collection.partitions
print(f"\nAll partitions in the paragraph collection: {[p.name for p in partitions]}")

# Query paragraph collection partition by partition
for partition in partitions:
    partition_name = partition.name
    print(f"\n--- Retrieving Paragraph IDs for partition: {partition_name} ---")
    partition_results = paragraph_collection.query(
        expr="id>=0", 
        output_fields=["id", "chapter", "text"], 
        partition_names=[partition_name]
    )
    if partition_results:
        partition_ids = [record["id"] for record in partition_results]
        print(f"Paragraph IDs in partition '{partition_name}': {partition_ids}")
        for rec in partition_results:
            print(f"Record: ID: {rec['id']}, Chapter: {rec['chapter']}, Text (first 100 chars): {rec['text'][:100]}")
    else:
        print(f"No paragraphs found in partition '{partition_name}'.")

# Disconnect from Milvus
connections.disconnect(alias="default")
