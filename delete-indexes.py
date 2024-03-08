from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os

load_dotenv()

# Define the Elasticsearch server host
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')
ELASTICSEARCH_INDEX_NAME = os.getenv('ELASTICSEARCH_INDEX_NAME')
# Initialize Elasticsearch client
es = Elasticsearch(ELASTICSEARCH_URL)

# Define the index name you want to delete
index_name = 'your_index_name'

# Delete the entire index
try:
    response = es.indices.delete(index=ELASTICSEARCH_INDEX_NAME)
    print(f"Index {index_name} deleted successfully.")
except Exception as e:
    print(f"Error deleting index: {e}")
