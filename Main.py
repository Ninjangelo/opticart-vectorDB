# IMPORTS
import os
import psycopg2
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# SAMPLE DATA
texts = [
    "Type: Desktop, OS: Ubuntu, GPU: NVIDIA, CPU: AMD, RAM: 64GB, SSD: 2TB",
    "Type: Desktop, OS: Linux Mint, GPU: NVIDIA, CPU: AMD, RAM: 64GB, SSD: 2TB",
    "Type: Desktop, OS: Manjaro, GPU: NVIDIA, CPU: AMD, RAM: 64GB, SSD: 2TB",
    "Type: Desktop, OS: Windows, GPU: NVIDIA, CPU: AMD, RAM: 64GB, SSD: 2TB",
    "Type: Desktop, OS: Windows, GPU: NVIDIA, CPU: Intel, RAM: 32GB, SSD: 1TB",
    "Type: Desktop, OS: Fedora, GPU: AMD, CPU: AMD, Intel: 16GB, SSD: 1TB",
    "Type: Desktop, OS: Windows, GPU: NVIDIA, CPU: AMD, RAM: 16GB, SSD: 2TB",
    "Type: Desktop, OS: Windows, GPU: AMD, CPU: AMD, RAM: 16GB, SSD: 1TB",
    "Type: Desktop, OS: Ubuntu, GPU: NVIDIA, CPU: AMD, RAM: 32GB, SSD: 1TB",
    "Type: Laptop, OS: Windows, GPU: NVIDIA, CPU: Intel, RAM: 16GB, SSD: 1TB",
    "Type: Laptop, OS: Ubuntu, GPU: AMD, CPU: AMD, RAM: 16GB, SSD: 500GB",
    "Type: Laptop, OS: Mac OS, GPU: NVIDIA, CPU: AMD, RAM: 16GB, SSD: 1TB",
]

# Model: HuggingFace - all-MiniLM-L6-v2 
# Generating Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings_list = []

for text in texts:
    embeddings_list.append(embeddings.embed_query(text))

print(f"Embedding length: {len(embeddings_list[0])}")
print("Embeddings Generated Successfully.")

