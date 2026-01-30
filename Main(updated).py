# IMPORTS
import os
import psycopg2
import numpy as np
import hashlib
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from lanchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# SAMPLE DATA
# COMPUTER SAMPLE TEXTS
"""
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
"""
# SAMPLE TEXTS ALTERNATIVE
texts = [
    "My computer was made by Apple",
    "My favorite fruit is Apple",
    "My Adam's Apple hurts",
    "My knee seems odd today",
    "My laptop is running on Arch Linux",
    "My breakfast usually includes berries",
]


# Model: HuggingFace - all-MiniLM-L6-v2
# EMBEDDINGS MODEL
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings_list = []

# CHAT MODEL
llm = ChatOllama(model="llama3")

for text in texts:
    embeddings_list.append(embeddings.embed_query(text))

conn = psycopg2.connect("dbname=vectortutorialdb user=postgres password=Longganisa143!")
cur = conn.cursor()

cur.execute("TRUNCATE TABLE items;")

for i in range(len(embeddings_list)):
    embedding = embeddings_list[i]
    content = texts[i]
    cur.execute("INSERT INTO items (content, embedding) VALUES (%s, %s)", (content, embedding))

conn.commit()
cur.close()
conn.close()

# QUERY NO.1
#new_text = "Type: Desktop, OS: Arch Linux, GPU: NVIDIA, CPU: AMD, RAM: 64GB, SSD: 2TB"
#QUERY NO.2
new_text = "My dinner would benefit from adding Apples to it"
print(f"\nUser Question: {new_text}")
new_embedding = embeddings.embed_query(new_text)

conn = psycopg2.connect("dbname=vectortutorialdb user=postgres password=Longganisa143!")
cur = conn.cursor()

cur.execute("""SELECT content
FROM items
ORDER BY embedding <-> %s::vector
LIMIT 5 
""", (new_embedding,))

results = cur.fetchall()
conn.close()

retrieved_context = "\n".join([row[0] for row in results])

print(f"\n--- DB found these facts ---\n{retrieved_context}\n----------------------------------")

# GENERATING RESPONSE
prompt = f"""
You are a helpful assistant. Answer the user's question using ONLY the context provided below.

Context:
{retrieved_context}

Question:
{new_text}
"""

print("Llama is thinking...\n")

# WAITING FOR RESPONSE GENERATION
"""
response = llm.invoke(prompt)
print(f"\nLlama Response:\n{response.content}")
"""

print("\nLlama Response:")
for chunk in llm.stream(prompt):
    print(chunk.content, end="", flush=True)

print("\n")


# SIMILARITY SEARCH
#cur.execute("""SELECT id, content
#FROM items
#ORDER BY embedding <-> %s::vector
#LIMIT 5 
#""", (new_embedding,))

#results = cur.fetchall()
#for row in results:
    #print(row)

#conn.close()


"""Thank you for the response Gemini.

I can declare that the console stream was successful.
"""
