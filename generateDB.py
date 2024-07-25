import os
import faiss
import numpy as np
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.document_loaders.base import Document

class VectorStore:
    def __init__(self, name):
        self.name = name
        self.index = faiss.IndexFlatL2(768)  # 768 is the dimension of BAAI/bge-base-en-v1.5 embeddings
        self.collection = {}

    def populate_vectors(self, file_path):
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create Document
        doc = Document(page_content=text)
        
        # Split Document
        splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
        chunked_docs = splitter.split_documents([doc])
        
        # Embed and add to FAISS
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        for i, chunk in enumerate(chunked_docs):
            embedding = embeddings.embed_documents([chunk.page_content])
            embedding_np = np.array(embedding)  # Convert to numpy array
            self.index.add(embedding_np)
            self.collection[str(i)] = chunk.page_content

    def save(self, faiss_path, collection_path):
        faiss.write_index(self.index, faiss_path)
        with open(collection_path, 'w', encoding='utf-8') as f:
            json.dump(self.collection, f)

    def load(self, faiss_path, collection_path):
        self.index = faiss.read_index(faiss_path)
        with open(collection_path, 'r', encoding='utf-8') as f:
            self.collection = json.load(f)

# Ensure the file path is correct
file_path = r"data\data_bremen.txt"
faiss_path = r"data\faiss_index2.bin"
collection_path = r"data\collection2.json"


# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} does not exist.")

# Create and populate the vector store
vector_store = VectorStore("embedding_vector")
vector_store.populate_vectors(file_path)

# Save the FAISS index and collection
vector_store.save(faiss_path, collection_path)

# Optionally, load the FAISS index and collection
vector_store.load(faiss_path, collection_path)
