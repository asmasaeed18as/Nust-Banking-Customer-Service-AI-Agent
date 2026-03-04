import json
import os
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger
from typing import List, Dict

try:
    from src.logging_config import setup_logger
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from src.logging_config import setup_logger

setup_logger("logs/indexing.log")

class NustBankIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Initializing Embedder with model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.metadata = []

    def load_processed_data(self, file_path: str):
        logger.info(f"Loading processed data from {file_path}")
        with open(file_path, 'r') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded {len(self.metadata)} data chunks.")

    def create_index(self):
        logger.info("Starting embedding generation...")
        
        # Prepare text for embedding: "Question: ... Answer: ..."
        texts = [f"Question: {item['question']} Answer: {item['answer']}" for item in self.metadata]
        
        # This will take some time depending on hardware
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        dimension = embeddings.shape[1]
        logger.info(f"Embedding generation complete. Dimension: {dimension}")

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        logger.success(f"FAISS index created with {self.index.ntotal} vectors.")

    def save_index(self, folder_path: str = "data/vector_store"):
        os.makedirs(folder_path, exist_ok=True)
        
        index_path = os.path.join(folder_path, "bank_faiss_index.bin")
        meta_path = os.path.join(folder_path, "bank_metadata.json")
        
        faiss.write_index(self.index, index_path)
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
            
        logger.success(f"Vector store saved to {folder_path}")

if __name__ == "__main__":
    indexer = NustBankIndexer()
    indexer.load_processed_data("data/processed/bank_knowledge_chunks.json")
    indexer.create_index()
    indexer.save_index()
