import os
import nltk
import json
from sentence_transformers import SentenceTransformer
from custom_embedder import GenerateAllEmbeddings
from custom_compressor import CompressPdf, CompressTxt

if __name__ == "__main__":
    # nltk.download('punkt')
    
    # Perform Embedding
    all_embeddings = GenerateAllEmbeddings(os.path.dirname(__file__) + '\\compressed\\',            # Input Directory (All Compressed Files)
                                           os.path.dirname(__file__) + '\\embedding\\',             # Output Directory (embeddings.pkl)
                                           SentenceTransformer('all-mpnet-base-v2', device="cuda")) # Load Embedding Model on GPU