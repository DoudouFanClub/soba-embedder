import os
import nltk
import json
from sentence_transformers import SentenceTransformer
from custom_embedder import GenerateAllEmbeddings
from custom_compressor import CompressPdf, CompressTxt

if __name__ == "__main__":
    # nltk.download('punkt')

    # # Perform Compression
    # files_to_compress = []
    # with open('config\\files_to_compress.json', 'r') as f:
    #     files_to_compress = json.load(f)
    #     f.close()

    # for i, cfg in enumerate(files_to_compress):
    #     # Clearer Names
    #     input_path = os.path.dirname(__file__) + cfg['InputDirectory']
    #     output_path = os.path.dirname(__file__) + cfg['OutputDirectory']
    #     valid_page_range = [int(x) for x in cfg['IncludeRange'].split()]

    #     # Display Current File
    #     print("Compressing File ", i, "/", len(files_to_compress), ": ", cfg['InputDirectory'])

    #     # Process Specific File Type
    #     if ".txt" in input_path:
    #         CompressTxt(input_path, output_path)
    #     elif ".pdf" in input_path:
    #         CompressPdf(input_path, output_path, valid_page_range)
    #     else:
    #         print("File type not supported - ", os.path.basename( cfg['InputDirectory'] ))
        

    # Perform Embedding
    all_embeddings = GenerateAllEmbeddings(os.path.dirname(__file__) + '\\compressed\\', # Input Directory (All Compressed Files)
                                           os.path.dirname(__file__) + '\\embedding\\',  # Output Directory (embeddings.pkl)
                                           SentenceTransformer('all-mpnet-base-v2'))     # Embedding Model