# Soba Compressor / Embedder

`Soba Compressor` supports the reading and sentence splitting of `*.pdf` and `*.txt` files provided within a directory. It performs chunking of the retrieved data based on `sentence + word count` and generates a `compressed_*.txt` file in the `soba-embedder/compressed/*` directory.

The compressed files are then subsequently passed to the `Soba Embedder` which performs embedding to convert it to a vector database for subsequent reference in a KNN search. By default, the embeddings are not converted to `Tensor` format, however you may choose to customize the settings in the `main.py` file by modifying `Model`'s parameters.

Refer to the concept: [Hugging Face - Advanced Rag](https://huggingface.co/learn/cookbook/advanced_rag)

## Installations

### Check required/supported CUDA Version

```bash
# Running the nvidia-smi command should show details such as the following
nvidia-smi
```

```bash
# Refer to "CUDA Version: 12.4" when performing the next step
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 551.61                 Driver Version: 551.61         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
```

### Installing relevant PyTorch Version (Cuda Support)

PyTorch Cuda Support - [Download Link](https://pytorch.org/get-started/locally/)

### Install punkt

Install punkt if required using NLTK

```python
if __name__== "__main__":
    nltk.download('punkt')
```

## How to use

### Standalone Data Compression

```python
from custom_compressor import GenerateCompressedFiles

"""
Reads all *.pdf and *.txt files from "your_file_directory" and
writes them to "soba-inferer/compressed/*" where * represents
the read file's prefix directory
"""
if __name__== "__main__":
    GenerateCompressedFiles("your_file_directory")
```

### Standalone Data Embedding

```python
from custom_embedder import GenerateAllEmbeddings

"""
Creates individual embedding files (*.pkl) using the 'all-mpnet-base-v2'
model from SentenceTransformer after reading each "compressed_file" from
"your_compressed_file_directory"
"""

if __name__ == "__main__":    
    # Perform Embedding
    all_embeddings = GenerateAllEmbeddings(os.path.dirname(__file__) + '\\compressed\\',            # Input Directory (All Compressed Files)
                                           os.path.dirname(__file__) + '\\embedding\\',             # Output Directory (embeddings.pkl)
                                           SentenceTransformer('all-mpnet-base-v2', device="cuda", )) # Load Embedding Model on GPU
```

For more Sentence Transformer Information - Refer to [Sentence Transformer Pretrained Models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models)