import os
import nltk
import pickle
import numpy as np
from mattsollamatools import chunker
from sklearn.neighbors import NearestNeighbors

from custom_compressor import FindValidFilesInDirectory

# Perform K-nearest neighbors (KNN) search
def KnnSearch(question_embedding, embeddings, k=5):
    X = np.array([item['embedding'] for article in embeddings for item in article['embeddings']])
    source_texts = [item['source'] for article in embeddings for item in article['embeddings']]
    
    # Fit a KNN model on the embeddings
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(X)

    # In case we have too few embedded data
    if k > knn.n_samples_fit_:
        k = knn.n_samples_fit_
    
    # Find the indices and distances of the k-nearest neighbors
    distances, indices = knn.kneighbors(question_embedding, n_neighbors=k)
    
    # Get the indices and source texts of the best matches
    best_matches = [(indices[0][i], source_texts[indices[0][i]]) for i in range(k)]
    
    return best_matches


def ReplaceDirectory(full_path, old_dir, new_dir):
    path_components = full_path.split(os.sep)
    
    for i, component in enumerate(path_components):
        if component == old_dir:
            path_components[i] = new_dir
            break
    
    # Join the modified path components
    modified_path = os.sep.join(path_components)
    
    return modified_path


def GenerateEmbeddings(compressed_filename, model):
    with open(compressed_filename, "r") as input_doc:
        if input_doc.closed:
            print(compressed_filename + ' could not be opened to be embedded')
            return
        
        # Extract out details for metadata
        filename = os.path.basename(compressed_filename)
        filename_without_extension = os.path.splitext(filename)[0]
        is_online_api = 'online-api' in compressed_filename

        compressed_text = input_doc.read()
        # Skip embeddings if less than 20 words
        # Might want to increase this number
        if len(compressed_text.split()) < 20: # Place this in a config file eventually
            return False, {}
        
        # Use this for the model's encode - convert_to_tensor=True
        # This can be helpful for GPU computation speed however it also
        # takes up a large amount of space
        chunks = chunker(compressed_text)
        embeddings = model.encode(chunks, show_progress_bar=True)

        story = {}
        story['embeddings'] = []
        for (chunk, embedding) in zip(chunks, embeddings):
            item = {}
            item['source'] = chunk
            item['embedding'] = embedding
            item['docname'] = f'online-api\\{filename_without_extension}' if is_online_api else filename_without_extension
            item['sourcelength'] = len(chunk)
            story['embeddings'].append(item)

        return True, story
    

def GenerateAllEmbeddings(folder_directory, outfile_directory, model):
    compressed_files = FindValidFilesInDirectory(folder_directory)

    all_embeddings = []
    for filename in compressed_files:
        print('Performing embedding for - ', filename)

        # New output directory
        # Essentially: __file_dir__\\compressed\\...\\my_compressed_data.txt
        # is converted to __file_dir__\\embedding\\...\\my_compressed_data.pkl
        new_out_dir = filename.replace('\\compressed\\', '\\embedding\\')
        new_out_dir = new_out_dir.replace('.txt', '.pkl')

        status, embeddings = GenerateEmbeddings(filename, model)
        if status == True:
            if new_out_dir != '':
                # Create directory if doesn't exist
                output_folder = os.path.dirname(new_out_dir)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                
                print("New write directory: ", new_out_dir)

                # Write each embedding as a binary *.pkl file
                with open(new_out_dir, 'wb') as f:
                    pickle.dump(embeddings, f)

        print('Completed embedding for - ', filename)

    return all_embeddings