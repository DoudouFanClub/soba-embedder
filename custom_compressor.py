import os
import glob
import nltk
import ollama
import pymupdf
import unicodedata
from pathlib import Path
from nltk.tokenize import sent_tokenize

def FindValidFilesInDirectory(directory):
    pdf_file_list = glob.glob(directory + '/**/*.pdf', recursive=True)
    text_file_list = glob.glob(directory + '/**/*.txt', recursive=True)
    return pdf_file_list + text_file_list


def GetPriorFolderPath(root_dir, file_dir):
    rel_path = os.path.relpath(file_dir, root_dir)
    folder_path = os.path.dirname(rel_path)
    return folder_path.replace("\\", "\\\\")


# Summarize the text provided without losing any core information and keep all code examples, and do not make any assumptions not within the text  |  1000 words
def CompressChunk(model_name, page_data):

    _system_prompt = """I have a series of text chunks extracted from a MAK RTI User Guide Document, and this is only a single portion.
    Can you provide me with a summary that keeps all the key points without losing too much information:
    {}"""

    response = ollama.generate(
        model=model_name,
        prompt=_system_prompt.format(page_data),
        stream=False
    )

    _supporting_prompt = """From this original text chunk:
    {}

    Are there any missing key points or descriptions that are not included within the summary below? If so, could merge the missing points into the existing summary below.
    If the sentences are too long, feel free to separate them into bullet points with additional description behind the bullet point.
    Also, remove any unnecessary text/headers that was not included within the text chunk provided, and do not tell me which points u have added or removed.
    {}"""

    updated_response = ollama.generate(
        model=model_name,
        prompt=_supporting_prompt.format(page_data, response['response']),
        stream=False
    )

    print(updated_response['response'])
    return updated_response['response']


def CompressAndStoreTextData(data, outputdir, chunk_len = 600, write_type = 'w'):
        chunk = []
        word_count = 0
        sentences = sent_tokenize(data)

        # Create the Output Directory if it does not exist
        output_folder = os.path.dirname(outputdir)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for _, sentence in enumerate(sentences):
            words = len(sentence.split())

            if word_count + words <= chunk_len:
                chunk.append(unicodedata.normalize('NFKD', sentence).encode('ascii', errors='ignore').decode('ascii'))
                word_count += words
            else:
                with open(outputdir, write_type) as out_text_doc:
                    long_sentence = ' '.join(chunk)

                    print('\nLong Sentence:\n' + long_sentence + "\n\n")

                    compressed_text = CompressChunk('mistral-nemo-Q8-temp0', long_sentence)
                    actual_compressed_text = compressed_text.split('\n')[2:]
                    compressed_text = ('\n'.join(actual_compressed_text))
                    compressed_text = unicodedata.normalize('NFKD', compressed_text).encode('ascii', errors='ignore').decode('ascii')
                    out_text_doc.write(compressed_text + '\n\n')
                    chunk = [sentence]
                    word_count = words
                    write_type = 'a'
                    out_text_doc.close()

        if chunk:
            with open(outputdir, write_type) as out_text_doc:
                long_sentence = ' '.join(chunk)

                print('\nLong Sentence:\n' + long_sentence + "\n\n")

                compressed_text = CompressChunk('mistral-nemo-Q8-temp0', long_sentence)
                actual_compressed_text = compressed_text.split('\n')[2:]
                compressed_text = ('\n'.join(actual_compressed_text))
                compressed_text = unicodedata.normalize('NFKD', compressed_text).encode('ascii', errors='ignore').decode('ascii')
                out_text_doc.write(compressed_text + '\n\n')
                out_text_doc.close()


def CompressPdf(inputdir, outputdir, validpages):
    data = ''
    input_doc = pymupdf.open(inputdir)
    
    for i, page in enumerate(input_doc):
        if i <= validpages[0]:
            continue
        elif i > validpages[1]:
            break
        data += unicodedata.normalize('NFKD', page.get_text()).encode('ascii', errors='ignore').decode('ascii') + '\n'

    CompressAndStoreTextData(data, outputdir)
    input_doc.close()


def CompressTxt(filename, outdir):
    with open(filename, 'r') as input_doc:
        CompressAndStoreTextData(input_doc.read(), outdir)
        input_doc.close()


# def GenerateCompressedFiles(directory_to_compress):
#     file_list = FindValidFilesInDirectory(directory_to_compress)

#     print("List of files in listed directory: ", file_list)

#     for filename in file_list:
#         filename_postfix = filename.split(".")[1]
#         leading_folder_dir = GetPriorFolderPath(directory_to_compress, filename)
#         print("Currently Compressing: ", filename)
#         if filename_postfix == 'txt':
#             CompressTxt(filename, os.path.dirname(__file__) + '\\compressed\\' + leading_folder_dir + "\\")
#         elif filename_postfix == 'pdf':
#             CompressPdf(filename, os.path.dirname(__file__) + '\\compressed\\' + leading_folder_dir + "\\", 15)


# if __name__ == "__main__":
#     nltk.download('punkt')
#     GenerateCompressedFiles(os.path.dirname(__file__) + "\\data")