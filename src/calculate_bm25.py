# PART 3
import json
import sys
from collections import defaultdict
import re
import os
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import numpy as np
from collections import Counter

def process_text(text):
     # transform to lower case
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove excess white spaces from the text and get words list
    words_list = word_tokenize(text)
    # Remove stopwords
    # stop_words = stopwords.words('english')
    for i in range(len(words_list)):
      words_list[i] = ps.stem(words_list[i])
    print(words_list)
    return words_list

# BM25 calculations
def doc_length_bm25(words, query_tf, tf, M, doc_freq_words, doc_len, avgdl, b, k):
  relevance = 0
  normalizer = 1-b+((b*doc_len)/avgdl)
  for word in words:
    if word in tf:
      relevance += query_tf[word]*(((k+1)*tf[word])/(tf[word]+k*(normalizer)))*(np.log((1+M)/doc_freq_words[word]))
  return relevance

# getting the relevance of the documents using the information from the inverted index and the queries
def main(inverted_index_path, vocabulary_path, queries_path):
    # Deserialize the JSON strings back into Python objects
    print(inverted_index_path, vocabulary_path, queries_path)

    # use relative path to the file
    with open(inverted_index_path, 'r') as f:
        inverted_index = json.load(f)
    with open(vocabulary_path, 'r') as f:
        vocab = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    tf = defaultdict(dict)
    doc_freq_words = defaultdict(int)
    doc_len_dict = defaultdict(int)
    M = 0

    for word, doc_counts_dict in inverted_index.items():
        for docId, count in doc_counts_dict.items():
            tf[docId][word] = count
            doc_freq_words[word] += 1
            doc_len_dict[docId] += count
    M = len(doc_len_dict)
    avgdl = sum(doc_len_dict.values())/M

    # if output directory does not exist, create it
    import os
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    for query in queries:
        query_words = process_text(query)
        query_tf = Counter(query_words)
        query_words = [word for word in query_tf.elements() if word in vocab]
        doc_relevances = {}
        b = 0.5
        k = 1
        for docId, doc_len in doc_len_dict.items():
            doc_relevances[docId] = doc_length_bm25(query_words, query_tf, tf[docId], M, doc_freq_words, doc_len, avgdl, b, k)
        doc_relevances = {k: v for k, v in sorted(doc_relevances.items(), key=lambda item: item[1], reverse=True) if v > 0}
    

        # SAVING THE OUTPUT TO A FILE
        with open(f"outputs/{query}_bm25_output.txt", "w") as f:
            f.write(f"Query: {query}\n")
            for docId, relevance in doc_relevances.items():
                f.write(f"{docId}\t{relevance}\n")
        return doc_relevances

        

if __name__ == "__main__":
    inverted_index_path = sys.argv[1]
    vocabulary_path = sys.argv[2]
    queries_path = sys.argv[3]
    main(inverted_index_path, vocabulary_path, queries_path)
