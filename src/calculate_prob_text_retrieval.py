# PART 4
import sys
import json
import os
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from collections import Counter, defaultdict
import numpy as np

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

def jm_smoothing(tf, doc_len, lam, vocab, query_tf, query_words, C, inverted_index):
    relevance = 0
    for query_word in query_words:
        if query_word not in vocab:
            continue
        c_w_q = query_tf[query_word]
        if query_word not in tf:
            c_w_d = 0
        else:
            c_w_d = tf[query_word]
        smoothing = sum(inverted_index[query_word].values())/C
        interpolation = ((1-lam)/lam)
        relevance += c_w_q*(np.log(1+(interpolation*c_w_d)/(doc_len*smoothing)))
    return relevance
    

def main(inverted_index_path, vocabulary_path, queries_path):
    with open(inverted_index_path, 'r') as f:
        inverted_index = json.load(f)
    with open(vocabulary_path, 'r') as f:
        vocab = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)

    # if output directory does not exist, create it
    if not os.path.exists("outputs/jm"):
        os.makedirs("outputs/jm")

    # Jelinek-Mercer Smoothing parameter
    lamba = 0.1
    tf = defaultdict(dict)
    doc_len_dict = defaultdict(int)

    for word, doc_counts_dict in inverted_index.items():
        for docId, count in doc_counts_dict.items():
            tf[docId][word] = count
            doc_len_dict[docId] += count
    

    # process queries
    for query in queries:
        query_words = process_text(query)
        query_tf = Counter(query_words)
        query_words = [word for word in query_tf.elements() if word in vocab]
        doc_relevances = {}

        for docId, doc_len in doc_len_dict.items():
            C = sum(doc_len_dict.values())
            doc_relevances[docId] = jm_smoothing(tf[docId], doc_len, lamba, vocab, query_tf, query_words, C, inverted_index)
        # Sort the documents by relevance to return all non-zero relevance documents
        doc_relevances = {k: v for k, v in sorted(doc_relevances.items(), key=lambda item: item[1], reverse=True) if v > 0}
        print(doc_relevances)
        # Example: Write something to the output file
        with open(f"outputs/jm/{query}_JM_output.txt", "w") as f:
            f.write(f"Query: {query}\n")
            for docId, relevance in doc_relevances.items():
                f.write(f"{docId}\t{relevance}\n")
        




if __name__ == "__main__":
    inverted_index_path = sys.argv[1]
    vocabulary_path = sys.argv[2]
    queries_path = sys.argv[3]
    main(inverted_index_path, vocabulary_path, queries_path)
