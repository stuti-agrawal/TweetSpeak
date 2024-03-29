import json

# inverted index dictionary with termId (same as term name): {docId: freq}
from collections import defaultdict
inverted_index_web = defaultdict(dict)

with open('results/part-00000', 'r') as file:
    for line in file:
        # Parse the output line
        line = line.strip()
        word, doc_counts = line.split('\t', 1)
        # Convert the string representation of list of tuples to actual list of tuples
        doc_counts_web = eval(doc_counts)
        # Sort the document entries by docId to create a sorted run
        sorted_docs_web = sorted(doc_counts_web, key=lambda x: x[0])
        inverted_index_web[word] = defaultdict(int)

        # Populate the inverted index
        for docId, freq in sorted_docs_web:
            inverted_index_web[word][docId] = freq
# vocabulary (200 most popular words)
from collections import Counter
word_counts_web = Counter()
for word,doc_counts_dict in inverted_index_web.items():
  word_counts_web[word] = sum(doc_counts_dict.values())

vocabulary_web = list(word_counts_web.keys())[:15000]
print(vocabulary_web)

queries_web = ["Microsoft work"]

with open('results/inverted_index_web.json', 'w') as f:
    json.dump(inverted_index_web, f)
with open('results/vocabulary_web.json', 'w') as f:
    json.dump(vocabulary_web, f)

# save queries and custom queries
with open('results/queries_web.json', 'w') as f:
    json.dump(queries_web, f)