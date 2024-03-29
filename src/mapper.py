#Code based on https://blog.devgenius.io/big-data-processing-with-hadoop-and-spark-in-python-on-colab-bff24d85782f
import sys
import io
import re
import nltk
import pandas as pd
# stemming library
from nltk.stem import PorterStemmer
nltk.download('stopwords',quiet=True)
from nltk.corpus import stopwords
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
# initialize stemmer
ps = PorterStemmer()

stop_words = set(stopwords.words('english'))
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='latin1')



docid = 1 #actually line id
# associative array to hold the word counts
DOCID = 0
docids = []

for line in input_stream:
  word_counts = {}
  try:
    DOCID = line.split(",")[0]
    line = line.split(",")[-1]
    if DOCID in docids:
      continue
  except:
    print("error")
  # remove leading and trailing whitespace
  line = line.strip()
  # remove non-alphanumeric characters
  line = re.sub(r'[^\w\s]', ' ',line)
  # remove digits
  line = re.sub(r'\d+', ' ', line)
  # convert to lowercase
  line = line.lower()
  # remove punctuation
  for x in line:
    if x in punctuations:
      line=line.replace(x, " ")
  # split the line into words
  words=line.split()
  for word in words:
    # remove stopwords
    if word not in stop_words:
      # stem the word
      word = ps.stem(word)
      # remove excess whitespace
      word = word.strip()
      # return stemmedword\t1
      word_counts[word] = word_counts.get(word, 0) + 1
  for word, count in word_counts.items():
    # return a (key,value) pair of word, (docid, count)
    print('%s\t%s\t%s' % (word, DOCID, count))
    
  docid +=1 
