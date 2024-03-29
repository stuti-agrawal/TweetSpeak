# TweetSpeak

This project implements a sentiment analysis tool for finding out the popular opinion on a given a topic.


## Progress so far

-> We are using a Twitter Sentiment Analysis Kaggle dataset to get an annotated twitter sentiment analysis dataset
-> We are preprocessing the dataset and created an inverted index on it, and getting an extensive vocabulary for it. We use Hadoop to get the inverted index
-> The user can input a search query when prompted, and using BM25 normalisation we find the most relevant tweets to the query and then run Spacy's sentiment analysis on it to get the overall sentiment and the top 5 positive and negative tweets about the query

## Steps to run 
 
 For a demo of the working pipeline, from the root directory run

```
cd src
python main.py

```

You should see where you can enter the query:
![alt text](image.png)

After you enter the query this is the response:
![alt text](image-1.png)





