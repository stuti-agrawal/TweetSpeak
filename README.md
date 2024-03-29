# TweetSpeak

This project implements a sentiment analysis tool for finding out the popular opinion on a given a topic.


## Progress so far

-> We are using a Twitter Sentiment Analysis Kaggle dataset to get an annotated twitter sentiment analysis dataset

-> We are preprocessing the dataset and created an inverted index on it, and getting an extensive vocabulary for it. We use Hadoop to get the inverted index

-> The user can input a search query when prompted, and using BM25 normalisation we find the most relevant tweets to the query and then run Spacy's sentiment analysis on it to get the overall sentiment and the top 5 positive and negative tweets about the query

