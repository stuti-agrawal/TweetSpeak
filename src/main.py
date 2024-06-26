import spacy
import json
from calculate_bm25 import main as bm25
from metrics import calculate_metrics
import spacy
import time
import numpy as np
from transformers import pipeline
nlp = pipeline("sentiment-analysis")

# load the tweets json
tweets = {}
with open('../dataset/twitter_training.json', 'r') as f:
    tweets = json.load(f)

print("Enter your query: ")
query = input()

# save the query in a json file
with open('../results/query.json', 'w') as f:
    json.dump([query], f)

# get relevant tweet ids using BM25
relevant_tweets = bm25('../results/inverted_index_web.json', '../results/vocabulary_web.json', '../results/query.json')
if len(relevant_tweets) == 0:
    print("No relevant tweets found")
    exit()

# calculate estimated overall sentiment
overall_sentiment_score = 0
for tweet_id in relevant_tweets:
    if tweets[tweet_id]["sentiment"] == "Positive":
        overall_sentiment_score += 1
    elif tweets[tweet_id]["sentiment"] == "Negative":
        overall_sentiment_score -= 1
print("Estimated overall sentiment: ", "Positive" if overall_sentiment_score > 0 else "Negative")

# tweets_score_list = []
# overall_sentiment = 0
# # get the tweets
# for tweet_id in relevant_tweets:
#     tweet = tweets[tweet_id]["tweet"]
#     doc = nlp(tweet)
#     tweets_score_list.append((tweet_id, doc._.polarity))
#     overall_sentiment += doc._.polarity

def print_top_tweets(positive_tweets_score_list, negative_tweets_score_list, overall_sentiment):
    # sort the tweets by polarity
    positive_tweets_score_list = sorted(positive_tweets_score_list, key=lambda x: x[1], reverse=True)
    negative_tweets_score_list = sorted(negative_tweets_score_list, key=lambda x: x[1], reverse=True)
    

    # return overall sentiment and the top 5 positive and negative tweets
    if overall_sentiment > 0:
        print(f"Overall sentiment: Positive")
    else:
        print(f"Overall sentiment: Negative")

    print("\n")

    print("Top 5 positive tweets:")
    for i in range(5):
        print(tweets[positive_tweets_score_list[i][0]]["tweet"])

    print("\n")

    print("Top 5 negative tweets:")
    for i in range(5):
        print(tweets[negative_tweets_score_list[i][0]]["tweet"])

    print("\n")

# Example function to analyze sentiment
def analyze_sentiment_transformers(text):
    start_time = time.time()
    response = nlp(text)
    end_time = time.time()
    response_time = end_time - start_time
    sentiment = response[0]['label'] = response[0]['label'].lower()
    confidence = response[0]['score']
    return sentiment, confidence, response_time

predicted_sentiment = []
actual_sentiment = []
positive_tweets_score_list = []
negative_tweets_score_list = []
overall_sentiment_score = 0
response_times = []

# # get the tweets
for tweet_id in relevant_tweets:
    if tweets[tweet_id]["sentiment"] != "Positive" and tweets[tweet_id]["sentiment"] != "Negative" and tweets[tweet_id]["sentiment"] != "Neutral":
        continue
    tweet = tweets[tweet_id]["tweet"]
    sentiment, confidence, response_time = analyze_sentiment_transformers(tweet)
    response_times.append(response_time)
    predicted_sentiment.append(sentiment.lower())
    actual_sentiment.append(tweets[tweet_id]["sentiment"].lower())
    if sentiment == "positive":
        positive_tweets_score_list.append((tweet_id, confidence))
        overall_sentiment_score += confidence
    else:
        negative_tweets_score_list.append((tweet_id, confidence))
        overall_sentiment_score -= confidence

calculate_metrics(actual_sentiment, predicted_sentiment, response_times)
print_top_tweets(positive_tweets_score_list, negative_tweets_score_list, overall_sentiment_score)

