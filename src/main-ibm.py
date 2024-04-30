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
    for i in range(10):
        print(tweets[positive_tweets_score_list[i][0]]["tweet"])

    print("\n")

    print("Top 5 negative tweets:")
    for i in range(10):
        print(tweets[negative_tweets_score_list[i][0]]["tweet"])

    print("\n")

from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
import time
import numpy as np

authenticator = IAMAuthenticator('api-key')
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2022-04-07',
    authenticator=authenticator)

natural_language_understanding.set_service_url('url')
# Example function to analyze sentiment
def analyze_sentiment_ibm(text):
    start_time = time.time()
    response = natural_language_understanding.analyze(
        text=text,
        features=Features(sentiment=SentimentOptions())
    ).get_result()
    end_time = time.time()
    response_time = end_time - start_time
    sentiment = response['sentiment']['document']['label']
    confidence = response['sentiment']['document']['score']
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
    sentiment, confidence, response_time = analyze_sentiment_ibm(tweet)
    response_times.append(response_time)
    sentiment = sentiment.lower()
    predicted_sentiment.append(sentiment)
    actual_sentiment.append(tweets[tweet_id]["sentiment"].lower())
    if sentiment == "positive":
        positive_tweets_score_list.append((tweet_id, confidence))
        overall_sentiment_score += confidence
    elif sentiment == "negative":
        negative_tweets_score_list.append((tweet_id, confidence))
        overall_sentiment_score -= confidence

calculate_metrics(actual_sentiment, predicted_sentiment, response_times)
print_top_tweets(positive_tweets_score_list, negative_tweets_score_list, overall_sentiment_score)
