import spacy
import json
from calculate_bm25 import main as bm25
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# load the tweets json
tweets = {}
with open('../dataset/twitter_training.json', 'r') as f:
    tweets = json.load(f)

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

print("Enter your query: ")
query = input()

# save the query in a json file
with open('../results/query.json', 'w') as f:
    json.dump([query], f)

# get relevant tweet ids using BM25
relevant_tweets = bm25('../results/inverted_index_web.json', '../results/vocabulary_web.json', '../results/query.json')

tweets_score_list = []
overall_sentiment = 0
# get the tweets
for tweet_id in relevant_tweets:
    tweet = tweets[tweet_id]["tweet"]
    doc = nlp(tweet)
    tweets_score_list.append((tweet_id, doc._.polarity))
    overall_sentiment += doc._.polarity

# sort the tweets by polarity
tweets_score_list = sorted(tweets_score_list, key=lambda x: x[1], reverse=True)

# return overall sentiment and the top 5 positive and negative tweets
if overall_sentiment > 0:
    print(f"Overall sentiment: Positive")
else:
    print(f"Overall sentiment: Negative")

print("\n")

print("Top 5 positive tweets:")
for i in range(5):
    print(tweets[tweets_score_list[i][0]]["tweet"])

print("\n")

print("Top 5 negative tweets:")
for i in range(5):
    print(tweets[tweets_score_list[-i-1][0]]["tweet"])

print("\n")






