import json
dataset_dict = {}
# open the csv file
with open('../dataset/twitter_training.csv', 'r', encoding='utf-8') as file:
    # read the csv file
    for line in file:
        # Parse the output line
        line = line.strip()
        tweet_id, topic, sentiment, tweet = line.split(',', 3)
        if tweet_id not in dataset_dict:
            dataset_dict[tweet_id] = { "topic": topic, "sentiment": sentiment, "tweet": tweet }


# save the dataset
with open('../dataset/training_twitter.json', 'w') as f:
    json.dump(dataset_dict, f)

    

