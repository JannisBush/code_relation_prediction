import json


# Create a summary of a tweet, only showing relevant fields.
def summarize(tweet, extra_fields = None):
    new_tweet = {}
    for field, value in tweet.items():
        if field in ["full_text", "id_str", "screen_name", "retweet_count", "favorite_count", "in_reply_to_status_id_str", "in_reply_to_screen_name", "in_reply_to_user_id_str"] and value is not None:
            new_tweet[field] = value
        elif extra_fields and field in extra_fields:
            new_tweet[field] = value
        elif field in ["retweeted_status", "quoted_status", "user"]:
            new_tweet[field] = summarize(value)
    return new_tweet


with open("./black_history/tweets.json", "r", encoding="utf-8") as infile:
    with open("./black_hist_tweets.json", "w", encoding="utf-8") as outfile:
        for line in infile:
            tweet = summarize(json.loads(line))
            outfile.write(json.dumps(tweet)+"\n")