import numpy as np
import json
import codecs


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


# Print out a tweet, with optional colorizing of selected fields.
def dump(tweet, colorize_fields=None, summarize_tweet=True):
    colorize_field_strings = []
    for line in json.dumps(summarize(tweet) if summarize_tweet else tweet, indent=4, sort_keys=True).splitlines():
        colorize = False
        for colorize_field in colorize_fields or []:
            if "\"{}\":".format(colorize_field) in line:
                print("\x1b" + line + "\x1b")
                break
        else:
            print(line)


tweet_dict = {}
org_reply_list_ids = []
org_reply_list_text = []
with open("../jk_rowling_tweets/tweets.json") as infile:
    for line in infile:
        tweet = summarize(json.loads(line))
        tweet_dict[tweet["id_str"]] = tweet

print(len(tweet_dict))
count = 0

for tweet_id, tweet in tweet_dict.items():
    try:
        if tweet["in_reply_to_status_id_str"] is not None:
            try:
                org_tweet = tweet_dict[tweet["in_reply_to_status_id_str"]]
                org_reply_list_ids.append((org_tweet["id_str"], tweet_id))
                org_reply_list_text.append((org_tweet["full_text"], tweet["full_text"]))
            except KeyError as e:
                count += 1
    except KeyError as e:
        pass

print(count)
print(len(org_reply_list_ids))
print(len(org_reply_list_text))
#print(org_reply_list_text)

print(len(set([a for a,_ in org_reply_list_ids])))

with open("../jk_rowling_tweets/reply_ids.json", "w") as outfile:
    json.dump(org_reply_list_ids, outfile)

with open("../jk_rowling_tweets/reply_texts.json", "w") as outfile:
    json.dump(org_reply_list_text, outfile)



