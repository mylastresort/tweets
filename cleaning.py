def remove_mentions(tweet):
    """Remove Twitter mentions from a tweet."""
    return ' '.join(word for word in tweet.split() if not word.startswith('@'))

def remove_urls(tweet):
    """Remove URLs from a tweet."""
    return ' '.join(word for word in tweet.split() if not word.startswith('http'))

def remove_hashtags(tweet):
    """Remove hashtags from a tweet."""
    return ' '.join(word for word in tweet.split() if not word.startswith('#'))

def remove_punctuation(tweet):
    """Remove punctuation from a tweet."""
    import string
    return tweet.translate(str.maketrans('', '', string.punctuation))

def convert_to_lowercase(tweet):
    """Convert all characters in a tweet to lowercase."""
    return tweet.lower()

# create a cleaner dataset
def clean_tweet(tweet):
    tweet = str(tweet)
    tweet = remove_mentions(tweet)
    tweet = remove_urls(tweet)
    tweet = remove_hashtags(tweet)
    tweet = remove_punctuation(tweet)
    tweet = convert_to_lowercase(tweet)
    return tweet

def clean_tweets(dataframe, text_column = 'tweet'):
    """Apply cleaning functions to a dataframe column containing tweets."""
    dataframe[text_column] = dataframe[text_column].apply(clean_tweet)
    dataframe = dataframe.dropna(subset=[text_column]).reset_index(drop=True)
    return dataframe
