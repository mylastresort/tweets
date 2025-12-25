from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import tokenize_tweet, stem_tokens
from cleaning import clean_tweets
from sklearn.feature_extraction.text import TfidfVectorizer

def custom_tokenizer(text):
    """Custom tokenizer that returns pre-tokenized input."""
    # using stemming by default
    return stem_tokens(tokenize_tweet(text))

def count_vectorize(tweets, text_column, cleaner = clean_tweets):
    """Create a Bag of Words representation of the cleaned tweets in the dataframe."""
    vectorizer = CountVectorizer(tokenizer=custom_tokenizer, lowercase=False)
    bow_matrix = vectorizer.fit_transform(cleaner(tweets, text_column=text_column)[text_column])
    return vectorizer, bow_matrix

def tfidf_vectorize(tweets, text_column, cleaner = clean_tweets):
    """Create a TF-IDF representation of the cleaned tweets in the dataframe."""
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(cleaner(tweets, text_column=text_column))
    return vectorizer, tfidf_matrix