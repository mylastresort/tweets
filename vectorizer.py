from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import tokenize_tweet
from cleaning import clean_tweets
from sklearn.feature_extraction.text import TfidfVectorizer


def custom_tokenizer(method=None):
    """Custom tokenizer that returns pre-tokenized input."""

    def cb(text):
        tokenized_text = tokenize_tweet(text)
        if method is None:
            return tokenized_text
        return method(tokenized_text)

    return cb


def count_vectorize(
    tweets, text_column, cleaner=clean_tweets, tokenizer=None, binary=False
):
    """Create a Bag of Words representation of the cleaned tweets in the dataframe."""
    vectorizer = CountVectorizer(
        tokenizer=custom_tokenizer(method=tokenizer), binary=binary
    )
    bow_matrix = vectorizer.fit_transform(
        cleaner(tweets, text_column=text_column)[text_column]
    )
    return vectorizer, bow_matrix


def binary_vectorize(tweets, text_column, cleaner=clean_tweets, tokenizer=None):
    return count_vectorize(tweets, text_column, cleaner, tokenizer, binary=True)


def tfidf_vectorize(tweets, text_column, cleaner=clean_tweets, tokenizer=None):
    """Create a TF-IDF representation of the cleaned tweets in the dataframe."""
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer(method=tokenizer))
    tfidf_matrix = vectorizer.fit_transform(
        cleaner(tweets, text_column=text_column)[text_column]
    )
    return vectorizer, tfidf_matrix


def vectorizer_transform(
    tweets,
    vectorizer,
    text_column,
    cleaner=clean_tweets,
):
    """Transform new tweets using an existing vectorizer."""
    transformed_matrix = vectorizer.transform(
        cleaner(tweets, text_column=text_column)[text_column]
    )
    return transformed_matrix
