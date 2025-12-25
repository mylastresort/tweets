from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def tokenize_tweet(tweet):
    """Tokenize a tweet into words."""
    return word_tokenize(tweet)

stemmer = PorterStemmer()

def stem_tokens(tokens):
    """Stem a list of tokens."""
    return [stemmer.stem(token) for token in tokens]


lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    """Lemmatize a list of tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]