from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from symspellpy import SymSpell, Verbosity
import pkg_resources

# Initialize once (outside your functions)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def tokenize_tweet(tweet):
    """Tokenize a tweet into words."""
    return word_tokenize(tweet)

stemmer = PorterStemmer()

def stem_tokens(tokens):
    """Stem a list of tokens."""
    return [stemmer.stem(token) for token in tokens]

snowball_stemmer = SnowballStemmer("english")

def snowball_stem_tokens(tokens):
    """Stem a list of tokens using SnowballStemmer."""
    return [snowball_stemmer.stem(token) for token in tokens]

LancasterStemmer = LancasterStemmer()

def lancaster_stem_tokens(tokens):
    """Stem a list of tokens using LancasterStemmer."""
    return [LancasterStemmer.stem(token) for token in tokens]

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    """Lemmatize a list of tokens."""
    return [lemmatizer.lemmatize(token) for token in tokens]

def misspell_tokens(tokens):
    """Placeholder function for handling misspellings in tokens."""
    corrected_tokens = []
    for token in tokens:
        suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_tokens.append(suggestions[0].term)
        else:
            corrected_tokens.append(token)
    return corrected_tokens

def misspell_and_lemmatize_tokens(tokens):
    """Handle misspellings in tokens."""
    tokens = misspell_tokens(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens