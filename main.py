import nltk
from config import get_merged_dataframe, config

def configure():
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

def main():
    df = get_merged_dataframe(config)

if __name__ == "__main__":
    print("This is a module for training tweet sentiment analysis models.")
    configure()
