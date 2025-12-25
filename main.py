import nltk
from config import get_merged_dataframe

def configure():
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

def main():
    negative_df_path = './data/processedNegative.csv'
    positive_df_path = './data/processedPositive.csv'
    neutral_df_path = './data/processedNeutral.csv'
    df = get_merged_dataframe(
        negative_df_path,
        positive_df_path,
        neutral_df_path,
    )

if __name__ == "__main__":
    print("This is a module for training tweet sentiment analysis models.")
    configure()
