import pandas as pd

config = {
    'data_paths': {
        'neutral': './data/processedNeutral.csv',
        'positive': './data/processedPositive.csv',
        'negative': './data/processedNegative.csv'
    },
    'text_cleaning': {
        'remove_mentions': True,
        'remove_urls': True,
        'remove_hashtags': True,
        'remove_punctuation': True,
        'convert_to_lowercase': True
    },
    'tokenization': [
        'word_tokenize',
        'stemming',
        'lemmatization',
    ],
    'vectorization': [
        'bag_of_words',
        'tfidf',
    ],
    'models': [
        'naive_bayes',
        'logistic_regression',
        'random_forest',
        'svm',
    ],
}

def get_merged_dataframe(config):
    negative_df = pd.read_csv(config['data_paths']['negative'], header=None)
    positive_df = pd.read_csv(config['data_paths']['positive'], header=None)
    neutral_df = pd.read_csv(config['data_paths']['neutral'], header=None)

    # transposing the datasets to have tweets in rows
    negative_df = negative_df.transpose().rename(columns={0: 'tweet'})
    positive_df = positive_df.transpose().rename(columns={0: 'tweet'})
    neutral_df = neutral_df.transpose().rename(columns={0: 'tweet'})

    # merge all datasets into a single dataframe with a sentiment label
    neutral_df['sentiment'] = 'neutral'
    positive_df['sentiment'] = 'positive'
    negative_df['sentiment'] = 'negative'
    all_tweets_df = pd.concat([neutral_df, positive_df, negative_df]).reset_index(drop=True)
    # drop any rows with missing values
    all_tweets_df = all_tweets_df.dropna()

    return all_tweets_df