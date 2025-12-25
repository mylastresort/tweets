import pandas as pd

def get_merged_dataframe(negative_df_path, positive_df_path, neutral_df_path):
    negative_df = pd.read_csv(negative_df_path, header=None)
    positive_df = pd.read_csv(positive_df_path, header=None)
    neutral_df = pd.read_csv(neutral_df_path, header=None)

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