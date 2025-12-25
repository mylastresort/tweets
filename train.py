from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from vectorizer import count_vectorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from cleaning import clean_tweets

def train_model(
        df,
        tweet_column = 'tweet',
        sentiment_column = 'sentiment',
        cleaner = clean_tweets,
        tokenizer = None,
        vectorizer = count_vectorize,
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)):
    vectorizer, bow_matrix = vectorizer(df, tweet_column, cleaner=cleaner, tokenizer=tokenizer)
    x_train, x_test, y_train, y_test = train_test_split(
        bow_matrix,
        df[sentiment_column],
        test_size=0.2,
        random_state=42
    )

    classifier.fit(x_train, y_train)
    return classifier, vectorizer, x_test, y_test

def evaluate_model(model, x_test, y_test, quiet=True):
    test_predictions = model.predict(x_test)
    if not quiet:
        print("Classification Report:")
        print(classification_report(y_test, test_predictions))
    # Return accuracy
    accuracy = (test_predictions == y_test).mean()
    return accuracy
