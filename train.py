from sklearn.ensemble import RandomForestClassifier
from vectorizer import count_vectorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from cleaning import clean_tweets


def generate_test_bow_matrix(
    df,
    tweet_column="tweet",
    cleaner=clean_tweets,
    tokenizer=None,
    vectorizer=count_vectorize,
):
    vectorizer, bow_matrix = vectorizer(
        df, tweet_column, cleaner=cleaner, tokenizer=tokenizer
    )

    return bow_matrix


def train_model(
    x_train,
    y_train,
    classifier=RandomForestClassifier(n_estimators=100, random_state=42),
):
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_model(model, x_test, y_test, quiet=True):
    test_predictions = model.predict(x_test)
    if not quiet:
        print("Classification Report:")
        print(classification_report(y_test, test_predictions))
    # Return accuracy
    accuracy = (test_predictions == y_test).mean()
    return accuracy
