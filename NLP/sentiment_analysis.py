from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_sklearn_sentiment_finder(X_train, y_train):

    # Create a pipeline with TfidfVectorizer and LogisticRegression
    model = Pipeline([
        ('embedding', TfidfVectorizer()),
        ('model', LogisticRegression())
    ])

    # Train the model
    model.fit(X_train, y_train)

    return model


def determine_sentiment_sklearn(model, text):
    """
    Determine the sentiment of a given text using a model trained with Scikit-Learn.

    Parameters:
    text (str): The input text.

    Returns:
    str: The sentiment of the text (positive, negative, neutral).
    """
    # Predict the sentiment of the input text
    sentiment = model.predict([text])[0]
    return sentiment


def determine_sentiment_textblob(text):
    """
    Determine the sentiment of a given text using TextBlob.

    Parameters:
    text (str): The input text.

    Returns:
    str: The sentiment of the text (positive, negative, neutral).
    """
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get the polarity of the text
    polarity = blob.sentiment.polarity

    # Determine the sentiment based on polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"


if __name__ == '__main__':

    # Sample data
    documents = [
        "I love this product!", "This is the best thing ever!", "Absolutely fantastic!",
        "I hate this.", "This is terrible.", "Absolutely horrible experience.",
        "It's okay, not great.", "I feel indifferent about this.", "It's alright."
    ]
    labels = ["positive", "positive", "positive", "negative", "negative", "negative", "neutral", "neutral", "neutral"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

    model = train_sklearn_sentiment_finder(X_train, y_train)

    for index, features in enumerate(X_test):

        prediction = determine_sentiment_sklearn(model, features)
        print(f"Prediction is {prediction}")
        print(f"True sentiment is {y_test[index]}")
