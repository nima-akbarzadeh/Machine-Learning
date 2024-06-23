from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


def generate_ngrams(text, n):
    """
    Generate a list of n-grams (tuples of n consecutive words) from a given string.

    Parameters:
    text (str): The input string.
    n (int): The number of words in each n-gram.

    Returns:
    list of tuple: The list of n-grams.
    """

    words = text.split()
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]


def tfidf_with_ngrams(documents, n=2):
    """
    Calculate the TF-IDF representation of documents using n-grams instead of single words.

    Parameters:
    documents (list of str): The collection of documents.
    n (int): The number of words in each n-gram.

    Returns:
    sparse matrix: The TF-IDF vectors for the documents.
    """

    vectorizer = TfidfVectorizer(ngram_range=(n, n))
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer


def ngram_frequency_distribution(text, n):
    """
    Calculate the frequency distribution of n-grams in a given text.

    Parameters:
    text (str): The input text.
    n (int): The number of words in each n-gram.

    Returns:
    Counter: A Counter object representing the frequency distribution of n-grams.
    """

    ngrams = generate_ngrams(text, n)
    return Counter(ngrams)


if __name__ == '__main__':
    # Example usage
    text = "I love programming in Python because it is very versatile and powerful"
    n = 3
    ngrams = generate_ngrams(text, n)
    print("N-Grams:", ngrams)

    # Example usage
    documents = [
        "K-Means clustering is used in machine learning.",
        "TF-IDF vectors are used to represent documents.",
        "Machine learning involves training models on data."
    ]
    n = 2
    tfidf_matrix, vectorizer = tfidf_with_ngrams(documents, n)
    print("TF-IDF Matrix (n-grams):", tfidf_matrix.toarray())

    # Example usage
    text = "I love programming in Python because it is very versatile and powerful. Python programming is fun."
    n = 2
    ngram_freq_dist = ngram_frequency_distribution(text, n)
    print("N-Gram Frequency Distribution:", ngram_freq_dist)