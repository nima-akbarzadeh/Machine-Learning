import math
from n_grams import *
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_sklearn(documents):
    """
    Convert a list of documents into their TF-IDF representation using Scikit-Learn.

    Parameters:
    documents (list of str): The list of input documents.

    Returns:
    sparse matrix: The TF-IDF representation of the documents.
    """
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix


def compute_tf(document, n_grams=1):
    """
    Compute the term frequency for a single document.

    Parameters:
    document (str): The input document.

    Returns:
    dict: A dictionary of term frequencies.
    """

    if n_grams == 0:
        print("n_grams must be positive!")

    elif n_grams == 1:
        words = document.split()
        word_count = Counter(words)

        return {word: counts / len(words) for word, counts in word_count.items()}

    else:
        grams = generate_ngrams(document, n_grams)
        grams_count = ngram_frequency_distribution(document, n_grams)

        return {gram: counts / len(grams) for gram, counts in grams_count.items()}


def compute_idf(documents, n_grams=1):
    """
    Compute the inverse document frequency for a list of documents.

    Parameters:
    documents (list of str): The list of input documents.

    Returns:
    dict: A dictionary of inverse document frequencies.
    """

    if n_grams == 0:
        print("n_grams must be positive!")

    elif n_grams == 1:
        n_documents = len(documents)
        all_words = set(word for document in documents for word in document.split())

        idf = {}
        for word in all_words:
            n_documents_containing_word = sum(1 for document in documents if word in document.split())
            idf[word] = 1 + math.log(n_documents / (1 + n_documents_containing_word))

        return idf

    else:
        n_documents = len(documents)
        all_grams = set(gram for document in documents for gram in generate_ngrams(document, n_grams))

        idf = {}
        for gram in all_grams:
            n_documents_containing_word = sum(1 for document in documents if gram in generate_ngrams(document, n_grams))
            idf[gram] = 1 + math.log(n_documents / (1 + n_documents_containing_word))

        return idf


def tfidf_fromscratch(documents, n_grams=1):
    """
    Convert a list of documents into their TF-IDF representation from scratch.

    Parameters:
    documents (list of str): The list of input documents.

    Returns:
    list of dict: The list of dictionaries containing TF-IDF values for each document.
    """

    def compute_tfidf(document, idf):
        """
        Compute the TF-IDF for a single document.

        Parameters:
        document (str): The input document.
        idf (dict): A dictionary of inverse document frequencies.

        Returns:
        dict: A dictionary of TF-IDF values.
        """

        tf = compute_tf(document, n_grams)

        if n_grams == 0:
            print("n_grams must be positive!")

        elif n_grams == 1:
            tfidf = {word: tf[word] * idf[word] for word in tf}
            return tfidf

        else:
            tfidf = {gram: tf[gram] * idf[gram] for gram in tf}
            return tfidf

    idf = compute_idf(documents, n_grams)

    return [compute_tfidf(document, idf) for document in documents]


if __name__ == '__main__':

    # Example usage
    documents = ["the cat sat on the mat", "the dog sat on the log"]
    tfidf_documents = tfidf_fromscratch(documents, n_grams=2)
    print(tfidf_documents)
