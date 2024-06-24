from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


def calculate_similarity(text1, text2):
    """
    Calculate the similarity between two texts using cosine similarity on their TF-IDF vectors.

    Parameters:
    text1 (str): The first text.
    text2 (str): The second text.

    Returns:
    float: The cosine similarity between the two texts.
    """

    corpus = [text1, text2]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity = similarity_matrix[0, 1]

    return similarity


def cluster_documents(documents, num_clusters=3):
    """
    Cluster a collection of documents into groups based on their content using K-Means clustering on TF-IDF vectors.

    Parameters:
    documents (list of str): The collection of documents.
    num_clusters (int): The number of clusters to create.

    Returns:
    list of int: The cluster labels for each document.
    """
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Transform the documents into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Perform K-Means clustering
    model = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = model.fit_predict(tfidf_matrix)

    return model, vectorizer, cluster_labels


if __name__ == '__main__':

    # Example usage
    text1 = "I love programming in Python. It's very versatile and powerful."
    text2 = "Python is a great language for programming. It's both versatile and powerful."

    similarity = calculate_similarity(text1, text2)
    print(f"Cosine Similarity: {similarity}")

    # Example usage
    documents = [
        "K-Means clustering is used in machine learning.",
        "TF-IDF vectors are used to represent documents.",
        "Machine learning involves training models on data.",
        "Clustering groups similar data points together.",
        "Document clustering is a type of unsupervised learning.",
        "TF-IDF stands for Term Frequency-Inverse Document Frequency.",
        "K-Means is a popular clustering algorithm.",
        "Unsupervised learning does not require labeled data.",
        "Data points within a cluster are similar to each other.",
        "K-Means clustering partitions data into K clusters."
    ]

    num_clusters = 3
    model, vectorizer, cluster_labels = cluster_documents(documents, num_clusters)
    print("Cluster Labels:", cluster_labels)

    new_sentence = "I love machine learning whenever K-Means clustering is used!"
    print(model.predict(vectorizer.transform([new_sentence])))
