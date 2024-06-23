import numpy as np


def cosine_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.

    Parameters:
    embedding1 (np.array): The first embedding vector.
    embedding2 (np.array): The second embedding vector.

    Returns:
    float: The cosine similarity between the two embeddings.
    """
    # Compute the dot product between the two embeddings
    dot_product = np.dot(embedding1, embedding2)

    # Compute the norm (magnitude) of each embedding
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    # Compute the cosine similarity
    cosine_sim = dot_product / (norm1 * norm2)

    return cosine_sim