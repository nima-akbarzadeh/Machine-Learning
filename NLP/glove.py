import numpy as np
import itertools
from collections import defaultdict


def load_glove_embeddings(file_path):
    """
    Load pre-trained GloVe embeddings from a file.

    Parameters:
    file_path (str): The path to the GloVe embeddings file.

    Returns:
    dict: A dictionary mapping words to their GloVe embeddings.
    """
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def build_cooccurrence_matrix(corpus, vocab, window_size=4):
    """
    Build a co-occurrence matrix from the corpus.

    Parameters:
    corpus (list of list of str): The input corpus as a list of sentences.
    vocab (list of str): The vocabulary list.
    window_size (int): The size of the context window.

    Returns:
    dict: A dictionary representing the co-occurrence matrix.
    """

    word_to_id = {word: i for i, word in enumerate(vocab)}
    cooccurrence_matrix = defaultdict(lambda: defaultdict(float))

    for sentence in corpus:
        for i, word in enumerate(sentence):
            if word in word_to_id:
                word_id = word_to_id[word]
                context = sentence[max(0, i - window_size):i] + sentence[i+1:i+window_size+1]
                for context_word in context:
                    if context_word in word_to_id:
                        context_word_id = word_to_id[context_word]
                        cooccurrence_matrix[word_id][context_word_id] += 1.0

    return cooccurrence_matrix


def glove_loss_grads(word_vector, context_vector, bias_word, bias_context, cooccurrence, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Perform a single GloVe step.

    Parameters:
    word_vector (np.array): The word vector.
    context_vector (np.array): The context vector.
    bias_word (float): The word bias.
    bias_context (float): The context bias.
    cooccurrence (float): The co-occurrence value.
    learning_rate (float): The learning rate.
    x_max (int): The maximum co-occurrence value.
    alpha (float): The weighting parameter.

    Returns:
    float: The loss value.
    """
    weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1.0
    diff = np.dot(word_vector, context_vector) + bias_word + bias_context - np.log(cooccurrence)
    loss = weight * (diff ** 2)

    grad_main = weight * diff * context_vector
    grad_context = weight * diff * word_vector
    grad_bias_main = weight * diff
    grad_bias_context = weight * diff

    word_vector -= learning_rate * grad_main
    context_vector -= learning_rate * grad_context
    bias_word -= learning_rate * grad_bias_main
    bias_context -= learning_rate * grad_bias_context

    return loss, word_vector, context_vector, bias_word, bias_context


def train_glove(corpus, vocab, embedding_dim=50, epochs=100, learning_rate=0.05):
    """
    Train GloVe embeddings from the corpus.

    Parameters:
    corpus (list of list of str): The input corpus as a list of sentences.
    vocab (list of str): The vocabulary list.
    embedding_dim (int): The dimension of the embeddings.
    epochs (int): The number of training epochs.
    learning_rate (float): The learning rate.

    Returns:
    np.array: The trained word embeddings.
    """
    vocab_size = len(vocab)
    cooccurrence_matrix = build_cooccurrence_matrix(corpus, vocab)

    word_vectors = np.random.rand(vocab_size, embedding_dim)
    context_vectors = np.random.rand(vocab_size, embedding_dim)
    biases_word = np.zeros(vocab_size)
    biases_context = np.zeros(vocab_size)

    for epoch in range(epochs):
        total_loss = 0
        for word_id, context_ids in cooccurrence_matrix.items():
            for context_id, cooccurrence in context_ids.items():
                loss, word_vector, context_vector, bias_word, bias_context = glove_loss_grads(
                    word_vectors[word_id], context_vectors[context_id], biases_word[word_id], biases_context[context_id], cooccurrence, learning_rate
                )
                word_vectors[word_id] = word_vector
                context_vectors[context_id] = context_vector
                biases_word[word_id] = bias_word
                biases_context[context_id] = bias_context

                total_loss += loss
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss}')

    return word_vectors


if __name__ == '__main__':

    # # Example usage
    # glove_file_path = 'path/to/glove.6B.50d.txt'
    # glove_embeddings = load_glove_embeddings(glove_file_path)
    #
    # words = ["hello", "world", "goodbye"]
    # embeddings = [glove_embeddings[word] for word in words if word in glove_embeddings]
    # print(embeddings)

    # Example usage
    corpus = [
        ["i", "love", "this", "movie"],
        ["this", "movie", "is", "great"],
        ["i", "hate", "this", "film"],
        ["this", "film", "is", "awful"]
    ]
    vocab = ["i", "love", "this", "movie", "is", "great", "hate", "film", "awful"]

    word_to_id = {word: i for i, word in enumerate(vocab)}
    embeddings = train_glove(corpus, vocab)

    words = ["i", "love", "this", "movie"]
    embeddings = [embeddings[word_to_id[word]] for word in words if word in word_to_id]
    print(embeddings)
