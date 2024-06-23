import numpy as np
import random
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter


def preprocess_text(corpus, window_size, num_negative_samples=0):
    """
    Preprocess the text data to generate training examples for CBOW and Skip-Gram.

    Parameters:
    corpus (list of list of str): The input sentences.
    window_size (int): The size of the context window.

    Returns:
    tuple: A tuple containing the vocabulary, word to index mapping, and training examples.
    """

    # Flatten the list of sentences and create a vocabulary
    words = list(chain(*corpus))
    vocabulary = list(set(words))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    index_to_word = {i: word for word, i in word_to_index.items()}

    if num_negative_samples > 0:
        word_counts = Counter(words)
        total_words = len(words)
        word_freq = {word: count / total_words for word, count in word_counts.items()}
        sampling_dist = [word_freq[index_to_word[i]] ** 0.75 for i in range(len(vocab))]
        sampling_dist = [val / sum(sampling_dist) for val in sampling_dist]

    # Generate training examples
    cbow_examples = []
    skipgram_examples = []

    for sentence in corpus:
        sentence_indices = [word_to_index[word] for word in sentence]
        for i, target_word in enumerate(sentence_indices):
            context = sentence_indices[max(0, i - window_size):i] + sentence_indices[i + 1:i + 1 + window_size]

            if len(context) < window_size * 2:
                context += [word_to_index[random.choice(vocabulary)] for _ in range(window_size * 2 - len(context))]

            # For CBOW (context to target)
            cbow_examples.append((context, target_word))

            # For Skip-Gram (target to context)
            for context_word in context:
                if num_negative_samples > 0:
                    negative_samples = random.choices(range(len(vocab)), sampling_dist, k=num_negative_samples)
                    skipgram_examples.append((target_word, context_word, negative_samples))
                else:
                    skipgram_examples.append((target_word, context_word))

    return vocabulary, word_to_index, index_to_word, cbow_examples, skipgram_examples


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        print('context shape')
        print(context.shape)
        print(context)
        embeds = self.embeddings(context)  # shape: (context_size, embedding_dim)
        print(embeds)
        combined = torch.mean(embeds, dim=0).view(1, -1)  # shape: (1, embedding_dim)
        out = self.linear1(combined)  # shape: (1, vocab_size)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs


def train_CBOW(model, cbow_data, lr=0.01, epochs=100):
    # Define loss function and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for context, target in cbow_data:
            target_vector = torch.tensor([target], dtype=torch.long)
            print(target_vector)

            context_vector = torch.tensor(context, dtype=torch.long)
            log_probs = model(context_vector)

            loss = loss_function(log_probs, target_vector)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(cbow_data)}')


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        return torch.log_softmax(self.linear1(self.embeddings(target)), dim=1)


def train_SkipGram(model, skipgram_data, index_to_word, word_to_index, lr=0.01, epochs=100):
    # Define loss function and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr)

    # Prepare data for training
    def make_target_context_vector(target, context, word_to_index):
        target_vector = torch.tensor([word_to_index[target]], dtype=torch.long)
        context_vector = torch.tensor([word_to_index[context]], dtype=torch.long)
        return target_vector, context_vector

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for target, context in skipgram_data:
            target_vector, context_vector = make_target_context_vector(index_to_word[target], index_to_word[context], word_to_index)

            log_probs = model(target_vector)
            loss = loss_function(log_probs, context_vector)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(skipgram_data)}')


def get_word_embedding(word, model):
    word_index = word_to_index[word]
    word_vector = torch.tensor([word_index], dtype=torch.long)
    return model.embeddings(word_vector).detach().numpy()


def negative_sampling(context_indices, vocab_size, num_samples=5):
    """
    Generate negative samples for context words.

    Parameters:
    context_indices (list of int): The indices of context words.
    vocab_size (int): The size of the vocabulary.
    num_samples (int): The number of negative samples to generate.

    Returns:
    list of int: The indices of negative samples.
    """
    negative_samples = []
    while len(negative_samples) < num_samples:
        sample = random.randint(0, vocab_size - 1)
        if sample not in context_indices:
            negative_samples.append(sample)
    return negative_samples


class SkipGramNegativeSampling(SkipGram):
    def backward(self, target_index, context_indices, negative_indices, y_pred, h, learning_rate):
        # Positive context words
        for context_index in context_indices:
            error = y_pred - np.eye(self.vocab_size)[context_index]
            dW2_pos = np.outer(h, error)
            dW1_pos = np.outer(error, self.W1[target_index])
            self.W1[target_index] -= learning_rate * dW1_pos
            self.W2 -= learning_rate * dW2_pos

        # Negative samples
        for neg_index in negative_indices:
            error = y_pred - np.eye(self.vocab_size)[neg_index]
            dW2_neg = np.outer(h, error)
            dW1_neg = np.outer(error, self.W1[target_index])
            self.W1[target_index] += learning_rate * dW1_neg
            self.W2 += learning_rate * dW2_neg


class SkipGramHierarchicalSoftmax(SkipGram):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__(vocab_size, embedding_dim)
        self.huffman_tree = self.build_huffman_tree(vocab_size)

    def build_huffman_tree(self, vocab_size):
        # Implementation of Huffman tree creation
        pass

    def forward(self, target_index):
        # Modify forward pass to traverse the Huffman tree
        pass

    def backward(self, target_index, context_indices, y_pred, h, learning_rate):
        # Modify backward pass to traverse the Huffman tree
        pass


if __name__ == '__main__':
    sentences = [
        ["I", "love", "programming", "in", "Python"],
        ["Python", "is", "a", "great", "language"],
        ["I", "use", "Python", "for", "data", "science"],
        ["Machine", "learning", "is", "fun", "with", "Python"]
    ]

    window_size = 2
    vocab, word_to_index, index_to_word, cbow_data, skipgram_data = preprocess_text(sentences, window_size)

    # CBOW model training
    CBOW_model = CBOW(len(vocab), 10)
    train_CBOW(CBOW_model, cbow_data, lr=0.01, epochs=100)
    embedding = get_word_embedding('Python', CBOW_model)
    print(f'Embedding for "Python": {embedding}')

    # Skip-Gram model training
    SkipGram_model = SkipGram(len(vocab), 10)
    train_SkipGram(SkipGram_model, skipgram_data, index_to_word, word_to_index, lr=0.01, epochs=100)
    embedding = get_word_embedding('Python', SkipGram_model)
    print(f'Embedding for "Python": {embedding}')

    # Skip-Gram model with negative sampling training
    skipgram_ns_model = SkipGramNegativeSampling(len(vocab), 10)
    train_SkipGram(skipgram_ns_model, skipgram_data, index_to_word, word_to_index, lr=0.01, epochs=100)
    embedding = get_word_embedding('Python', skipgram_ns_model)
    print(f'Embedding for "Python": {embedding}')
