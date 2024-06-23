import numpy as np


def get_probabilities(corpus):
    # Extract unique words and tags
    words = [word for word, tag in corpus]
    tags = [tag for word, tag in corpus]
    unique_words = list(set(words))
    unique_tags = list(set(tags))

    # Create mappings from words/tags to indices and vice versa
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    tag_to_index = {tag: i for i, tag in enumerate(unique_tags)}
    index_to_tag = {i: tag for tag, i in tag_to_index.items()}

    # Initialize HMM parameters
    num_tags = len(unique_tags)
    num_words = len(unique_words)

    # Transition probabilities
    transitions = np.zeros((num_tags, num_tags))
    # Emission probabilities
    emissions = np.zeros((num_tags, num_words))
    # Initial state probabilities
    initial_probs = np.zeros(num_tags)

    # Fill the matrices with counts
    for i in range(len(corpus)):
        word, tag = corpus[i]
        word_idx = word_to_index[word]
        tag_idx = tag_to_index[tag]

        emissions[tag_idx, word_idx] += 1

        if i == 0 or corpus[i - 1][1] == ".":
            initial_probs[tag_idx] += 1
        else:
            prev_tag_idx = tag_to_index[corpus[i - 1][1]]
            transitions[prev_tag_idx, tag_idx] += 1

    # Normalize the matrices to get probabilities
    initial_probs /= initial_probs.sum()
    transitions /= transitions.sum(axis=1, keepdims=True)
    emissions /= emissions.sum(axis=1, keepdims=True)

    return num_tags, emissions, initial_probs, word_to_index, transitions, index_to_tag


def viterbi(observation_sequence, num_tags, emissions, initial_probs, word_to_index, transitions, index_to_tag):
    num_observations = len(observation_sequence)
    viterbi_matrix = np.zeros((num_tags, num_observations))
    backpointer_matrix = np.zeros((num_tags, num_observations), dtype=int)

    # Initialization step
    for s in range(num_tags):
        viterbi_matrix[s, 0] = initial_probs[s] * emissions[s, word_to_index[observation_sequence[0]]]

    # Recursion step
    for t in range(1, num_observations):
        for s in range(num_tags):
            max_prob, best_prev_state = max(
                (viterbi_matrix[prev_s, t - 1] * transitions[prev_s, s] * emissions[s, word_to_index[observation_sequence[t]]], prev_s)
                for prev_s in range(num_tags)
            )
            viterbi_matrix[s, t] = max_prob
            backpointer_matrix[s, t] = best_prev_state

    # Termination step
    best_last_state = np.argmax(viterbi_matrix[:, -1])
    best_path = [best_last_state]

    for t in range(num_observations - 1, 0, -1):
        best_last_state = backpointer_matrix[best_last_state, t]
        best_path.insert(0, best_last_state)

    best_path_tags = [index_to_tag[state] for state in best_path]
    return best_path_tags


if __name__ == '__main__':
    # Sample corpus with POS tags
    corpus = [
        ("I", "PRP"), ("love", "VBP"), ("programming", "VBG"),
        ("in", "IN"), ("Python", "NNP"),
        ("Python", "NNP"), ("is", "VBZ"), ("a", "DT"), ("great", "JJ"), ("language", "NN"),
        ("I", "PRP"), ("use", "VBP"), ("Python", "NNP"), ("for", "IN"), ("data", "NN"), ("science", "NN"),
        ("Machine", "NNP"), ("learning", "NN"), ("is", "VBZ"), ("fun", "JJ"), ("with", "IN"), ("Python", "NNP")
    ]
    num_tags, emissions, initial_probs, word_to_index, transitions, index_to_tag = get_probabilities(corpus)

    # Example sentence for POS tagging
    sentence = ["I", "love", "Python"]
    pos_tags = viterbi(sentence, num_tags, emissions, initial_probs, word_to_index, transitions, index_to_tag)
    print(f"Sentence: {sentence}")
    print(f"POS Tags: {pos_tags}")
