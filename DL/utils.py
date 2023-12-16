import math
import sklearn
import numpy as np
from collections import Counter
import torch
import spacy
from torchtext.data.metrics import bleu_score


def majority_vote(x):
    most_common_data = Counter(x).most_common(1)
    return most_common_data[0][0]


def euclidean_distance(x1, x2):
    if len(x1) != len(x1):
        raise Exception('The vectors must have the same size to calculate accuracy!')
    else:
        return math.dist(x1, x2)


def mse_distance(x1, x2):
    if len(x1) != len(x1):
        raise Exception('The vectors must have the same size to calculate accuracy!')
    else:
        return sklearn.metrics.mean_squared_error(x1, x2)


def accuracy(agent, oracle):
    if len(agent) != len(oracle):
        raise Exception('Agent and Oracle must have the same size to calculate accuracy!')
    else:
        return 100 * np.sum(agent == oracle) / len(oracle)


def sigmoid(x, temperature=1):
    return 1 / (1 + np.exp(-temperature * x))


def unit_step(x):
    return np.where(x >= 0, 1, 0)


def normal_pdf(x, mean, var):
    zero_indices = np.where(var == 0)
    if not zero_indices:
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    else:
        pdf = np.zeros(x.shape)
        for idx in range(len(x)):
            if idx in zero_indices:
                pdf[idx] = 1
            else:
                numerator = np.exp(- (x[idx] - mean[idx]) ** 2 / (2 * var[idx]))
                denominator = np.sqrt(2 * np.pi * var[idx])
                pdf[idx] = numerator / denominator
        return pdf


def entropy(x):
    hist = np.bincount(x)
    ps = hist / len(x)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Load german tokenizer
    spacy_ger = spacy.load("de")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="checkpoints/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])










