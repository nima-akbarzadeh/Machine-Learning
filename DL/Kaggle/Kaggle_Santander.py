import torch
from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
from datasets import load_classification_data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


class FC_NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(FC_NeuralNet, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class FCN_COMBO(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(FCN_COMBO, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(input_size//2*hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        x = self.bn(x)
        orig_features = x[:, :200].unsqueeze(2)  # (N, 200, 1)
        new_features = x[:, 200:].unsqueeze(2)  # (N, 200, 1)
        x = torch.cat([orig_features, new_features], dim=2)  # (N, 200, 2)
        x = self.relu(self.fc1(x)).reshape(N, -1)  # (N, 200*hidden_dim)
        return torch.sigmoid(self.fc2(x)).view(-1)


def get_training(params, train_loader, flag='original'):

    if flag == 'original':
        # Model
        hidden_dim = params['hidden_dim']
        model = FCN_COMBO(input_size=400, hidden_dim=hidden_dim).to(device)

        # Optimizer
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Loss
        loss_fn = nn.BCELoss()

        # Training Loop
        for _ in tqdm(range(params['num_epochs'])):

            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)

                # compute the loss
                scores = model(data)
                loss = loss_fn(scores.squeeze(1), targets)

                # backward prop.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    else:

        # Model
        hidden_dim = params['hidden_dim']
        model = FCN_COMBO(input_size=400, hidden_dim=hidden_dim).to(device)

        # Optimizer
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Loss
        loss_fn = nn.BCELoss()

        # Training Loop
        for _ in tqdm(range(params['num_epochs'])):

            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)

                # compute the loss
                scores = model(data)
                loss = loss_fn(scores, targets)

                # backward prop.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return model


def get_best_params(params_set, num_epochs, train_ds, valid_ds):

    params_combinations = []
    scores = []
    for bs in params_set['batch_size']:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=bs)
        for hd in params_set['hidden_dim']:
            for lr in params_set['learning_rate']:
                for wd in params_set['weight_decay']:
                    params = {
                        'batch_size': bs,
                        'hidden_dim': hd,
                        'learning_rate': lr,
                        'weight_decay': wd,
                        'num_epochs': num_epochs,
                    }
                    params_combinations.append(params)
                    print(params)

                    model = get_training(params, train_loader, flag='agumented')

                    probabilities, true = get_predictions(valid_loader, model)
                    score = metrics.roc_auc_score(true, probabilities)
                    scores.append(score)
                    print(f"VALIDATION ROC: {score}")

    scores_np = np.array(scores)
    idx = scores_np.argmax()

    return params_combinations[idx], scores_np[idx]


def get_predictions(loader, model):
    model.eval()
    saved_preds = []
    true_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            saved_preds += scores.tolist()
            true_labels += y.tolist()

    model.train()
    return saved_preds, true_labels


def get_submission(model, loader, test_ids):
    print(len(test_ids.values))
    all_preds = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            score = model(x)
            prediction = score.float()
            all_preds += prediction.tolist()
    output = np.array(all_preds)
    output = output.squeeze(1)

    df = pd.DataFrame({
        "ID_code": test_ids.values,
        "target": output,
    })

    df.to_csv("./data/santander_submission.csv", index=False)



if __name__ == "__main__":

    num_epochs = 5

    params_set = {
        'batch_size': [128],
        'hidden_dim': [128],
        'learning_rate': [1e-3, 2e-3],
        'weight_decay': [1e-5],
    }

    # Dataset preprocessing
    train_ds, valid_ds, test_ds, test_ids = load_classification_data('santander')

    best_params, best_score = get_best_params(params_set, num_epochs, train_ds, valid_ds)

    print(f"BEST PARAMS: {best_params}")
    print(f"BEST SCORE: {best_score}")

    train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True)
    best_model = get_training(best_params, train_loader, flag='agumented')
    test_loader = DataLoader(test_ds, batch_size=1)
    get_submission(best_model, test_loader, test_ids)


