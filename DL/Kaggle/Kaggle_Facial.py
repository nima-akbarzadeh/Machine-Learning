from torch import nn, optim
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet
from datasets import load_classification_data

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_submission(loader, dataset, model_15, model_4):
    model_15.eval()
    model_4.eval()
    id_lookup = pd.read_csv("data/IdLookupTable.csv")
    predictions = []
    image_id = 1

    for image, label in tqdm(loader):
        image = image.to(device)
        preds_15 = torch.clip(model_15(image).squeeze(0), 0.0, 96.0)
        preds_4 = torch.clip(model_4(image).squeeze(0), 0.0, 96.0)
        feature_names = id_lookup.loc[id_lookup["ImageId"] == image_id]["FeatureName"]

        for feature_name in feature_names:
            feature_index = dataset.category_names.index(feature_name)
            if feature_names.shape[0] < 10:
                predictions.append(preds_4[feature_index].item())
            else:
                predictions.append(preds_15[feature_index].item())

        image_id += 1

    df = pd.DataFrame({"RowId": np.arange(1, len(predictions)+1), "Location": predictions})
    df.to_csv("submission.csv", index=False)
    model_15.train()
    model_4.train()


def get_rmse(loader, model, loss_fn):
    model.eval()
    num_examples = 0
    losses = []
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = loss_fn(scores[targets != -1], targets[targets != -1])
        num_examples += scores[targets != -1].shape[0]
        losses.append(loss.item())

    return (sum(losses)/num_examples)**0.5


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_training(params, basic_params, train_loader):

    lr = params['lerning_rate']
    wd = params['weight_decay']
    hd = params['hidden_dim']
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(hd, 30)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model = EfficientNet.from_pretrained("efficientnet-b0")
    model._fc = nn.Linear(hd, 30)
    model = model.to(device)
    
    loss_fn = nn.MSELoss(reduction="sum")

    # Training Loop
    for epoch in range(basic_params['num_epochs']):

        losses = []
        loop = tqdm(train_loader)
        num_examples = 0
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            scores[targets == -1] = -1
            loss = loss_fn(scores, targets)
            num_examples += torch.numel(scores[targets != -1])
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Loss average over epoch: {(sum(losses) / num_examples) ** 0.5}")

        # get on validation
        if basic_params['save_model']:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=basic_params['checkpoint_file'])

    return model


def get_best_params(params_set, basic_params, train_ds, valid_ds):

    params_combinations = []
    scores = []
    for bs in params_set['batch_size']:
        train_loader = DataLoader(
            train_ds,
            batch_size=bs,
            num_workers=basic_params['num_workers'],
            pin_memory=basic_params['pin_memory'],
            shuffle=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=bs,
            num_workers=basic_params['num_workers'],
            pin_memory=basic_params['pin_memory'],
            shuffle=False,
        )
        for hd in params_set['hidden_dim']:
            for lr in params_set['learning_rate']:
                for wd in params_set['weight_decay']:
                    params = {
                        'batch_size': bs,
                        'hidden_dim': hd,
                        'learning_rate': lr,
                        'weight_decay': wd,
                    }
                    params_combinations.append(params)
                    print(params)

                    model = get_training(params, basic_params, train_loader)

                    loss_fn = nn.MSELoss(reduction="sum")
                    rmse = get_rmse(valid_loader, model, loss_fn)
                    print(f"VALIDATION RMSE: {rmse}")

        scores_np = np.array(scores)
        idx = scores_np.argmax()

        return params_combinations[idx], scores_np[idx]


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    params_set = {
        'batch_size': [128],
        'hidden_dim': [128],
        'learning_rate': [1e-4, 1e-3],
        'weight_decay': [1e-5],
    }

    basic_params = {
        'num_epochs': 5,
        'num_workers': 4,
        'checkpoint_file': "b0.pth.tar",
        'pin_memory': True,
        'save_model': True,
        'load_model': True,
    }

    train_dss, valid_dss, test_ds = load_classification_data('facial')
    train_ds_4 = train_dss[0]
    valid_ds_4 = valid_dss[0]
    train_ds_15 = train_dss[1]
    valid_ds_15 = valid_dss[1]

    basic_params['checkpoint_file'] = "b0_4.pth.tar"
    best_params_4, best_score_4 = get_best_params(params_set, basic_params, train_ds_4, valid_ds_4)
    print(f"BEST PARAMS: {best_params_4}")
    print(f"BEST SCORE: {best_score_4}")
    train_loader = DataLoader(
        train_ds_4,
        batch_size=best_params_4['batch_size'],
        num_workers=basic_params['num_workers'],
        pin_memory=basic_params['pin_memory'],
        shuffle=True)
    best_model_4 = get_training(best_params_4, basic_params, train_loader)

    basic_params['checkpoint_file'] = "b0_15.pth.tar"
    best_params_15, best_score_15 = get_best_params(params_set, basic_params, train_ds_15, valid_ds_15)
    print(f"BEST PARAMS: {best_params_15}")
    print(f"BEST SCORE: {best_score_15}")
    train_loader = DataLoader(
        train_ds_15,
        batch_size=best_params_15['batch_size'],
        num_workers=basic_params['num_workers'],
        pin_memory=basic_params['pin_memory'],
        shuffle=True)
    best_model_15 = get_training(best_params_15, basic_params, train_loader)

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=basic_params['num_workers'],
        pin_memory=basic_params['pin_memory'],
        shuffle=False,
    )

    get_submission(test_loader, test_ds, best_model_15, best_model_4)
