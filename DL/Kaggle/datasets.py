import torch
import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def load_classification_data(data_name, device=torch.device('cpu'), data_info=True):
    if data_name == 'santander':

        # Check correlation between features and remove one of the two highly correlated ones
        def check_correlation(x):
            corr_coef = x.corr().abs()
            max_coeff = corr_coef.mean()
            flag = False
            for mc in max_coeff:
                if mc > 0.5:
                    flag = True
            if flag:
                print("There are features which are highly correlated!")
            else:
                print("The features seem to be uncorrelated!")

            pass


        def data_norm(x_train, x_test, type_norm='standard'):
            scaler = None
            if type_norm == 'standard':
                print('Standard Normalization is done!')
                scaler = StandardScaler()
            elif type_norm == 'minmax':
                print('Minmax Normalization is done!')
                scaler = MinMaxScaler()
            scaler.fit(x_train)

            return pd.DataFrame(scaler.transform(x_train), columns=x_train.columns), pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

        # Check if the maximum and minimum variance and see if any unneccessary features can be removed
        def check_pca(x_train, x_test, drop_coeff=0.1):
            new_x = x_train - np.mean(x_train, axis=0)
            new_xtest = x_test - np.mean(x_test, axis=0)
            # covariance, function needs samples as columns
            cov = np.cov(new_x.T)
            # eigenvalues, eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            print(np.max(eigenvalues))
            print(np.min(eigenvalues))
            # Drop features if the eigenvalues are less than drop_coeff=0.1 of np.max(eigenvalues)
            if np.min(eigenvalues) < drop_coeff * np.max(eigenvalues):
                list_e = []
                for e in range(len(eigenvalues)):
                    if eigenvalues[e] < drop_coeff * np.max(eigenvalues):
                        list_e.append(e)
                np.delete(eigenvalues, list_e, axis=0)
                np.delete(eigenvectors, list_e, axis=0)
                eigenvectors = eigenvectors.T
                idxs = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[idxs]

                return np.dot(new_x, eigenvectors.T), np.dot(new_xtest, eigenvectors.T)

            else:
                print("No need for dimensionality reduction!")

                return x_train, x_test

        # Check class imbalance
        def data_imbalance(labels):
            unique, counts = np.unique(labels, return_counts=True)
            if np.max(counts) > 2 * np.min(counts):
                print('DATA IMBALANCE!')
            count_dict = dict(zip(unique, counts))
            print(f"labels count is {count_dict}")

            pass

        # Check if a new feature based on uniquness of the values for each feature can be added
        def add_features(train_features, test_features):
            train_and_test = pd.concat([train_features, test_features], axis=0)
            col_names = [f"var_{i}" for i in range(200)]
            for col in tqdm(col_names):
                count = train_and_test[col].value_counts().to_dict()
                train_and_test[col + "_unique"] = train_and_test[col].apply(lambda x: 1 if count[x] == 1 else 0).values
            new_train_data = train_and_test[train_and_test["ID_code"].str.contains("train")].copy()
            new_test_data = train_and_test[train_and_test["ID_code"].str.contains("test")].copy()

            return new_train_data, new_test_data.drop(["target"], axis=1)

        def get_preprocessing(load_data=True):
            if load_data:
                train_data = pd.read_csv("new_santander_train.csv")
                new_train_features = train_data.drop(["ID_code", "target"], axis=1)
                train_labels = train_data["target"]
                test_data = pd.read_csv("new_santander_test.csv")
                new_test_features = test_data.drop(["ID_code"], axis=1)
                test_ids = test_data["ID_code"]

            else:
                train_data = pd.read_csv("santander_train.csv")
                train_features = train_data.drop(["ID_code", "target"], axis=1)
                train_labels = train_data["target"]
                test_data = pd.read_csv("santander_test.csv")
                test_features = test_data.drop(["ID_code"], axis=1)
                test_ids = test_data["ID_code"]
                check_correlation(train_features)
                train_features, test_features = data_norm(train_features, test_features, type_norm='standard')
                train_features, test_features = check_pca(train_features, test_features)
                data_imbalance(train_labels)
                norm_train_data = pd.concat([train_data[["ID_code", "target"]], train_features], axis=1)
                norm_test_data = pd.concat([test_data["ID_code"], test_features], axis=1)
                new_train_data, new_test_data = add_features(norm_train_data, norm_test_data)
                new_train_data.to_csv("new_santander_train.csv", index=False)
                new_test_data.to_csv("new_santander_test.csv", index=False)
                new_train_features = new_train_data.drop(["ID_code", "target"], axis=1)
                new_test_features = new_test_data.drop(["ID_code"], axis=1)
                new_train_features, new_test_features = check_pca(new_train_features, new_test_features)
                data_imbalance(train_labels)

            return new_train_features, new_test_features, train_labels, test_ids

        # new_train_features, new_test_features, train_labels, test_ids = get_preprocessing(load_data=False)
        new_train_features, new_test_features, train_labels, test_ids = get_preprocessing(load_data=True)

        X_tensor = torch.tensor(new_train_features.values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(train_labels.values, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_ds, valid_ds = random_split(dataset, [int(0.8 * len(dataset)), ceil(0.2 * len(dataset))])

        X_tensor = torch.tensor(new_test_features.values, dtype=torch.float32).to(device)
        y_tensor = torch.zeros(len(test_ids), dtype=torch.float32)
        test_ds = TensorDataset(X_tensor, y_tensor)

        return train_ds, valid_ds, test_ds, test_ids

    elif data_name == 'facial':

        # Data augmentation for images
        train_transforms = A.Compose(
            [
                A.Resize(width=96, height=96),
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.8),
                A.IAAAffine(shear=15, scale=1.0, mode="constant", p=0.2),
                A.RandomBrightnessContrast(contrast_limit=0.5, brightness_limit=0.5, p=0.2),
                A.OneOf([
                    A.GaussNoise(p=0.8),
                    A.CLAHE(p=0.8),
                    A.ImageCompression(p=0.8),
                    A.RandomGamma(p=0.8),
                    A.Posterize(p=0.8),
                    A.Blur(p=0.8),
                ], p=1.0),
                A.OneOf([
                    A.GaussNoise(p=0.8),
                    A.CLAHE(p=0.8),
                    A.ImageCompression(p=0.8),
                    A.RandomGamma(p=0.8),
                    A.Posterize(p=0.8),
                    A.Blur(p=0.8),
                ], p=1.0),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(
                    mean=[0.4897, 0.4897, 0.4897],
                    std=[0.2330, 0.2330, 0.2330],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

        valid_transforms = A.Compose(
            [
                A.Resize(height=96, width=96),
                A.Normalize(
                    mean=[0.4897, 0.4897, 0.4897],
                    std=[0.2330, 0.2330, 0.2330],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

        train_ds_4 = FacialKeypointDataset(
            csv_file="data/train_4.csv",
            transform=train_transforms,
        )

        valid_ds_4 = FacialKeypointDataset(
            transform=valid_transforms,
            csv_file="data/valid_4.csv",
        )

        train_ds_15 = FacialKeypointDataset(
            csv_file="data/train_15.csv",
            transform=train_transforms,
        )

        valid_ds_15 = FacialKeypointDataset(
            transform=valid_transforms,
            csv_file="data/valid_15.csv",
        )

        test_ds = FacialKeypointDataset(
            csv_file="data/test.csv",
            transform=valid_transforms,
            train=False,
        )

        return [train_ds_4, train_ds_15], [valid_ds_4, valid_ds_15], test_ds


class FacialKeypointDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.category_names = [
            'left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
            'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
            'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
            'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x',
            'mouth_left_corner_y', 'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y'
        ]
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.train:
            image = np.array(self.data.iloc[index, 30].split()).astype(np.float32)
            labels = np.array(self.data.iloc[index, :30].tolist())
            labels[np.isnan(labels)] = -1
        else:
            image = np.array(self.data.iloc[index, 1].split()).astype(np.float32)
            labels = np.zeros(30)

        ignore_indices = labels == -1
        labels = labels.reshape(15, 2)

        if self.transform:
            image = np.repeat(image.reshape(96, 96, 1), 3, 2).astype(np.uint8)
            augmentations = self.transform(image=image, keypoints=labels)
            image = augmentations["image"]
            labels = augmentations["keypoints"]

        labels = np.array(labels).reshape(-1)
        labels[ignore_indices] = -1

        return image, labels.astype(np.float32)
