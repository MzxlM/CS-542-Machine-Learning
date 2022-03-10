import os
import json
import argparse
import importlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import metrics


def import_model(bu_id: str):
    model = importlib.import_module(f"model_{bu_id}")
    return model.Model


def load_and_preprocess_data(model, split: str = "train") -> Tuple[np.array, np.array]:
    X = pd.read_csv(f"../Data/data_cleaned_{split}_comments_X.csv").to_numpy()
    y = pd.read_csv(f"../Data/data_cleaned_{split}_y.csv").to_numpy()
    print(f"Shape of {split} features :", X.shape)
    print(f"Shape of {split} labels :", y.shape)
    model.preprocess(X, y)

    return X, y


def evaluate(model, X: np.array, y: np.array) -> Dict[str, float]:
    pred = model.predict(X)
    return {
        "median_abs_err": metrics.median_absolute_error(y, pred),
        "mean_abs_err": metrics.mean_absolute_error(y, pred),
        "mse": metrics.mean_squared_error(y, pred),
        "R2": metrics.r2_score(y, pred),
        **model.ID_DICT
    }


def main(args):
    np.random.seed(args.seed)
    print("-------------initializing model------------")
    ModelClass = import_model(args.bu_id)
    model = ModelClass()

    print("-------------loading data------------")
    X_train, y_train = load_and_preprocess_data(model, split="train")
    X_test, y_test = load_and_preprocess_data(model, split="val")

    print("-------------training model------------")
    model.train(X_train, y_train)

    print("-------------evaluating model------------")
    score = evaluate(model, X_test, y_test)
    print("model score:", score)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/{args.bu_id}.json", "w") as f:
        json.dump(score, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model for housing data")

    # main parameters
    parser.add_argument("--train_data", type=str, default="../Data", help="Train dataset folder path")
    parser.add_argument("--test_data", type=str, default="../Data", help="Test dataset folder path")
    parser.add_argument("--output_dir", type=str, default="../results", help="Results output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bu_id", type=str, default="0000", help="Last four digit of student's BU ID")
    args = parser.parse_args()

    main(args)

