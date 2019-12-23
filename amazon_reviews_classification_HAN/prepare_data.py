import numpy as np
import pandas as pd
import os
import pickle

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.preprocessors import HANPreprocessor, TextPreprocessor


def preprocess(df, out_path, mode="standard", text_col="reviewText"):

    train_dir = out_path / "train"
    valid_dir = out_path / "valid"
    test_dir = out_path / "test"
    paths = [train_dir, valid_dir, test_dir]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    texts = df[text_col].tolist()

    if mode == "standard":
        tok = TextPreprocessor()
        tr_fname = "train.npz"
        val_fname = "valid.npz"
        te_fname = "test.npz"
        tok_name = "TextPreprocessor.p"
    elif mode == "han":
        tok = HANPreprocessor()
        tr_fname = "han_train.npz"
        val_fname = "han_valid.npz"
        te_fname = "han_test.npz"
        tok_name = "HANPreprocessor.p"

    padded_texts = tok.tokenize(texts)
    pickle.dump(tok, open(out_path / tok_name, "wb"))

    X_train, X_valid, y_train, y_valid = train_test_split(
        padded_texts, df.overall, train_size=0.8, random_state=1, stratify=df.overall
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid, y_valid, train_size=0.5, random_state=1, stratify=y_valid
    )

    np.savez(train_dir / tr_fname, X_train=X_train, y_train=y_train)
    np.savez(valid_dir / val_fname, X_valid=X_valid, y_valid=y_valid)
    np.savez(test_dir / te_fname, X_test=X_test, y_test=y_test)


if __name__ == "__main__":

    DATA_PATH = Path("../datasets/amazon_reviews/")
    OUT_PATH = Path("data/")
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # DATA_PATH = Path('/home/ubuntu/projects/nlp-stuff/datasets/amazon_reviews')
    df_org = pd.read_json(DATA_PATH / "reviews_Clothing_Shoes_and_Jewelry_5.json.gz", lines=True)

    # classes from [0,num_class)
    df = df_org.copy()
    df["overall"] = (df["overall"] - 1).astype("int64")

    # group reviews with 1 and 2 scores into one class
    df.loc[df.overall == 0, "overall"] = 1

    # and back again to [0,num_class)
    df["overall"] = (df["overall"] - 1).astype("int64")

    # agressive preprocessing: drop short reviews
    df["reviewLength"] = df.reviewText.apply(lambda x: len(x.split(" ")))
    df = df[df.reviewLength >= 5]
    df = df.drop("reviewLength", axis=1).reset_index()
    df.to_csv(OUT_PATH / "reviews_Clothing_Shoes_and_Jewelry.csv", index=False)

    # prepare the arrays
    preprocess(df, OUT_PATH)
    preprocess(df, OUT_PATH, mode="han")
