import numpy as np
import pandas as pd
import os
import pickle

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.tokenizers import HANTokenizer, BertFamilyTokenizer

RAW_DATA = Path("../datasets/amazon_reviews/")
PROCESSED_DATA = Path("processed_data/")
if not os.path.exists(PROCESSED_DATA):
    os.makedirs(PROCESSED_DATA)


def preprocess_bert(
    df,
    out_path,
    max_length=120,
    text_col="reviewText",
    pretrained_tokenizer="bert-base-uncased",
):

    train_dir = out_path / "train"
    valid_dir = out_path / "valid"
    test_dir = out_path / "test"
    paths = [train_dir, valid_dir, test_dir]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
    tr_fname = "_".join([pretrained_tokenizer, "train.npz"])
    val_fname = "_".join([pretrained_tokenizer, "valid.npz"])
    te_fname = "_".join([pretrained_tokenizer, "test.npz"])

    texts = df[text_col].tolist()

    tok = BertFamilyTokenizer(
        pretrained_tokenizer=pretrained_tokenizer,
        do_lower_case=True,
        max_length=max_length,
    )

    bert_texts, bert_masks = tok.fit_transform(texts)

    X_train, X_valid, mask_train, mask_valid, y_train, y_valid = train_test_split(
        bert_texts,
        bert_masks,
        df.overall,
        train_size=0.8,
        random_state=1,
        stratify=df.overall,
    )

    X_valid, X_test, mask_valid, mask_test, y_valid, y_test = train_test_split(
        X_valid, mask_valid, y_valid, train_size=0.5, random_state=1, stratify=y_valid
    )

    np.savez(
        train_dir / tr_fname, X_train=X_train, mask_train=mask_train, y_train=y_train
    )
    np.savez(
        valid_dir / val_fname, X_valid=X_valid, mask_valid=mask_valid, y_valid=y_valid
    )
    np.savez(test_dir / te_fname, X_test=X_test, mask_test=mask_test, y_test=y_test)


def preprocess_han(df, out_path, text_col="reviewText"):

    train_dir = out_path / "train"
    valid_dir = out_path / "valid"
    test_dir = out_path / "test"
    paths = [train_dir, valid_dir, test_dir]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)
    tr_fname = "han_train.npz"
    val_fname = "han_valid.npz"
    te_fname = "han_test.npz"
    tok_name = "HANTokenizer.p"

    texts = df[text_col].tolist()

    tok = HANTokenizer()

    han_texts = tok.fit_transform(texts)
    with open(out_path / tok_name, "wb") as f:
        pickle.dump(tok, f)

    X_train, X_valid, y_train, y_valid = train_test_split(
        han_texts, df.overall, train_size=0.8, random_state=1, stratify=df.overall
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid, y_valid, train_size=0.5, random_state=1, stratify=y_valid
    )

    np.savez(train_dir / tr_fname, X_train=X_train, y_train=y_train)
    np.savez(valid_dir / val_fname, X_valid=X_valid, y_valid=y_valid)
    np.savez(test_dir / te_fname, X_test=X_test, y_test=y_test)


def write_or_read_csv():

    inp_fname = RAW_DATA / "reviews_Clothing_Shoes_and_Jewelry_5.json.gz"
    out_fname = PROCESSED_DATA / "reviews_Clothing_Shoes_and_Jewelry.csv"

    if out_fname.exists():
        return pd.read_csv(out_fname)
    else:
        df_org = pd.read_json(inp_fname, lines=True)

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
        df.to_csv(out_fname, index=False)

        return df


if __name__ == "__main__":

    df = write_or_read_csv()

    # prepare the arrays
    preprocess_han(df, PROCESSED_DATA)
    preprocess_bert(df, PROCESSED_DATA)
    preprocess_bert(df, PROCESSED_DATA, pretrained_tokenizer="distilbert-base-uncased")
