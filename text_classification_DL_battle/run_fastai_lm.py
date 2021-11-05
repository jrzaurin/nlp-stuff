import os
import pickle
import numpy as np
from pathlib import Path
import torch
from torch import nn

import pandas as pd
from fastai.callback.all import EarlyStoppingCallback
from fastai.metrics import accuracy, Perplexity
from fastai.text.all import text_classifier_learner, language_model_learner
from fastai.text.data import TextDataLoaders
from fastai.text.models import AWD_LSTM
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split

RAW_DATA = Path("../datasets/amazon_reviews/")
PROCESSED_DATA = Path("processed_data/")
RESULTS_DIR = Path("results")
if not os.path.exists(PROCESSED_DATA):
    os.makedirs(PROCESSED_DATA)


def prepare_original_data():

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


def df_split(df):

    train_fname = PROCESSED_DATA / "train" / "fastai_train_and_eval.csv"
    test_fname = PROCESSED_DATA / "test" / "fastai_test.csv"

    if train_fname.exists():
        # one cannot exist without the other
        train_and_eval_df = pd.read_csv(train_fname)
        df_test = pd.read_csv(test_fname)
    else:
        df_train, df_valid = train_test_split(
            df, train_size=0.8, random_state=1, stratify=df.overall
        )
        df_valid, df_test = train_test_split(
            df_valid, train_size=0.5, random_state=1, stratify=df_valid.overall
        )

        df_train["is_valid"] = False
        df_valid["is_valid"] = True
        train_and_eval_df = pd.concat([df_train, df_valid], ignore_index=True)

        train_and_eval_df.to_csv(train_fname)
        df_test.to_csv(test_fname)

    return train_and_eval_df, df_test


def build_loader_for_lm(train_df, force=False):

    dl_fname = PROCESSED_DATA / "fastai_dl_lm.p"

    if dl_fname.exists() and not force:
        with open(dl_fname, "rb") as f:
            return pickle.open(f)
    else:
        dl_lm = TextDataLoaders.from_df(
            train_df,
            text_col="reviewText",
            is_lm=True,
            valid_pct=0.1,
            max_vocab=35000,
            min_freq=3,
        )
        with open(dl_fname, "wb") as f:
            pickle.dump(dl_lm, f)
        return dl_lm


def build_lm(dl_lm):

    learner = language_model_learner(
        dl_lm,
        AWD_LSTM,
        metrics=[accuracy, Perplexity()],
        path=RESULTS_DIR,
        wd=0.1,
        cbs=EarlyStoppingCallback(patience=2),
    ).to_fp16()

    learner.fit_one_cycle(1, 5e-3)

    learner.unfreeze()
    learner.fit_one_cycle(8, 1e-3)

    learner.save_encoder("fastai_finetuned_encoder")

    return dl_lm


def build_loader_for_classification(train_df, dl_lm, force=False):

    dl_cls_fname = PROCESSED_DATA / "fastai_dl_lm_cls.p"

    if dl_cls_fname.exists() and not force:
        with open(dl_cls_fname, "rb") as f:
            return pickle.open(f)
    else:
        dl_cls = TextDataLoaders.from_df(
            train_df,
            text_col="reviewText",
            label_col="overall",
            valid_col="is_valid",
            text_vocab=dl_lm.vocab,
            seq_len=120,
            bs=128,
        )
        with open(dl_cls_fname, "wb") as f:
            pickle.dump(dl_cls, f)

        return dl_cls


def train_learner(dl_cls):

    learner = text_classifier_learner(
        dl_cls,
        AWD_LSTM,
        drop_mult=0.5,
        path=RESULTS_DIR,
        metrics=accuracy,
        cbs=EarlyStoppingCallback(patience=2),
    )
    learner.load_encoder("fastai_finetuned_encoder")

    learner.fit_one_cycle(1, 3e-3)

    learner.freeze_to(-2)
    learner.fit_one_cycle(1, slice(3e-3 / (2.6 ** 4), 3e-3))

    learner.freeze_to(-3)
    learner.fit_one_cycle(1, slice(1e-3 / (2.6 ** 4), 1e-3))

    learner.unfreeze()
    learner.fit_one_cycle(2, slice(2e-4 / (2.6 ** 4), 2e-4))

    learner.save("fastai_learner_lm_cls")

    return learner


def get_results(y_true, y_pred):
    results_d = {}
    results_d["acc"] = accuracy_score(y_true, y_pred)
    results_d["f1"] = f1_score(y_true, y_pred, average="weighted")
    results_d["prec"] = precision_score(y_true, y_pred, average="weighted")
    return results_d


if __name__ == "__main__":

    df = prepare_original_data()

    train_and_eval_df, df_test = df_split(df)
    train_df = train_and_eval_df[~train_and_eval_df.is_valid]

    dl_lm = build_loader_for_lm(train_df, force=True)

    dl_lm = build_lm(dl_lm)

    dl_cls = build_loader_for_classification(train_and_eval_df, dl_lm, force=True)

    learner = train_learner(dl_cls)

    dl_test = learner.dls.test_dl(df_test.reviewText)
    preds, _ = learner.get_preds(dl=dl_test)

    loss_fn = nn.CrossEntropyLoss()
    target = torch.from_numpy(df_test.overall.values)
    loss = loss_fn(preds, target).item()

    pred_label = np.asarray(preds).argmax(1)
    results_d = get_results(df_test.overall, pred_label)
    results_d["loss"] = loss

    with open(RESULTS_DIR / "fastai_results_lm_cls.p", "wb") as f:
        pickle.dump(results_d, f)
