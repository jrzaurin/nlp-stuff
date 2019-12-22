import pandas as pd
import numpy as np
import pickle
import os

from pathlib import Path
from datetime import datetime
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, precision_score

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.metric import Accuracy

from models.mxnet_models import HierAttnNet, RNNAttn
from utils.parser import parse_args

import pdb

n_cpus = os.cpu_count()


def train_step(model, train_loader, trainer, metric, epoch):

    metric.reset()
    train_steps = len(train_loader)
    running_loss = 0
    with trange(train_steps) as t:
        for batch_idx, (data, target) in zip(t, train_loader):
            t.set_description("epoch %i" % (epoch + 1))

            X = data.as_in_context(ctx)
            y = target.as_in_context(ctx)

            with autograd.record():
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss.backward()

            trainer.step(X.shape[0])
            running_loss += nd.mean(loss).asscalar()
            avg_loss = running_loss / (batch_idx + 1)
            metric.update(preds=nd.argmax(y_pred, axis=1), labels=y)
            t.set_postfix(acc=metric.get()[1], loss=avg_loss)


def eval_step(model, eval_loader, metric, is_valid=True):

    metric.reset()
    eval_steps = len(eval_loader)
    running_loss = 0
    preds = []
    with trange(eval_steps) as t:
        for batch_idx, (data, target) in zip(t, eval_loader):
            if is_valid:
                t.set_description("valid")
            else:
                t.set_description("test")

            X = data.as_in_context(ctx)
            y = target.as_in_context(ctx)

            with autograd.record(train_mode=False):
                y_pred = model(X)
                loss = criterion(y_pred, y)

            running_loss += nd.mean(loss).asscalar()
            avg_loss = running_loss / (batch_idx + 1)
            metric.update(preds=nd.argmax(y_pred, axis=1), labels=y)
            preds.append(y_pred)
            t.set_postfix(acc=metric.get()[1], loss=avg_loss)

    return avg_loss, preds


def early_stopping(curr_value, best_value, stop_step, patience):
    if curr_value <= best_value:
        stop_step, best_value = 0, curr_value
    else:
        stop_step += 1
    if stop_step >= patience:
        print(
            "Early stopping triggered. patience: {} log:{}".format(patience, curr_value)
        )
        stop = True
    else:
        stop = False
    return best_value, stop_step, stop


if __name__ == "__main__":

    args = parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    test_dir = data_dir / "test"

    log_dir = Path(args.log_dir)
    model_weights = log_dir / "weights"
    results_tab = os.path.join(args.log_dir, "mx_results_df.csv")
    paths = [log_dir, model_weights]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    if args.model == "han":
        ftrain, fvalid, ftest = "han_train.npz", "han_valid.npz", "han_test.npz"
        tokf = "HANPreprocessor.p"
        model_name = (
            args.model
            + "_mx_"
            + "_lr_"
            + str(args.lr)
            + "_wdc_"
            + str(args.weight_decay)
            + "_bsz_"
            + str(args.batch_size)
            + "_whd_"
            + str(args.word_hidden_dim)
            + "_shd_"
            + str(args.sent_hidden_dim)
            + "_emb_"
            + str(args.embed_dim)
            + "_ini_"
            + args.init_method
            + "_pre_"
            + ("no" if args.embedding_matrix is None else "yes")
        )
    elif args.model == "rnn":
        ftrain, fvalid, ftest = "train.npz", "valid.npz", "test.npz"
        tokf = "TextPreprocessor.p"
        model_name = (
            args.model
            + "_mx_"
            + "_lr_"
            + str(args.lr)
            + "_wdc_"
            + str(args.weight_decay)
            + "_bsz_"
            + str(args.batch_size)
            + "_nl_"
            + str(args.num_layers)
            + "_hd_"
            + str(args.hidden_dim)
            + "_drp_"
            + str(args.rnn_dropout)
            + "_emb_"
            + str(args.embed_dim)
            + "_att_"
            + ("yes" if args.with_attention else "no")
            + "_pre_"
            + ("no" if args.embedding_matrix is None else "yes")
        )

    train_mtx = np.load(train_dir / ftrain)
    train_set = gluon.data.dataset.ArrayDataset(
        train_mtx["X_train"], train_mtx["y_train"]
    )
    train_loader = gluon.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, num_workers=n_cpus
    )

    valid_mtx = np.load(valid_dir / fvalid)
    eval_set = gluon.data.dataset.ArrayDataset(
        valid_mtx["X_valid"], valid_mtx["y_valid"]
    )
    eval_loader = gluon.data.DataLoader(
        dataset=eval_set, batch_size=args.batch_size, num_workers=n_cpus
    )

    test_mtx = np.load(test_dir / ftest)
    test_set = gluon.data.dataset.ArrayDataset(test_mtx["X_test"], test_mtx["y_test"])
    test_loader = gluon.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=n_cpus
    )

    tok = pickle.load(open(data_dir / tokf, "rb"))
    if args.model == "han":
        model = HierAttnNet(
            vocab_size=len(tok.vocab.stoi),
            maxlen_sent=tok.maxlen_sent,
            maxlen_doc=tok.maxlen_doc,
            word_hidden_dim=args.word_hidden_dim,
            sent_hidden_dim=args.sent_hidden_dim,
            padding_idx=args.padding_idx,
            embed_dim=args.embed_dim,
            embed_dropout=args.embed_dropout,
            embedding_matrix=args.embedding_matrix,
            num_class=args.num_class,
            init_method=args.init_method,
        )
    elif args.model == "rnn":
        model = RNNAttn(
            vocab_size=len(tok.vocab.stoi),
            maxlen=tok.maxlen,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            rnn_dropout=args.rnn_dropout,
            padding_idx=args.padding_idx,
            embed_dim=args.embed_dim,
            embed_dropout=args.embed_dropout,
            embedding_matrix=args.embedding_matrix,
            num_class=args.num_class,
            with_attention=args.with_attention,
        )

    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
    model.initialize(ctx=ctx)
    model.hybridize()
    criterion = gluon.loss.SoftmaxCELoss()
    metric = Accuracy()
    trainer = gluon.Trainer(
        model.collect_params(),
        "Adam",
        {"learning_rate": args.lr, "wd": args.weight_decay},
    )

    stop_step = 0
    best_loss = 1e6
    for epoch in range(args.n_epochs):
        train_step(model, train_loader, trainer, metric, epoch)
        if epoch % args.eval_every == (args.eval_every - 1):
            val_loss, _ = eval_step(model, eval_loader, metric)
            best_loss, stop_step, stop = early_stopping(
                val_loss, best_loss, stop_step, args.patience
            )
        if stop == True:
            break
        if (stop_step == 0) & (args.save_results):
            best_epoch = epoch
            model.save_parameters(str(model_weights / (model_name + ".params")))

    if args.save_results:

        metric.reset()
        model.load_parameters(str(model_weights / (model_name + ".params")), ctx=ctx)
        test_loss, preds = eval_step(model, test_loader, metric, is_valid=False)
        pdb.set_trace()
        preds = nd.concat(*preds, dim=0)
        preds = nd.argmax(nd.softmax(preds), axis=1).asnumpy()
        test_acc = accuracy_score(test_mtx["y_test"], preds)
        test_f1 = f1_score(test_mtx["y_test"], preds, average="weighted")
        test_prec = precision_score(test_mtx["y_test"], preds, average="weighted")

        cols = ["modelname", "loss", "acc", "f1", "prec", "best_epoch"]
        vals = [model_name, test_loss, test_acc, test_f1, test_prec, best_epoch]
        if not os.path.isfile(results_tab):
            results_df = pd.DataFrame(columns=cols)
            experiment_df = pd.DataFrame(data=[vals], columns=cols)
            results_df = results_df.append(experiment_df, ignore_index=True, sort=False)
            results_df.to_csv(results_tab, index=False)
        else:
            results_df = pd.read_csv(results_tab)
            experiment_df = pd.DataFrame(data=[vals], columns=cols)
            results_df = results_df.append(experiment_df, ignore_index=True, sort=False)
            results_df.to_csv(results_tab, index=False)
