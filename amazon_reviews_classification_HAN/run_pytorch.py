import pandas as pd
import numpy as np
import pickle
import torch
import os
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, precision_score
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader

from models.pytorch_models import HierAttnNet, RNNAttn
from utils.metrics import CategoricalAccuracy
from utils.parser import parse_args

import pdb

n_cpus = os.cpu_count()
use_cuda = torch.cuda.is_available()


def train_step(model, optimizer, train_loader, epoch, scheduler, metric):
    model.train()
    metric.reset()
    train_steps = len(train_loader)
    running_loss = 0
    with trange(train_steps) as t:
        for batch_idx, (data, target) in zip(t, train_loader):
            t.set_description("epoch %i" % (epoch + 1))

            X = data.cuda() if use_cuda else data
            y = target.cuda() if use_cuda else target

            optimizer.zero_grad()
            y_pred = model(X)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            if isinstance(scheduler, CyclicLR):
                scheduler.step()

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            acc = metric(F.softmax(y_pred, dim=1), y)

            t.set_postfix(acc=acc, loss=avg_loss)


def eval_step(model, eval_loader, metric, is_valid=True):
    model.eval()
    metric.reset()
    eval_steps = len(eval_loader)
    running_loss = 0
    preds = []
    with torch.no_grad():
        with trange(eval_steps) as t:
            for batch_idx, (data, target) in zip(t, eval_loader):
                if is_valid:
                    t.set_description("valid")
                else:
                    t.set_description("test")

                X = data.cuda() if use_cuda else data
                y = target.cuda() if use_cuda else target

                y_pred = model(X)
                loss = F.cross_entropy(y_pred, y)
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)
                acc = metric(F.softmax(y_pred, dim=1), y)
                preds.append(y_pred)

                t.set_postfix(acc=acc, loss=avg_loss)

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
    results_tab = os.path.join(args.log_dir, "results_df.csv")
    paths = [log_dir, model_weights]
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

    if args.model == "han":
        ftrain, fvalid, ftest = "han_train.npz", "han_valid.npz", "han_test.npz"
        tokf = "HANPreprocessor.p"
        model_name = (
            args.model
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
    train_set = TensorDataset(
        torch.Tensor(train_mtx["X_train"]), torch.Tensor(train_mtx["y_train"]).long()
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, num_workers=n_cpus
    )

    valid_mtx = np.load(valid_dir / fvalid)
    eval_set = TensorDataset(
        torch.Tensor(valid_mtx["X_valid"]), torch.Tensor(valid_mtx["y_valid"]).long()
    )
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=args.batch_size, num_workers=n_cpus, shuffle=False
    )

    test_mtx = np.load(test_dir / ftest)
    test_set = TensorDataset(
        torch.Tensor(test_mtx["X_test"]), torch.Tensor(test_mtx["y_test"]).long()
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=n_cpus, shuffle=False
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

    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    training_steps = len(train_loader)
    step_size = training_steps * (args.n_epochs // (args.n_cycles * 2))
    if args.lr_scheduler.lower() == "reducelronplateau":
        # Since this scheduler runs within evaluation, and we evaluate every
        # eval_every epochs. Therefore the n_epochs before decreasing the lr
        # is lr_patience*eval_every (it we don't trigger early stop before)
        scheduler = ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=0.4)
    elif args.lr_scheduler.lower() == "cycliclr":
        scheduler = CyclicLR(
            optimizer,
            step_size_up=step_size,
            base_lr=args.lr,
            max_lr=args.lr * 10,
            cycle_momentum=False,
        )
    else:
        scheduler = None

    metric = CategoricalAccuracy()
    stop_step = 0
    best_loss = 1e6
    for epoch in range(args.n_epochs):
        train_step(model, optimizer, train_loader, epoch, scheduler, metric)
        if epoch % args.eval_every == (args.eval_every - 1):
            val_loss, _ = eval_step(model, eval_loader, metric)
            best_loss, stop_step, stop = early_stopping(
                val_loss, best_loss, stop_step, args.patience
            )
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
        if stop == True:
            break
        if (stop_step == 0) & (args.save_results):
            best_epoch = epoch
            torch.save(model.state_dict(), model_weights / (model_name + ".pt"))

    if args.save_results:

        model.load_state_dict(torch.load(model_weights / (model_name + ".pt")))
        test_loss, preds = eval_step(model, test_loader, metric, is_valid=False)
        preds = (F.softmax(torch.cat(preds), 1).topk(1, 1)[1]).cpu().numpy().squeeze(1)
        test_acc  = accuracy_score(test_mtx["y_test"], preds)
        test_f1   = f1_score(test_mtx["y_test"], preds, average="weighted")
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
