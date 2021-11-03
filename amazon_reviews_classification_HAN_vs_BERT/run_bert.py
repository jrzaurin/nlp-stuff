import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from transformers import AdamW, get_linear_schedule_with_warmup

from models.bert import BertClassifier
from parsers.bert_parser import parse_args
from utils.metrics import CategoricalAccuracy

n_cpus = os.cpu_count()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train_step(model, optimizer, train_loader, epoch, metric, scheduler=None):
    model.train()
    metric.reset()
    train_steps = len(train_loader)
    running_loss = 0
    with trange(train_steps) as t:
        for batch_idx, (data, mask, target) in zip(t, train_loader):
            t.set_description("epoch %i" % (epoch + 1))

            X = data.to(device)
            M = mask.to(device)
            y = target.to(device)

            optimizer.zero_grad()
            y_pred = model(X, M)
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
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
            for batch_idx, (data, mask, target) in zip(t, eval_loader):
                if is_valid:
                    t.set_description("valid")
                else:
                    t.set_description("test")

                X = data.to(device)
                M = mask.to(device)
                y = target.to(device)

                y_pred = model(X, M)
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
            "Early stopping triggered. patience: {} log:{}".format(patience, best_value)
        )
        stop = True
    else:
        stop = False
    return best_value, stop_step, stop


def load_arrays_and_return_loaders(
    train_dir, valid_dir, test_dir, ftrain, fvalid, ftest, batch_size
):

    train_mtx = np.load(train_dir / ftrain)
    train_set = TensorDataset(
        torch.from_numpy(train_mtx["X_train"]),
        torch.from_numpy(train_mtx["mask_train"]),
        torch.from_numpy(train_mtx["y_train"]).long(),
    )
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, num_workers=n_cpus, shuffle=True
    )

    valid_mtx = np.load(valid_dir / fvalid)
    eval_set = TensorDataset(
        torch.from_numpy(valid_mtx["X_valid"]),
        torch.from_numpy(valid_mtx["mask_valid"]),
        torch.from_numpy(valid_mtx["y_valid"]).long(),
    )
    eval_loader = DataLoader(
        dataset=eval_set, batch_size=batch_size, num_workers=n_cpus, shuffle=False
    )

    test_mtx = np.load(test_dir / ftest)
    test_set = TensorDataset(
        torch.from_numpy(test_mtx["X_test"]),
        torch.from_numpy(test_mtx["mask_test"]),
        torch.from_numpy(test_mtx["y_test"]).long(),
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, num_workers=n_cpus, shuffle=False
    )

    return train_loader, eval_loader, test_loader, test_mtx


if __name__ == "__main__":  # noqa: C901

    args = parse_args()

    data_dir = Path(args.data_dir)

    if args.save_results:
        suffix = str(datetime.now()).replace(" ", "_").split(".")[:-1][0]
        filename = "_".join(["bert", suffix]) + ".p"
        log_dir = Path(args.log_dir)
        model_weights_dir = log_dir / "bert_weights"
        paths = [log_dir, model_weights_dir]
        for p in paths:
            if not os.path.exists(p):
                os.makedirs(p)

    train_loader, eval_loader, test_loader, test_mtx = load_arrays_and_return_loaders(
        train_dir=data_dir / "train",
        valid_dir=data_dir / "valid",
        test_dir=data_dir / "test",
        ftrain="_".join([args.model_name, "train.npz"]),
        fvalid="_".join([args.model_name, "valid.npz"]),
        ftest="_".join([args.model_name, "test.npz"]),
        batch_size=args.batch_size,
    )

    model = BertClassifier(
        head_hidden_dim=eval(args.head_hidden_dim),
        model_name=args.model_name,
        freeze_bert=args.freeze_bert,
        head_dropout=args.head_dropout,
        num_class=args.num_class,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.with_scheduler:
        total_steps = len(train_loader) * args.n_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
    else:
        scheduler = None

    metric = CategoricalAccuracy()
    stop_step = 0
    best_loss = 1e6
    for epoch in range(args.n_epochs):
        train_step(model, optimizer, train_loader, epoch, metric, scheduler)
        if epoch % args.eval_every == (args.eval_every - 1):
            val_loss, _ = eval_step(model, eval_loader, metric)
            best_loss, stop_step, stop = early_stopping(
                val_loss, best_loss, stop_step, args.patience
            )
        if stop:
            break
        if (stop_step == 0) & (args.save_results):
            best_epoch = epoch
            torch.save(model.state_dict(), model_weights_dir / filename)

    if args.save_results:

        results_d = {}
        results_d["args"] = args.__dict__

        model.load_state_dict(torch.load(model_weights_dir / filename))
        test_loss, preds = eval_step(model, test_loader, metric, is_valid=False)

        preds = (F.softmax(torch.cat(preds), 1).topk(1, 1)[1]).cpu().numpy().squeeze(1)

        results_d["loss"] = test_loss
        results_d["acc"] = accuracy_score(test_mtx["y_test"], preds)
        results_d["f1"] = f1_score(test_mtx["y_test"], preds, average="weighted")
        results_d["prec"] = precision_score(
            test_mtx["y_test"], preds, average="weighted"
        )
        results_d["best_epoch"] = best_epoch

        with open(log_dir / filename, "wb") as f:
            pickle.dump(results_d, f)
