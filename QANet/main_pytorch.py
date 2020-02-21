import numpy as np
import pickle
import math
import os
import torch
import torch.nn.functional as F

from tqdm import trange
from datetime import datetime
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau, LambdaLR
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import config
from utils.config import parse_args
from utils.evaluate_v1 import convert_tokens, evaluate
from utils.text_utils import Vocab
from utils.ema import EMA

# from models.pytorch_models import QANet

from models.replica import QANet
# from original_models import QANet
import json

import pdb


n_cpus = os.cpu_count()
device = config.device
# writer = SummaryWriter(log_dir=config.log_dir)


def get_metrics(eval_file, ids, y1_pred, y2_pred, answer_dict):
    p1 = F.softmax(y1_pred, dim=1)
    p2 = F.softmax(y2_pred, dim=1)
    outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
    for j in range(outer.size()[0]):
        outer[j] = torch.triu(outer[j])
        #outer[j] = torch.tril(outer[j], config.ans_limit)
    a1, _ = torch.max(outer, dim=2)
    a2, _ = torch.max(outer, dim=1)
    ymin = torch.argmax(a1, dim=1)
    ymax = torch.argmax(a2, dim=1)
    # outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
    # outer = torch.triu(outer) - torch.triu(outer, diagonal=config.ans_limit + 1)
    # ymin = torch.argmax(torch.max(outer, dim=2)[0], dim=1)
    # ymax = torch.argmax(torch.max(outer, dim=1)[0], dim=1)
    answer_dict_, _ = convert_tokens(
        eval_file, ids.tolist(), ymin.tolist(), ymax.tolist()
    )
    answer_dict.update(answer_dict_)
    metrics = evaluate(eval_file, answer_dict)
    return metrics


def early_stopping(curr_value, best_value, stop_step, patience, is_lower_better=True):

    if isinstance(curr_value, list):
        is_lower = all([cv <= bv for cv, bv in zip(curr_value, best_value)])
        is_higher = all([cv >= bv for cv, bv in zip(curr_value, best_value)])
    else:
        is_lower = curr_value <= best_value
        is_higher = curr_value >= best_value

    if (is_lower_better and is_lower) or (not is_lower_better and is_higher):
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


class SQuADDataset(Dataset):
    def __init__(self, npz_file):

        data = np.load(npz_file)
        self.context_word_seqs = data["context_word_seqs"][rand_sample]
        self.context_char_seqs = data["context_char_seqs"][rand_sample]
        self.ques_word_seqs = data["ques_word_seqs"][rand_sample]
        self.ques_char_seqs = data["ques_char_seqs"][rand_sample]
        self.y1s = data["y1s"][rand_sample]
        self.y2s = data["y2s"][rand_sample]
        self.ids = data["ids"][rand_sample]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return (
            self.context_word_seqs[idx],
            self.context_char_seqs[idx],
            self.ques_word_seqs[idx],
            self.ques_char_seqs[idx],
            self.y1s[idx],
            self.y2s[idx],
            self.ids[idx],
        )


def train_step(model, optimizer, train_loader, train_ctx_ans, epoch, ema, scheduler=None):

    model.train()
    train_steps = len(train_loader)
    answer_dict = {}
    running_loss, running_f1, running_em = 0, 0, 0
    with trange(train_steps) as t:
        for batch_idx, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in zip(t, train_loader):
            t.set_description("epoch %i" % (epoch + 1))
            n_iter = batch_idx + (train_steps * epoch)

            Cwid, Ccid, Qwid, Qcid = (
                Cwid.to(device),
                Ccid.to(device),
                Qwid.to(device),
                Qcid.to(device),
            )
            y1, y2 = y1.to(device), y2.to(device)

            optimizer.zero_grad()
            y1_pred, y2_pred = model(Cwid, Ccid, Qwid, Qcid)
            loss1 = F.cross_entropy(y1_pred, y1)
            loss2 = F.cross_entropy(y2_pred, y2)
            loss = (loss1 + loss2)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
            optimizer.step()

            ema(model, n_iter)

            if isinstance(scheduler, CyclicLR) or isinstance(scheduler, LambdaLR):
                scheduler.step()

            # print(optimizer.param_groups[0]["lr"])
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)

            metrics = get_metrics(train_ctx_ans, ids, y1_pred, y2_pred, answer_dict)
            running_em += metrics["exact_match"]
            running_f1 += metrics["f1"]
            avg_em = running_em / (batch_idx + 1)
            avg_f1 = running_f1 / (batch_idx + 1)

            # writer.add_scalar('data/avg_loss', avg_loss, n_iter)
            # writer.add_scalar('data/EM', metrics["exact_match"], n_iter)
            # writer.add_scalar('data/f1', metrics["f1"], n_iter)

            t.set_postfix(f1=avg_f1, EM=avg_em, loss=avg_loss)


def eval_step(model, eval_loader, eval_ctx_ans, is_valid=True):

    model.eval()
    eval_steps = len(eval_loader)
    running_loss, running_f1, running_em = 0, 0, 0
    answer_dict = {}
    with torch.no_grad():
        with trange(eval_steps) as t:
            for batch_idx, (Cwid, Ccid, Qwid, Qcid, y1, y2, ids) in zip(t, eval_loader):
                if is_valid:
                    t.set_description("valid")
                else:
                    t.set_description("test")

                Cwid, Ccid, Qwid, Qcid = (
                    Cwid.to(device),
                    Ccid.to(device),
                    Qwid.to(device),
                    Qcid.to(device),
                )
                y1, y2 = y1.to(device), y2.to(device)
                y1_pred, y2_pred = model(Cwid, Ccid, Qwid, Qcid)
                loss1 = F.cross_entropy(y1_pred, y1)
                loss2 = F.cross_entropy(y2_pred, y2)
                loss = (loss1 + loss2)
                running_loss += loss.item()
                avg_loss = running_loss / (batch_idx + 1)

                metrics = get_metrics(eval_ctx_ans, ids, y1_pred, y2_pred, answer_dict)
                running_em += metrics["exact_match"]
                running_f1 += metrics["f1"]
                avg_em = running_em / (batch_idx + 1)
                avg_f1 = running_f1 / (batch_idx + 1)

                # n_iter = batch_idx+(train_steps*epoch)
                # writer.add_scalar('data/avg_loss', avg_loss, n_iter)
                # writer.add_scalar('data/EM', metrics["exact_match"], n_iter)
                # writer.add_scalar('data/f1', metrics["f1"], n_iter)

                t.set_postfix(f1=avg_f1, EM=avg_em, loss=avg_loss)

    return avg_loss, metrics["f1"], metrics["exact_match"]


if __name__ == "__main__":

    args = parse_args()
    model_name = "qanet_" + str(datetime.now()).replace(" ", "_")

    if args.full_train:
        train_ctx_ques = SQuADDataset(config.full_train_dir / "full_train_seq.npz")
        train_ctx_ans = pickle.load(
            open(config.full_train_dir / "full_train_c_a.p", "rb")
        )
        train_ctx_ques_loader = DataLoader(
            train_ctx_ques, shuffle=True, batch_size=args.batch_size
        )
        eval_ctx_ques = SQuADDataset(config.test_dir / "test_seq.npz")
        eval_ctx_ans = pickle.load(open(config.test_dir / "test_c_a.p", "rb"))
        eval_ctx_ques_loader = DataLoader(
            eval_ctx_ques, shuffle=False, batch_size=args.batch_size
        )
    else:
        np.random.seed(1)
        rand_sample = np.random.choice(np.arange(77375), 4800)
        train_ctx_ques = SQuADDataset(config.train_dir / "train_seq.npz")
        train_ctx_ans = pickle.load(open(config.train_dir / "train_c_a.p", "rb"))
        train_ctx_ques_loader = DataLoader(
            train_ctx_ques, shuffle=False, batch_size=args.batch_size
        )
        np.random.seed(2)
        rand_sample = np.random.choice(np.arange(9978), 2400)
        eval_ctx_ques = SQuADDataset(config.valid_dir / "valid_seq.npz")
        eval_ctx_ans = pickle.load(open(config.valid_dir / "valid_c_a.p", "rb"))
        eval_ctx_ques_loader = DataLoader(
            eval_ctx_ques, shuffle=False, batch_size=args.batch_size
        )

    if args.wd_pretrained:
        word_mat = pickle.load(open(config.data_dir / "word_emb_mtx.p", "rb"))
        word_vocab = None
    else:
        word_vocab = Vocab.load(config.data_dir / "word_vocab.p")
        word_mat = None

    if args.ch_pretrained:
        char_mat = pickle.load(open(config.data_dir / "char_emb_mtx.p", "rb"))
        char_vocab = None
    else:
        char_vocab = Vocab.load(config.data_dir / "char_vocab.p")
        char_mat = None

    with open("data/char_emb.pkl", "rb") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    # Model
    # model = QANet(
    #     word_mat=word_mat,
    #     char_mat=char_mat,
    #     word_vocab=word_vocab,
    #     char_vocab=char_vocab,
    #     d_word=args.d_word,
    #     d_char=args.d_char,
    #     d_model=args.d_model,
    #     dropout=args.dropout,
    #     dropout_char=args.dropout_char,
    #     para_limit=config.para_limit,
    #     ques_limit=config.ques_limit,
    #     n_heads=args.n_heads,
    #     freeze=args.freeze,
    # )

    model = QANet(word_mat, char_mat)

    model.to(device)

    ema = EMA(args.ema_decay)
    ema.register(model)

    # Optimizer
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta1),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta1),
            weight_decay=args.weight_decay,
        )

    # learning rate scheduler
    training_steps = len(train_ctx_ques_loader)
    step_size = training_steps * (args.n_epochs // (args.n_cycles * 2))
    if args.lr_scheduler.lower() == "lambdalr":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: math.log2(1 + steps) / math.log2(args.warm_up_steps)
            if steps < args.warm_up_steps
            else 1.0,
        )
    elif args.lr_scheduler.lower() == "reducelronplateau":
        # Since this scheduler runs within evaluation, and we evaluate every
        # eval_every epochs. Therefore the n_epochs before decreasing the lr
        # is lr_patience*eval_every (it we don't trigger early stop before)
        if args.early_stop_criterium == "loss":
            mode = "min"
        elif args.early_stop_criterium == "metric":
            mode = "max"
        scheduler = ReduceLROnPlateau(
            optimizer, patience=args.lr_patience, factor=0.4, mode=mode
        )
    elif args.lr_scheduler.lower() == "cycliclr":
        scheduler = CyclicLR(
            optimizer,
            step_size_up=step_size,
            base_lr=args.lr,
            max_lr=args.lr * 10.0,
            cycle_momentum=False,
        )
    else:
        scheduler = None

    # Train and validate
    best_metrics = [0, 0]
    best_loss = 1e6
    stop_step = 0
    for epoch in range(args.n_epochs):
        train_step(
            model, optimizer, train_ctx_ques_loader, train_ctx_ans, epoch, ema, scheduler
        )
        if epoch % args.eval_every == (args.eval_every - 1):
            ema.assign(model)
            val_loss, f1, em = eval_step(model, eval_ctx_ques_loader, eval_ctx_ans)
            if args.early_stop_criterium == "loss":
                best_loss, stop_step, stop = early_stopping(
                    val_loss, best_loss, stop_step, args.patience
                )
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
            elif args.early_stop_criterium == "metrics":
                best_metrics, stop_step, stop = early_stopping(
                    [f1, em],
                    best_metrics,
                    stop_step,
                    args.patience,
                    is_lower_better=False,
                )
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(f1 + em / 2.0)
            ema.resume(model)
        if stop:
            break
        if (stop_step == 0) & (args.save_results):
            best_epoch = epoch
            torch.save(model.state_dict(), config.weights_dir / (model_name + ".pt"))
            results_d = {}
            results_d["args"] = args.__dict__
            results_d["best_epoch"] = best_epoch
            results_d["loss"] = val_loss
            results_d["f1"] = f1
            results_d["em"] = em
            pickle.dump(
                results_d, open(os.path.join(config.log_dir, model_name) + ".p", "wb")
            )
