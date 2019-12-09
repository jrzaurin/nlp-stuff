import argparse
import os
import sys
import numpy as np

from joblib import dump
from random import choice
from glob import glob
from tqdm import tqdm,trange

import torch
from torch.autograd import Variable
from train_pytorch import generate_batches, use_cuda, n_chars
from torch_utils import RNNCharTagger, BiRNNCharTagger


def main(model_path, output_dir, dir_a, dir_b, min_jump_a, max_jump_a, min_jump_b, max_jump_b,
    steps, sequence_length, bidirectional, lstm_layers, rnn_size, batch_size, dropout):

    val_a = glob(os.path.join(dir_a, "test/*"))
    val_b = glob(os.path.join(dir_b, "test/*"))
    juma = [min_jump_a, max_jump_a]
    jumb = [min_jump_b, max_jump_b]

    # the model needs to be the same that was saved
    if bidirectional:
        model  = BiRNNCharTagger(lstm_layers,n_chars,rnn_size,batch_size,dropout)
    else:
        model  = RNNCharTagger(lstm_layers,n_chars,rnn_size,batch_size,dropout)
    model.load_state_dict(torch.load(model_path))
    if use_cuda:
        model = model.cuda()
    model.eval()

    gen = generate_batches(val_a, juma, val_b, jumb, batch_size, sequence_length, return_text=True)

    predictions, labels, texts = [],[],[]
    with trange(steps) as t:
        for i in t:
            X,y,text = gen.__next__()
            X_var = Variable(torch.from_numpy(X).float())
            y_var = Variable(torch.from_numpy(y).float())
            if use_cuda:
                X_var, y_var = X_var.cuda(), y_var.cuda()
            pr = model(X_var)
            predictions.append(pr.data)
            labels.append(y_var.data)
            texts.append(text)

    preds = torch.cat(predictions,dim=1).reshape(batch_size,steps*sequence_length)
    preds = preds.cpu().numpy()
    labs = torch.cat(labels,dim=1).reshape(batch_size,steps*sequence_length)
    labs = labs.cpu().numpy()
    txts = []
    for j in range(batch_size):
        txts.append("".join([texts[i][j] for i in range(steps)]))

    try:
        os.makedirs(output_dir)
    except os.error:
        pass
    for i in range(batch_size):
        path = os.path.join(output_dir, 'part_' + str(i).zfill(5) + ".joblib")
        dump((txts[i], preds[i], labs[i]), path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make predictions with a trained tagger")
    parser.add_argument("model_path", help="path to trained model")
    parser.add_argument("output_dir", help="where to put predictions")
    parser.add_argument("dir_a", help="directory with first set of input files (it should contain "
                                      "'test' subdirectory")
    parser.add_argument("dir_b", help="directory with the second set of input files (it shouold "
                                      "contain 'test' subdirectory)")
    parser.add_argument("--min_jump_a", type=int, default=20, help="snippets from source A will "
                                                                   "be at least this long")
    parser.add_argument("--max_jump_a", type=int, default=200, help="snippets from source B will "
                                                                    "be at most this long")
    parser.add_argument("--min_jump_b", type=int, default=20, help="snippets from source B will "
                                                                   "be at least this long")
    parser.add_argument("--max_jump_b", type=int, default=200, help="snippets from source B will "
                                                                    "be at most this long")
    parser.add_argument("--steps", type=int, default=50, help="how many batches to predict")
    parser.add_argument("--sequence_length", type=int, default=100, help="how many characters in "
                                                                         "single sequence")
    parser.add_argument("--bidirectional", action="store_true", help="Whether to use bidirectional LSTM")
    parser.add_argument("--lstm_layers", type=int, default=3, help="how many LSTM layers")
    parser.add_argument("--rnn_size", type=int, default=128, help="how many LSTM units per layer")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--dropout", type=int, default=0.2, help="dropout rate")

    args = parser.parse_args()
    main(args.model_path,
        args.output_dir,
        args.dir_a,
        args.dir_b,
        args.min_jump_a,
        args.max_jump_a,
        args.min_jump_b,
        args.max_jump_b,
        args.steps,
        args.sequence_length,
        args.bidirectional,
        args.lstm_layers,
        args.rnn_size,
        args.batch_size,
        args.dropout)


