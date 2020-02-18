import argparse
import torch
import os

from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = Path("data")
squad_dir = data_dir / "squad"
orig_train_fpath = squad_dir / "train-v1.1.json"
orig_test_fpath = squad_dir / "dev-v1.1.json"

train_dir = data_dir / "train"
test_dir = data_dir / "test"
valid_dir = data_dir / "valid"
weights_dir = data_dir / "weights"
logs_dir = data_dir / "logs"
glove_dir = data_dir / "glove"
fasttext_dir = data_dir / "fasttext"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(valid_dir)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
if not os.path.exists(glove_dir):
    os.makedirs(glove_dir)
if not os.path.exists(fasttext_dir):
    os.makedirs(fasttext_dir)

glove_wordv_fpath = glove_dir / "glove.6B.300d.txt"
glove_charv_fpath = glove_dir / "glove.840B.300d-char.txt"
fastt_wordv_fpath = None

para_limit = 400
ques_limit = 50
ans_limit = 30
char_limit = 16


def parse_args():

    # model related parameters
    parser.add_argument(
        "--wd_pretrained", action="store_true", help="pretrained word embeddings",
    )
    parser.add_argument(
        "--ch_pretrained", action="store_true", help="pretrained character embeddings",
    )
    parser.add_argument(
        "--freeze", action="store_true", help="freeze pretrained word embeddings"
    )
    parser.add_argument(
        "--d_word", type=int, default=300, help="size of the word embeddings",
    )
    parser.add_argument(
        "--d_char", type=int, default=64, help="size of the character embeddings",
    )
    parser.add_argument(
        "--d_model", type=int, default=96, help="size of the model representation",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout throughout the entire model and for the word embeddings",
    )
    parser.add_argument(
        "--dropout_char", type=float, default=0.05, help="character embeddings dropout",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=1,
        help="number of heads for the attention mechanism",
    )

    # Training related parameters
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="number of epochs",
    )
    parser.add_argument(
        "--full_train",
        action="store_true",
        help="Whether to use the full train dataset",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning_rate",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=10.0, help="Clips gradient at grad_clip",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="optimizer to use. one of Adam/AdamW",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="l2 reg.")
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1 in the Adam optimiser (or AdamW)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="beta2 in the Adam optimiser (or AdamW)",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="exponential moving average decay",
    )
    parser.add_argument(
        "--n_cycles", type=int, default=1, help="number of cycles when using cycliclr"
    )
    parser.add_argument(
        "--warm_up_steps",
        type=int,
        default=1000,
        help="number of steps to warm up for a LambdaLR scheduler",
    )
    parser.add_argument(
        "--steps_epochs",
        type=str,
        default="[2,4,6]",
        help="list of steps to schedule a change for the multifactorscheduler scheduler",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1,
        help="Patience for ReduceLROnPlateau lr_scheduler before decreasing lr",
    )

    # Evaluation related parameters
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="evaluate and save every eval_every number of epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="number or epochs before early stopping",
    )
