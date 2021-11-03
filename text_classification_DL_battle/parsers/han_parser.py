import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Amazon Reviews classifier.")

    parser.add_argument(
        "--data_dir", type=str, default="processed_data", help="Input data path."
    )
    parser.add_argument(
        "--log_dir", type=str, default="results", help="Store model path."
    )

    # Parameters that are common to HierAttnNet and RNNAttn
    parser.add_argument(
        "--padding_idx",
        type=int,
        default=1,
        help="padding index for the numericalised token sequences.",
    )
    parser.add_argument(
        "--zero_padding",
        action="store_true",
        help="Manually zero padding idx when using mxnet.",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=50, help="input embeddings dimension."
    )
    parser.add_argument(
        "--embed_drop",
        type=float,
        default=0.0,
        help="embeddings dropout. Taken from the awd-lstm lm from Salesforce: https://github.com/salesforce/awd-lstm-lm",
    )
    parser.add_argument(
        "--locked_drop",
        type=float,
        default=0.0,
        help="embeddings dropout. Taken from the awd-lstm lm from Salesforce: https://github.com/salesforce/awd-lstm-lm",
    )
    parser.add_argument(
        "--last_drop",
        type=float,
        default=0.0,
        help="dropout before the last fully connected layer (i.e. the prediction layer)",
    )
    parser.add_argument(
        "--embedding_matrix",
        type=str,
        default=None,
        help="path to the pretrained word vectors.",
    )
    parser.add_argument("--num_class", type=int, default=4, help="number of classes.")

    # HAN parameters
    parser.add_argument(
        "--word_hidden_dim",
        type=int,
        default=32,
        help="hidden dimension for the GRU processing words.",
    )
    parser.add_argument(
        "--sent_hidden_dim",
        type=int,
        default=32,
        help="hidden dimension for the GRU processing senteces.",
    )
    parser.add_argument(
        "--weight_drop",
        type=float,
        default=0.0,
        help="weight dropout. Taken from the awd-lstm lm from Salesforce: https://github.com/salesforce/awd-lstm-lm",
    )

    # RNN parameters
    parser.add_argument(
        "--num_layers", type=int, default=3, help="number of LSTMs to stack"
    )
    parser.add_argument(
        "--rnn_dropout", type=float, default=0.0, help="internal rnn dropout."
    )
    parser.add_argument("--hidden_dim", type=int, default=32, help="LSTM's hidden_dim")
    parser.add_argument(
        "--with_attention", action="store_true", help="LSTM with/without attention"
    )

    # Train/Test parameters
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of epoch.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="l2 reg.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="no",
        help="Specify the lr_scheduler {multifactorscheduler, reducelronplateau, cycliclr, no (nothing)}",
    )
    parser.add_argument(
        "--n_cycles", type=int, default=1, help="number of cycles when using cycliclr"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=2, help="Patience for early stopping"
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1,
        help="Patience for ReduceLROnPlateau lr_scheduler before decreasing lr",
    )
    parser.add_argument(
        "--steps_epochs",
        type=str,
        default="[2,4,6]",
        help="list of steps to schedule a change for the multifactorscheduler scheduler",
    )

    return parser.parse_args()
