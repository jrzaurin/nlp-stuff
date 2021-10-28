import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run Amazon Reviews classifier.")

    parser.add_argument(
        "--data_dir", type=str, default="processed_data", help="Input data path."
    )
    parser.add_argument(
        "--log_dir", type=str, default="results", help="Store model path."
    )

    # Set model
    parser.add_argument(
        "--head_hidden_dim",
        type=str,
        default="[256,64]",
        help="head hidden dimensions.",
    )
    parser.add_argument("--head_dropout", type=float, default=0.1, help="head dropout.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="bert family model name",
    )
    parser.add_argument(
        "--freeze_bert", action="store_true", help="freeze pretrained weights"
    )
    parser.add_argument("--num_class", type=int, default=4, help="number of classes.")

    # Train/Test parameters
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epoch.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="l2 reg.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr_scheduler", action="store_true", help="use lr scheduler.")

    parser.add_argument(
        "--save_results", action="store_true", help="Save model and results"
    )
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=2, help="Patience for early stopping"
    )

    return parser.parse_args()
