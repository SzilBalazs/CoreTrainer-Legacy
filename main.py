#!/usr/bin/env python3

import torch
import argparse
import logging
import os.path
import batchloader
import trainer

from model import NNUE

ARCHITECTURE = """
+-------------------------+
|    NNUE Architecture    |
|     2x(768->256)->1     |
|    Activation: ReLU     |
+-------------------------+
"""


def main():
    parser = argparse.ArgumentParser(description="NNUE trainer")

    parser.add_argument("experiment", help="Name of the experiment")
    parser.add_argument("training", help="Path to the .bin file containing training data")
    parser.add_argument("validation", help="Path to the .bin file containing validation data")

    parser.add_argument("--epochs", type=int, dest="epochs", default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16384, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LEARNING_RATE", help="Learning rate")

    logging.basicConfig(filename="train.log", filemode="w", format="%(asctime)s: %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S", level=logging.DEBUG)

    logging.info("CoreTrainer has started.")

    args = parser.parse_args()

    if not os.path.exists(args.training):
        logging.error(f"Training data {args.training} does not exist!")
        raise Exception(f"Training data {args.training} does not exist!")

    if not os.path.exists(args.validation):
        logging.error(f"Validation data {args.training} does not exist!")
        raise Exception(f"Validation data {args.training} does not exist!")

    print("\nTraining data:", args.training, "\n")
    print("\nValidation data:", args.validation, "\n")

    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)

    print(ARCHITECTURE)

    model = NNUE()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with batchloader.BatchProvider(args.training, args.batch_size, args.epochs) as batch_provider:
        trainer.train(batch_provider, model, optimizer, args.validation, args.experiment)


if __name__ == "__main__":
    main()
