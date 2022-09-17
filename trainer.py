import logging
import time
import torch.optim
import json
import batchloader
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")

ITERATIONS_BETWEEN_CHECKPOINTS = 200
EPOCHS_BETWEEN_CHECKPOINTS = 1
eval_influence = 0.9
scale = 400


# https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/2
class WeightClipper:

    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data.clamp_(-1.98, 1.98)


def save_model(model: torch.nn.Module, epoch):
    print("Saving model...")

    path = "nets/nnue_" + (str(epoch) if epoch != 0 else "final")

    torch.save(model.state_dict(), path + ".net")

    with open(path + ".json", "w") as f:
        data = {}
        for param_tensor in model.state_dict():
            data[param_tensor] = model.state_dict()[param_tensor].numpy().tolist()
        json.dump(data, f)

    print("Model saved!")
    logging.info("Model saved!")


def train(batch_provider: batchloader.BatchProvider, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    epoch = 1
    iterations = 0
    since_checkpoint = 0
    epoch_time = time.time()
    positions = 0
    current_loss = 0

    checkpoints = []
    losses = []

    clipper = WeightClipper()

    plt.ion()

    print("Started training...")
    logging.info("Started training...")
    try:
        for batch in batch_provider:

            # Start of a new epoch
            if batch_provider.reader.contents.epoch != epoch:
                current_time = time.time()

                print("------------------------")
                print(f"Epoch = {epoch}\n"
                      f"Epoch time = {round(current_time - epoch_time + 1)}s\n"
                      f"Positions/second = {round(positions / (current_time - epoch_time + 1))}\n"
                      f"Total loss = {current_loss / since_checkpoint}")
                print("------------------------")
                print()

                if epoch % EPOCHS_BETWEEN_CHECKPOINTS == 0:
                    save_model(model, epoch)

                current_loss = 0
                epoch = batch_provider.reader.contents.epoch
                since_checkpoint = 0
                positions = 0
                logging.info(f"Started epoch {epoch}")
                epoch_time = time.time()

            # Updating variables
            features, scores, wdl = batch.get_tensors()

            # The actual training
            optimizer.zero_grad()
            output = model(features)

            # Calculating loss using mean square error
            expected = torch.sigmoid(scores / scale) * eval_influence + wdl * (1 - eval_influence)
            loss = torch.mean((output - expected) ** 2)
            loss.backward()

            # Optimizer step
            optimizer.step()

            # We clip the weights, so we can scale them to int8
            model.apply(clipper)

            # Updating variables for logging training process
            iterations += 1
            since_checkpoint += 1
            positions += batch.size

            with torch.no_grad():
                current_loss += loss.item()

            if since_checkpoint >= ITERATIONS_BETWEEN_CHECKPOINTS:
                print(f"Iteration = {iterations}\nLoss = {current_loss / since_checkpoint}\n")

                # Plotting
                checkpoints.append(iterations)
                losses.append(current_loss / since_checkpoint)

                plt.plot(checkpoints, losses)

                plt.title("Training")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")

                plt.draw()

                plt.savefig("graph.png")

                current_loss = 0
                since_checkpoint = 0
    except KeyboardInterrupt:
        save_model(model, 0)
        print("Training has stopped!")
        exit(0)

    print(f"\n\n"
          f"Finished training of {epoch} epochs!\nSaving model...")

    save_model(model, epoch)

    logging.info("Training has finished.\nSaving model...")