import logging
import time
import torch.optim
import json
import batchloader
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")

ITERATIONS_BETWEEN_CHECKPOINTS = 50
EPOCHS_BETWEEN_CHECKPOINTS = 1
eval_influence = 0.9
scale = 400


# https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/2
class WeightClipper:

    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data.clamp_(-1.98, 1.98)


def save_model(model: torch.nn.Module, epoch, experiment):
    print("Saving model...")

    path = "nets/" + experiment + "_" + (str(epoch) if epoch != 0 else "final")

    torch.save(model.state_dict(), path + ".net")

    with open(path + ".json", "w") as f:
        data = {}
        for param_tensor in model.state_dict():
            data[param_tensor] = model.state_dict()[param_tensor].numpy().tolist()
        json.dump(data, f)

    print("Model saved!")
    logging.info("Model saved!")


def test_validation(model: torch.nn.Module, validation):
    total_loss = 0
    iterations = 0

    with batchloader.BatchProvider(validation, 16384, 1) as provider:
        for batch in provider:
            white_features, black_features, stm, scores, wdl = batch.get_tensors()

            output = model(white_features, black_features, stm)
            expected = torch.sigmoid(scores / scale) * eval_influence + wdl * (1 - eval_influence)

            loss = torch.mean((output - expected) ** 2)

            total_loss += loss.item()
            iterations += 1

    return total_loss / iterations


def train(batch_provider: batchloader.BatchProvider, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, validation, experiment):
    epoch = 1
    iterations = 0
    since_checkpoint = 0
    epoch_time = time.time()
    positions = 0
    current_loss = 0

    checkpoints = []
    train_losses = []
    epochs = []
    val_losses = []

    clipper = WeightClipper()

    plt.ion()

    print("Started training...")
    logging.info("Started training...")
    try:
        for batch in batch_provider:

            # Start of a new epoch
            if batch_provider.reader.contents.epoch != epoch:
                current_time = time.time()

                validation_loss = test_validation(model, validation)

                epochs.append(iterations)
                val_losses.append(validation_loss)

                print("------------------------")
                print(f"Epoch = {epoch}\n"
                      f"Epoch time = {round(current_time - epoch_time + 1)}s\n"
                      f"Positions/second = {round(positions / (current_time - epoch_time + 1))}\n"
                      f"Training loss = {current_loss / since_checkpoint}\n"
                      f"Validation loss = {validation_loss}")
                print("------------------------")
                print()

                logging.info(f"Epoch {epoch} finished. "
                             f"(train loss = {current_loss / since_checkpoint} val loss = {validation_loss}")

                if epoch % EPOCHS_BETWEEN_CHECKPOINTS == 0:
                    save_model(model, epoch, experiment)

                current_loss = 0
                epoch = batch_provider.reader.contents.epoch
                positions = 0
                epoch_time = time.time()

            # Updating variables
            white_features, black_features, stm, scores, wdl = batch.get_tensors()

            # The actual training
            optimizer.zero_grad()
            output = model(white_features, black_features, stm)

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
                train_losses.append(current_loss / since_checkpoint)

                plt.plot(checkpoints, train_losses, label="Train loss")
                plt.plot(epochs, val_losses, label="Val loss")

                plt.title("Training")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")

                plt.draw()

                plt.savefig("graph.png")

                current_loss = 0
                since_checkpoint = 0
    except KeyboardInterrupt:
        save_model(model, 0, experiment)
        print("Training has stopped!")
        exit(0)

    print(f"\n\n"
          f"Finished training of {epoch} epochs!\n")

    save_model(model, epoch, experiment)

    logging.info("Training has finished.\n")
