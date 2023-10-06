import torch
import sys
from torch.nn import functional as F
import torch.nn as nn

from vast import tools
import Metrics
import Networks
from Penalties import negative_penalty

from Training import Dataset, Training


class Training_Mixed(Training):
    def __init__(self):
        super().__init__()
        if (self.args.arch != "LeNet_pp" or self.args.approach == "OOD"):
            print("Invalid arguments!")
            sys.exit()
        self.third_loss_func = nn.BCEWithLogitsLoss()
        self.net = Networks.__dict__[self.args.arch](network_type=self.args.net_type, mixed = True)
        self.net = tools.device(self.net)


    def train(self):
        torch.manual_seed(0)

        # train network
        prev_confidence = None
        for epoch in range(1, self.args.no_of_epochs + 1, 1):
            loss_history = []
            train_accuracy = torch.zeros(2, dtype=int)
            train_magnitude = torch.zeros(2, dtype=float)
            train_confidence = torch.zeros(2, dtype=float)
            self.net.train()
            for x, y in self.train_data_loader:
                x = tools.device(x)
                y = tools.device(y)
                self.optimizer.zero_grad()
                logits, logits_2, features = self.net(x)

                loss = self.first_loss_func(logits, y) + self.args.second_loss_weight * self.second_loss_func(features, y) + self.args.negative_penalty_weight * negative_penalty(self.net.single_fc.weight)
                y_2 = torch.where(y < 0, torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                y_2 = y_2.unsqueeze(1).float()
                loss += self.third_loss_func(logits_2, y_2)

                # metrics on training set
                train_accuracy += Metrics.accuracy(logits, y)
                train_confidence += Metrics.confidence(logits, y)
                if self.args.approach not in ("SoftMax", "OOD"):
                    train_magnitude += Metrics.sphere(features, y, self.args.Minimum_Knowns_Magnitude if self.args.approach in self.args.approach == "Objectosphere" else None)

                loss_history.extend(loss.tolist())
                loss.mean().backward()
                self.optimizer.step()

            # metrics on validation set
            with torch.no_grad():
                val_loss = torch.zeros(2, dtype=float)
                val_accuracy = torch.zeros(2, dtype=int)
                val_magnitude = torch.zeros(2, dtype=float)
                val_confidence = torch.zeros(2, dtype=float)
                self.net.eval()
                for x,y in self.val_data_loader:
                    x = tools.device(x)
                    y = tools.device(y)
                    logits, logits_2 ,features = self.net(x)

                    loss = self.first_loss_func(logits, y) + self.args.second_loss_weight * self.second_loss_func(features, y) + self.args.negative_penalty_weight * negative_penalty(self.net.single_fc.weight)
                    
                    y_2 = torch.where(y < 0, torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                    y_2 = y_2.unsqueeze(1).float()
                    loss += self.third_loss_func(logits_2, y_2)

                    val_loss += torch.tensor((torch.sum(loss), len(loss)))
                    val_accuracy += Metrics.accuracy(logits, y)
                    val_confidence += Metrics.confidence(logits, y)
                    if self.args.approach not in ("SoftMax", "OOD"):
                        val_magnitude += Metrics.sphere(features, y, self.args.Minimum_Knowns_Magnitude if self.args.approach == "Objectosphere" else None)

            # log statistics
            epoch_running_loss = torch.mean(torch.tensor(loss_history))

            # save network based on confidence metric of validation set
            save_status = "NO"
            if prev_confidence is None or (val_confidence[0] > prev_confidence):
                torch.save(self.net.state_dict(), self.model_file)
                prev_confidence = val_confidence[0]
                save_status = "YES"

            # print some statistics
            print(f"Epoch {epoch} "
                f"train loss {epoch_running_loss:.10f} "
                f"accuracy {float(train_accuracy[0]) / float(train_accuracy[1]):.5f} "
                f"confidence {train_confidence[0] / train_confidence[1]:.5f} "
                f"magnitude {train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else -1:.5f} -- "
                f"val loss {float(val_loss[0]) / float(val_loss[1]):.10f} "
                f"accuracy {float(val_accuracy[0]) / float(val_accuracy[1]):.5f} "
                f"confidence {val_confidence[0] / val_confidence[1]:.5f} "
                f"magnitude {val_magnitude[0] / val_magnitude[1] if val_magnitude[1] else -1:.5f} -- "
                f"Saving Model {save_status}")


if __name__ == "__main__":
    t = Training_Mixed()
    t.train()