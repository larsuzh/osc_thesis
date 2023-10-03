import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim

from vast import tools, losses
import Metrics
import Networks
from Penalties import negative_penalty

import pathlib

class Training:
    def __init__(self):
        self.args = self.command_line_options()
        self.set_gpu()
        self.is_ood = self.args.approach == "OOD"

        self.results_dir = pathlib.Path(f"{self.args.arch}/{self.args.net_type}/{self.args.approach}")
        self.model_file = f"{self.results_dir}/{self.args.approach}.model"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.first_loss_func, self.second_loss_func, self.training_data, self.validation_data = list(zip(*self.get_loss_functions().items()))[-1]

        self.net = Networks.__dict__[self.args.arch](network_type=self.args.net_type, num_classes = 1 if self.is_ood else 10, bias = self.is_ood or self.args.net_type == "regular")
        self.net = tools.device(self.net)
        
        self.train_data_loader = torch.utils.data.DataLoader(
            self.training_data,
            batch_size=self.args.Batch_Size,
            shuffle=True,
            num_workers=5,
            pin_memory=True
        )
        self.val_data_loader = torch.utils.data.DataLoader(
            self.validation_data,
            batch_size=self.args.Batch_Size,
            pin_memory=True
        )

        if self.args.solver == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)
        elif self.args.solver == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
    

    def command_line_options(self):
        import argparse

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='This is the main training script for all MNIST experiments. \
                        Where applicable roman letters are used as negatives. \
                        During training model with best performance on validation set in the no_of_epochs is used.'
        )

        parser.add_argument("--approach", "-a", required=True, choices=['SoftMax', 'OOD', 'EOS', 'Objectosphere'])
        parser.add_argument("--arch", default='LeNet_pp', choices=['LeNet', 'LeNet_pp'])
        parser.add_argument("--net_type", default='regular', choices=['regular', 'single_fc', 'single_fc_poslin'])
        parser.add_argument('--second_loss_weight', "-w", help='Loss weight for Objectosphere loss', type=float, default=0.0001)
        parser.add_argument('--negative_penalty_weight', help='Penalty weight for negative weights in last layer', type=float, default=0)
        parser.add_argument('--Minimum_Knowns_Magnitude', "-m", help='Minimum Possible Magnitude for the Knowns', type=float,
                            default=50.)
        parser.add_argument("--solver", dest="solver", default='sgd',choices=['sgd','adam'])
        parser.add_argument("--lr", "-l", dest="lr", default=0.01, type=float)
        parser.add_argument('--batch_size', "-b", help='Batch_Size', action="store", dest="Batch_Size", type=int, default=128)
        parser.add_argument("--no_of_epochs", "-e", dest="no_of_epochs", type=int, default=70)
        parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
        parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

        return parser.parse_args()


    def set_gpu(self):
        if torch.cuda.is_available():
            tools.set_device_gpu(self.args.gpu if self.args.gpu is not None else 0)
        else:
            print("Running in CPU mode, might be slow")
            tools.set_device_cpu()

    
    def get_loss_functions(self):
        if self.args.approach == "SoftMax":
            return dict(
                        first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                        second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                        training_data = Dataset(self.args.dataset_root, include_unknown=False),
                        val_data = Dataset(self.args.dataset_root, which_set="val", include_unknown=False),
                    )
        elif self.args.approach == "EOS":
            return dict(
                        first_loss_func=losses.entropic_openset_loss(),
                        second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                        training_data=Dataset(self.args.dataset_root),
                        val_data = Dataset(self.args.dataset_root, which_set="val")
                    )
        elif self.args.approach == "Objectosphere":
            return dict(
                        first_loss_func=losses.entropic_openset_loss(),
                        second_loss_func=losses.objectoSphere_loss(self.args.Minimum_Knowns_Magnitude),
                        training_data=Dataset(self.args.dataset_root),
                        val_data = Dataset(self.args.dataset_root, which_set="val")
                    )
        elif self.args.approach == "OOD":
            return dict(
                        first_loss_func=nn.BCEWithLogitsLoss(),
                        second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                        training_data = Dataset(self.args.dataset_root, ood_approach=True),
                        val_data = Dataset(self.args.dataset_root, which_set="val", ood_approach=True),
                    )
    

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
                logits, features = self.net(x)
                if self.is_ood:
                    y = y.unsqueeze(1).float()
                loss = self.first_loss_func(logits, y) + self.args.second_loss_weight * self.second_loss_func(features, y) + self.args.negative_penalty_weight * negative_penalty(self.net.single_fc.weight)

                # metrics on training set
                train_accuracy += Metrics.accuracy(logits, y, is_ood = self.is_ood)
                train_confidence += Metrics.confidence(logits, y, is_ood = self.is_ood)
                if self.args.approach not in ("SoftMax", "OOD"):
                    train_magnitude += Metrics.sphere(features, y, self.args.Minimum_Knowns_Magnitude if self.args.approach in self.args.approach == "Objectosphere" else None)

                if self.is_ood:
                    loss_history.append(loss.item())
                else:
                    loss_history.extend(loss.tolist())
                loss.mean().backward()
                self.optimizer.step()

            # metrics on validation set
            with torch.no_grad():
                if self.is_ood:
                    val_loss_history = []
                else:
                    val_loss = torch.zeros(2, dtype=float)
                val_accuracy = torch.zeros(2, dtype=int)
                val_magnitude = torch.zeros(2, dtype=float)
                val_confidence = torch.zeros(2, dtype=float)
                self.net.eval()
                for x,y in self.val_data_loader:
                    x = tools.device(x)
                    y = tools.device(y)
                    outputs = self.net(x)
                    if self.is_ood:
                        y = y.unsqueeze(1).float()

                    loss = self.first_loss_func(outputs[0], y) + self.args.second_loss_weight * self.second_loss_func(outputs[1], y) + self.args.negative_penalty_weight * negative_penalty(self.net.single_fc.weight)
                    if self.is_ood:
                        val_loss_history.append(loss.item())
                    else:
                        val_loss += torch.tensor((torch.sum(loss), len(loss)))
                    val_accuracy += Metrics.accuracy(outputs[0], y, is_ood = self.is_ood)
                    val_confidence += Metrics.confidence(outputs[0], y, is_ood = self.is_ood)
                    if self.args.approach not in ("SoftMax", "OOD"):
                        val_magnitude += Metrics.sphere(outputs[1], y, self.args.Minimum_Knowns_Magnitude if self.args.approach == "Objectosphere" else None)

            # log statistics
            epoch_running_loss = torch.mean(torch.tensor(loss_history))
            if self.is_ood:
                epoch_running_val_loss = torch.mean(torch.tensor(val_loss_history))

            # save network based on confidence metric of validation set
            save_status = "NO"
            if prev_confidence is None or (val_confidence[0] > prev_confidence):
                torch.save(self.net.state_dict(), self.model_file)
                prev_confidence = val_confidence[0]
                save_status = "YES"

            # print some statistics
            val_loss_to_print = epoch_running_val_loss if self.is_ood else float(val_loss[0]) / float(val_loss[1])
            print(f"Epoch {epoch} "
                f"train loss {epoch_running_loss:.10f} "
                f"accuracy {float(train_accuracy[0]) / float(train_accuracy[1]):.5f} "
                f"confidence {train_confidence[0] / train_confidence[1]:.5f} "
                f"magnitude {train_magnitude[0] / train_magnitude[1] if train_magnitude[1] else -1:.5f} -- "
                f"val loss {val_loss_to_print:.10f} "
                f"accuracy {float(val_accuracy[0]) / float(val_accuracy[1]):.5f} "
                f"confidence {val_confidence[0] / val_confidence[1]:.5f} "
                f"magnitude {val_magnitude[0] / val_magnitude[1] if val_magnitude[1] else -1:.5f} -- "
                f"Saving Model {save_status}")


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_root, which_set="train", include_unknown=True, ood_approach=False):
        self.mnist = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split="mnist",
            transform=transforms.Compose([transforms.ToTensor(), self.transpose])
        )
        self.letters = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split='letters',
            transform=transforms.Compose([transforms.ToTensor(), self.transpose])
        )
        self.which_set = which_set
        targets = list() if not include_unknown else [1,2,3,4,5,6,8,10,11,13,14] if which_set != "test" else [16,17,18,19,20,21,22,23,24,25,26]
        self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in targets]
        self.ood_approach = ood_approach

    def transpose(self, x):
        """Used for correcting rotation of EMNIST Letters"""
        return x.transpose(2,1)

    def __getitem__(self, index):
        if index < len(self.mnist):
            return self.mnist[index][0], 1 if self.ood_approach else self.mnist[index][1]
        else:
            return self.letters[self.letter_indexes[index - len(self.mnist)]][0], 0 if self.ood_approach else -1

    def __len__(self):
        return len(self.mnist) + len(self.letter_indexes)
    



if __name__ == "__main__":
    t = Training()
    t.train()