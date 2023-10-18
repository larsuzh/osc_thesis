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


def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the main training script for all MNIST experiments. \
                    Where applicable roman letters are used as negatives. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approach", "-a", required=True, choices=['SoftMax', 'OOD', 'EOS', 'Objectosphere'])
    parser.add_argument("--arch", default='LeNet_pp', choices=['LeNet', 'LeNet_pp'])
    parser.add_argument("--net_type", default='regular', choices=['regular', 'single_fc', 'single_fc_poslin', 'double_fc', 'double_fc_poslin'])
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

def transpose(x):
    """Used for correcting rotation of EMNIST Letters"""
    return x.transpose(2,1)

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_root, which_set="train", include_unknown=True, ood_approach=False):
        self.mnist = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split="mnist",
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.letters = torchvision.datasets.EMNIST(
            root=dataset_root,
            train=which_set == "train",
            download=True,
            split='letters',
            transform=transforms.Compose([transforms.ToTensor(), transpose])
        )
        self.which_set = which_set
        targets = list() if not include_unknown else [1,2,3,4,5,6,8,10,11,13,14] if which_set != "test" else [16,17,18,19,20,21,22,23,24,25,26]
        self.letter_indexes = [i for i, t in enumerate(self.letters.targets) if t in targets]
        self.ood_approach = ood_approach

    def __getitem__(self, index):
        if index < len(self.mnist):
            return self.mnist[index][0], 1 if self.ood_approach else self.mnist[index][1]
        else:
            return self.letters[self.letter_indexes[index - len(self.mnist)]][0], 0 if self.ood_approach else -1

    def __len__(self):
        return len(self.mnist) + len(self.letter_indexes)
    
def get_loss_functions(args):
    if args.approach == "SoftMax":
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data = Dataset(args.dataset_root, include_unknown=False),
                    val_data = Dataset(args.dataset_root, which_set="val", include_unknown=False),
                )
    elif args.approach == "EOS":
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data=Dataset(args.dataset_root),
                    val_data = Dataset(args.dataset_root, which_set="val")
                )
    elif args.approach == "Objectosphere":
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=losses.objectoSphere_loss(args.Minimum_Knowns_Magnitude),
                    training_data=Dataset(args.dataset_root),
                    val_data = Dataset(args.dataset_root, which_set="val")
                )
    elif args.approach == "OOD":
        return dict(
                    first_loss_func=nn.BCEWithLogitsLoss(),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    training_data = Dataset(args.dataset_root, ood_approach=True),
                    val_data = Dataset(args.dataset_root, which_set="val", ood_approach=True),
                )

def train(args):
    torch.manual_seed(0)
    is_ood = args.approach == "OOD"

    first_loss_func,second_loss_func,training_data,validation_data = list(zip(*get_loss_functions(args).items()))[-1]

    results_dir = pathlib.Path(f"{args.arch}/{args.net_type}/{args.approach}")
    model_file = f"{results_dir}/{args.approach}.model"
    results_dir.mkdir(parents=True, exist_ok=True)

    net = Networks.__dict__[args.arch](network_type=args.net_type, num_classes = 1 if is_ood else 10, bias= is_ood)
    net = tools.device(net)
    
    train_data_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=args.Batch_Size,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.Batch_Size,
        pin_memory=True
    )

    if args.solver == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.solver == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # train network
    prev_confidence = None
    for epoch in range(1, args.no_of_epochs + 1, 1):
        loss_history = []
        train_accuracy = torch.zeros(2, dtype=int)
        train_magnitude = torch.zeros(2, dtype=float)
        train_confidence = torch.zeros(2, dtype=float)
        net.train()
        for x, y in train_data_loader:
            x = tools.device(x)
            y = tools.device(y)
            optimizer.zero_grad()
            logits, features = net(x)
            if is_ood:
                y = y.unsqueeze(1).float()
            loss = first_loss_func(logits, y) + args.second_loss_weight * second_loss_func(features, y)
            if args.net_type == 'single_fc':
                loss += args.negative_penalty_weight * negative_penalty(net.single_fc.weight)
            elif args.net_type == 'double_fc':
                loss += args.negative_penalty_weight * negative_penalty(net.fc2.weight)


            # metrics on training set
            train_accuracy += Metrics.accuracy(logits, y, is_ood = is_ood)
            train_confidence += Metrics.confidence(logits, y, is_ood = is_ood)
            if args.approach not in ("SoftMax", "OOD"):
                train_magnitude += Metrics.sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach in args.approach == "Objectosphere" else None)

            if is_ood:
                loss_history.append(loss.item())
            else:
                loss_history.extend(loss.tolist())
            loss.mean().backward()
            optimizer.step()

        # metrics on validation set
        with torch.no_grad():
            if is_ood:
                val_loss_history = []
            else:
                val_loss = torch.zeros(2, dtype=float)
            val_accuracy = torch.zeros(2, dtype=int)
            val_magnitude = torch.zeros(2, dtype=float)
            val_confidence = torch.zeros(2, dtype=float)
            net.eval()
            for x,y in val_data_loader:
                x = tools.device(x)
                y = tools.device(y)
                outputs = net(x)
                if is_ood:
                    y = y.unsqueeze(1).float()

                loss = first_loss_func(outputs[0], y) + args.second_loss_weight * second_loss_func(outputs[1], y)
                if args.net_type == 'single_fc':
                    loss += args.negative_penalty_weight * negative_penalty(net.single_fc.weight)
                elif args.net_type == 'double_fc':
                    loss += args.negative_penalty_weight * negative_penalty(net.fc2.weight)

                if is_ood:
                    val_loss_history.append(loss.item())
                else:
                    val_loss += torch.tensor((torch.sum(loss), len(loss)))
                val_accuracy += Metrics.accuracy(outputs[0], y, is_ood = is_ood)
                val_confidence += Metrics.confidence(outputs[0], y, is_ood = is_ood)
                if args.approach not in ("SoftMax", "OOD"):
                    val_magnitude += Metrics.sphere(outputs[1], y, args.Minimum_Knowns_Magnitude if args.approach == "Objectosphere" else None)

        # log statistics
        epoch_running_loss = torch.mean(torch.tensor(loss_history))
        if is_ood:
            epoch_running_val_loss = torch.mean(torch.tensor(val_loss_history))

        # save network based on confidence metric of validation set
        save_status = "NO"
        if prev_confidence is None or (val_confidence[0] > prev_confidence):
            torch.save(net.state_dict(), model_file)
            prev_confidence = val_confidence[0]
            save_status = "YES"
            i = 0
        i += 1
        if i > 11:
            return

        # print some statistics
        val_loss_to_print = epoch_running_val_loss if is_ood else float(val_loss[0]) / float(val_loss[1])
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


if __name__ == "__main__":

    args = command_line_options()
    if torch.cuda.is_available():
            tools.set_device_gpu(args.gpu if args.gpu is not None else 0)
    else:
        print("Running in CPU mode, training might be slow")
        tools.set_device_cpu()
    
    train(args)