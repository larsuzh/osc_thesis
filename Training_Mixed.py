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

from Training import Dataset, command_line_options

import pathlib
    
def get_loss_functions(args):
    if args.approach == "SoftMax":
        return dict(
                    first_loss_func=nn.CrossEntropyLoss(reduction='none'),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    third_loss_func=nn.BCEWithLogitsLoss(),
                    training_data = Dataset(args.dataset_root),
                    val_data = Dataset(args.dataset_root, which_set="val"),
                )
    elif args.approach == "EOS":
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=lambda arg1, arg2, arg3=None, arg4=None: torch.tensor(0.),
                    third_loss_func=nn.BCEWithLogitsLoss(),
                    training_data=Dataset(args.dataset_root),
                    val_data = Dataset(args.dataset_root, which_set="val")
                )
    elif args.approach == "Objectosphere":
        return dict(
                    first_loss_func=losses.entropic_openset_loss(),
                    second_loss_func=losses.objectoSphere_loss(args.Minimum_Knowns_Magnitude),
                    third_loss_func=nn.BCEWithLogitsLoss(),
                    training_data=Dataset(args.dataset_root),
                    val_data = Dataset(args.dataset_root, which_set="val")
                )

def train(args):
    torch.manual_seed(0)

    first_loss_func,second_loss_func,third_loss_func,training_data,validation_data = list(zip(*get_loss_functions(args).items()))[-1]

    results_dir = pathlib.Path(f"{args.arch}/mixed/{args.net_type}/{args.approach}")
    model_file = f"{results_dir}/{args.approach}.model"
    results_dir.mkdir(parents=True, exist_ok=True)

    net = Networks.__dict__[args.arch](network_type=args.net_type, bias=args.net_type == "regular", mixed=True)
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
            logits, logits_2 ,features = net(x)


            mask = y >= -1
            if args.approach == "SoftMax":
                mask = y >= 0
                loss = first_loss_func(logits[mask], y[mask])
            else:
                loss = first_loss_func(logits, y) + args.second_loss_weight * second_loss_func(features, y) + args.negative_penalty_weight * negative_penalty(net.single_fc.weight)
            y_2 = torch.where(y < 0, torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            y_2 = y_2.unsqueeze(1).float()
            loss += third_loss_func(logits_2, y_2)

            # metrics on training set
            train_accuracy += Metrics.accuracy(logits[mask], y[mask])
            train_confidence += Metrics.confidence(logits[mask], y[mask])
            if args.approach not in ("SoftMax", "OOD"):
                train_magnitude += Metrics.sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach in args.approach == "Objectosphere" else None)

            loss_history.extend(loss.tolist())
            loss.mean().backward()
            optimizer.step()

        # metrics on validation set
        with torch.no_grad():
            val_loss = torch.zeros(2, dtype=float)
            val_accuracy = torch.zeros(2, dtype=int)
            val_magnitude = torch.zeros(2, dtype=float)
            val_confidence = torch.zeros(2, dtype=float)
            net.eval()
            for x,y in val_data_loader:
                x = tools.device(x)
                y = tools.device(y)
                logits, logits_2 ,features = net(x)

                mask = y >= -1
                if args.approach == "SoftMax":
                    mask = y >= 0
                    loss = first_loss_func(logits[mask], y[mask])
                else:
                    loss = first_loss_func(logits, y) + args.second_loss_weight * second_loss_func(features, y) + args.negative_penalty_weight * negative_penalty(net.single_fc.weight)
                
                y_2 = torch.where(y < 0, torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                y_2 = y_2.unsqueeze(1).float()
                loss += third_loss_func(logits_2, y_2)

                val_loss += torch.tensor((torch.sum(loss), len(loss)))
                val_accuracy += Metrics.accuracy(logits[mask], y[mask])
                val_confidence += Metrics.confidence(logits[mask], y[mask])
                if args.approach not in ("SoftMax", "OOD"):
                    val_magnitude += Metrics.sphere(features, y, args.Minimum_Knowns_Magnitude if args.approach == "Objectosphere" else None)

        # log statistics
        epoch_running_loss = torch.mean(torch.tensor(loss_history))

        # save network based on confidence metric of validation set
        save_status = "NO"
        if prev_confidence is None or (val_confidence[0] > prev_confidence):
            torch.save(net.state_dict(), model_file)
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

    args = command_line_options()
    if torch.cuda.is_available():
            tools.set_device_gpu(args.gpu if args.gpu is not None else 0)
    else:
        print("Running in CPU mode, training might be slow")
        tools.set_device_cpu()
    
    train(args)