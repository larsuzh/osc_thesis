import torch
import sys
import os
import numpy
import Networks
import matplotlib
matplotlib.rcParams["font.size"] = 18

from vast import tools
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--approach", choices=['SoftMax', 'OOD', 'EOS', 'Objectosphere'])
    parser.add_argument("--arch", default='LeNet_pp', choices=['LeNet', 'LeNet_pp'])
    parser.add_argument("--net_type", default='regular', choices=['regular', 'single_fc', 'single_fc_poslin', 'double_fc', 'double_fc_poslin'])
    parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--plot", "-p", default="Evaluate_Magnitudes.pdf", help = "Where to write results into")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()


def extract_features(dataset, net):
    gt, features = [], []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

    with torch.no_grad():
        for (x, y) in data_loader:
            gt.extend(y.tolist())
            feat = net(tools.device(x))[-2]
            features.extend(feat.tolist())

    return numpy.array(gt), numpy.array(features)



def load_network(arch, approach, net_type, mixed=False):
    if mixed:
        network_file = f"{arch}/mixed/{net_type}/{approach}/{approach}.model"
    else:
        network_file = f"{arch}/{net_type}/{approach}/{approach}.model"
    if os.path.exists(network_file):
        net = Networks.__dict__[arch](network_type=net_type, num_classes = 1 if approach == "OOD" else 10, bias = approach == "OOD", mixed=mixed)
        net.load_state_dict(torch.load(network_file))
        tools.device(net)
        return net
    else:
        return None


if __name__ == '__main__':

    args = command_line_options()
    if torch.cuda.is_available():
            tools.set_device_gpu(args.gpu if args.gpu is not None else 0)
    else:
        print("Running in CPU mode, might be slow")
        tools.set_device_cpu()

    net = load_network(args.arch, args.approach, args.net_type)

    from Training import Dataset

    val_set = Dataset(args.dataset_root, "validation")
    test_set = Dataset(args.dataset_root, "test")

    if net is None:
        sys.exit()
    
    val_gt, val_features = extract_features(val_set, net)
    test_gt, test_features = extract_features(test_set, net)

    pdf = PdfPages("Evaluation/" + args.plot)

    try:
        cmap = pyplot.get_cmap("tab20")
        pyplot.figure()
        x_values = [feature[0] for feature in val_features]
        y_values = [feature[1] for feature in val_features]
        if args.approach == "OOD":
            pyplot.scatter(x_values, y_values, c=[1 if val >= 0 else 0 for val in val_gt], cmap=cmap, marker='o', alpha=0.3)
        else:
            pyplot.scatter(x_values, y_values, c=test_gt, cmap=cmap, marker='o', alpha=0.3)
        pdf.savefig(bbox_inches='tight', pad_inches=0)
        pyplot.figure()
        x_values = [feature[0] for feature in test_features]
        y_values = [feature[1] for feature in test_features]
        if args.approach == "OOD":
            pyplot.scatter(x_values, y_values, c=[1 if val >= 0 else 0 for val in val_gt], cmap=cmap, marker='o', alpha=0.3)
        else:
            pyplot.scatter(x_values, y_values, c=test_gt, cmap=cmap, marker='o', alpha=0.3)
        pdf.savefig(bbox_inches='tight', pad_inches=0)

    finally:
        print("Wrote", args.plot)
        pdf.close()
