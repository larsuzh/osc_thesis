import torch
import numpy
import os
import sys
from vast import tools
import networks

import matplotlib
matplotlib.rcParams["font.size"] = 18
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--approach", choices=['SoftMax', 'OOD', 'EOS', 'Objectosphere'])
    parser.add_argument("--arch", default='LeNet_pp', choices=['LeNet', 'LeNet_pp'])
    parser.add_argument("--net_type", default='regular', choices=['regular', 'single_fc', 'single_fc_poslin'])
    parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--plot", "-p", default="Evaluate_Magnitudes.pdf", help = "Where to write results into")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()


def load_network(args):
    network_file = f"{args.arch}/{args.net_type}/{args.approach}/{args.approach}.model"
    if os.path.exists(network_file):
        net = networks.__dict__[args.arch](network_type=args.net_type, num_classes = 1 if args.approach == "OOD" else 10, bias = args.approach == "OOD" or args.net_type == "regular")
        net.load_state_dict(torch.load(network_file))
        tools.device(net)
        return net
    else:
        return None

def extract_features(dataset, net):
    gt, features = [], []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

    with torch.no_grad():
        for (x, y) in data_loader:
            gt.extend(y.tolist())
            logs, feat = net(tools.device(x))
            features.extend(feat.tolist())

    return numpy.array(gt), numpy.array(features)


if __name__ == '__main__':

    args = command_line_options()
    if torch.cuda.is_available():
            tools.set_device_gpu(args.gpu if args.gpu is not None else 0)
    else:
        print("Running in CPU mode, might be slow")
        tools.set_device_cpu()

    net = load_network(args)

    from Training import Dataset

    val_set = Dataset(args.dataset_root, "validation")
    test_set = Dataset(args.dataset_root, "test")

    if net is None:
        sys.exit()
    
    val_gt, val_features = extract_features(val_set, net)
    val_positives = val_features[val_gt >= 0]
    val_negatives = val_features[val_gt < 0]
    val_magnitudes_positives = [round(sum(feature ** 2 for feature in features)/len(features), 3) for features in val_positives] # FIXME try using mean because sublist might be featuremap and not vector
    val_magnitudes_negatives = [round(sum(feature ** 2 for feature in features)/len(features), 3) for features in val_negatives]

    test_gt, test_features = extract_features(test_set, net)
    test_positives = test_features[test_gt >= 0]
    test_negatives = test_features[test_gt < 0]
    test_magnitudes_positives = [round(sum(feature ** 2 for feature in features)/len(features), 3) for features in test_positives]
    test_magnitudes_negatives = [round(sum(feature ** 2 for feature in features)/len(features), 3) for features in test_negatives]

    pdf = PdfPages("Evaluation/" + args.plot)

    try:
        pyplot.figure()
        values_positive = sorted(list(set(val_magnitudes_positives)))
        frequencies_positive = [val_magnitudes_positives.count(value)/len(val_magnitudes_positives) for value in values_positive]

        pyplot.plot(values_positive, frequencies_positive, color='red', linestyle='-')

        values_negative = sorted(list(set(val_magnitudes_negatives)))
        frequencies_negative = [val_magnitudes_negatives.count(value)/len(val_magnitudes_negatives) for value in values_negative]
        pyplot.plot(values_negative, frequencies_negative, color='blue', linestyle='-')

        pyplot.xlabel('values')
        pyplot.ylabel('freq')
        pyplot.title('validation set')
        pyplot.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)

        pyplot.figure()
        values_positive = sorted(list(set(test_magnitudes_positives)))
        frequencies_positive = [test_magnitudes_positives.count(value)/len(test_magnitudes_positives) for value in values_positive]

        pyplot.plot(values_positive, frequencies_positive, color='red', linestyle='-')

        values_negative = sorted(list(set(test_magnitudes_negatives)))
        frequencies_negative = [test_magnitudes_negatives.count(value)/len(test_magnitudes_negatives) for value in values_negative]
        pyplot.plot(values_negative, frequencies_negative, color='blue', linestyle='-')

        pyplot.xlabel('values')
        pyplot.ylabel('freq')
        pyplot.title('test set')
        pyplot.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)

    finally:
        print("Wrote", args.plot)
        pdf.close()
