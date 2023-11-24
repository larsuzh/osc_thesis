import torch
import sys
import os
import json
import numpy
import Networks
import matplotlib
matplotlib.rcParams["font.size"] = 18

from vast import tools
from scipy import stats
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--approach", choices=['SoftMax', 'OOD', 'EOS', 'Objectosphere'])
    parser.add_argument("--arch", default='LeNet', choices=['LeNet', 'LeNet_pp'])
    parser.add_argument("--mixed", default='False', choices=['False', 'True'])
    parser.add_argument("--net_type", default='regular', choices=['regular', 'regular_single', 'single_fc', 'single_fc_poslin', 'double_fc', 'double_fc_poslin'])
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
            feat = net(tools.device(x))[-1]
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

    net = load_network(args.arch, args.approach, args.net_type, mixed = args.mixed == "True")

    from Training import Dataset

    val_set = Dataset(args.dataset_root, "validation")
    test_set = Dataset(args.dataset_root, "test")

    if net is None:
        sys.exit()
    
    val_gt, val_features = extract_features(val_set, net)
    val_positives = numpy.array(val_features[val_gt >= 0])
    val_negatives = numpy.array(val_features[val_gt < 0])
    val_magnitudes_positives = numpy.round(numpy.sum(val_positives**2, axis=1) / val_positives.shape[1], 3)
    val_magnitudes_negatives = numpy.round(numpy.sum(val_negatives**2, axis=1) / val_negatives.shape[1], 3)
    print("val positives \n"
          "Mean: ", numpy.mean(val_magnitudes_positives), " Median: ", numpy.median(val_magnitudes_positives), " SD: ", numpy.std(val_magnitudes_positives))
    print("val negatives \n"
          "Mean: ", numpy.mean(val_magnitudes_negatives), " Median: ", numpy.median(val_magnitudes_negatives), " SD: ", numpy.std(val_magnitudes_negatives))

    test_gt, test_features = extract_features(test_set, net)
    test_positives = numpy.array(test_features[test_gt >= 0])
    test_negatives = numpy.array(test_features[test_gt < 0])
    test_magnitudes_positives = numpy.round(numpy.sum(test_positives**2, axis=1) / test_positives.shape[1] , 3)
    test_magnitudes_negatives = numpy.round(numpy.sum(test_negatives**2, axis=1) / test_negatives.shape[1] , 3)
    print("test positives \n"
          "Mean: ", numpy.mean(test_magnitudes_positives), " Median: ", numpy.median(test_magnitudes_positives), " SD: ", numpy.std(test_magnitudes_positives))
    print("test negatives \n"
          "Mean: ", numpy.mean(test_magnitudes_negatives), " Median: ", numpy.median(test_magnitudes_negatives), " SD: ", numpy.std(test_magnitudes_negatives))
    print(stats.ttest_ind(val_magnitudes_negatives, val_magnitudes_positives, alternative='less'))
    print(stats.ttest_ind(test_magnitudes_negatives, val_magnitudes_positives, alternative='less'))

    with open("mixed_singlefc_objecto.json", 'w') as json_file:
        json.dump(val_magnitudes_positives.tolist(), json_file)
        json.dump(test_magnitudes_negatives.tolist(), json_file)

    pdf = PdfPages("Evaluation/" + args.plot)

    try:
        pyplot.figure()
        values_positive = sorted(list(set(val_magnitudes_positives)))
        frequencies_positive = [numpy.sum(val_magnitudes_positives == value)/len(val_magnitudes_positives) for value in values_positive]

        pyplot.plot(values_positive, frequencies_positive, color='red', linestyle='-')

        values_negative = sorted(list(set(val_magnitudes_negatives)))
        frequencies_negative = [numpy.sum(val_magnitudes_negatives == value)/len(val_magnitudes_negatives) for value in values_negative]
        max_frequency_positive = max(frequencies_positive)
        max_frequency_negative = max(frequencies_negative)
        pyplot.plot(values_negative, [freq * (max_frequency_positive / max_frequency_negative) for freq in frequencies_negative], color='blue', linestyle='-')
        pyplot.xscale('log')
        pyplot.yscale('linear')

        pyplot.xlabel('values')
        pyplot.ylabel('freq')
        pyplot.yticks([])
        pyplot.title('validation set')
        pyplot.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)

        pyplot.figure()
        values_positive = sorted(list(set(test_magnitudes_positives)))
        frequencies_positive = [numpy.sum(test_magnitudes_positives == value)/len(test_magnitudes_positives) for value in values_positive]

        pyplot.plot(values_positive, frequencies_positive, color='red', linestyle='-')

        values_negative = sorted(list(set(test_magnitudes_negatives)))
        frequencies_negative = [numpy.sum(test_magnitudes_negatives == value)/len(test_magnitudes_negatives) for value in values_negative]
        max_frequency_positive = max(frequencies_positive)
        max_frequency_negative = max(frequencies_negative)
        pyplot.plot(values_negative, [freq * (max_frequency_positive / max_frequency_negative) for freq in frequencies_negative], color='blue', linestyle='-')
        pyplot.xscale('log')
        pyplot.yscale('linear')

        pyplot.xlabel('values')
        pyplot.ylabel('freq')
        pyplot.yticks([])
        pyplot.title('test set')
        pyplot.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)

    finally:
        print("Wrote", args.plot)
        pdf.close()
