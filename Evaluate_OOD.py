import torch
import numpy

from vast import tools
from Evaluate_Util import extract, load_network, writeOSCRCurve

labels={
  "SoftMax" : "Plain SoftMax",
  "EOS" : "Entropic Open-Set",
  "Objectosphere" : "Objectosphere"
}


def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--approaches", "-a", nargs="+", default=list(labels.keys()), choices=list(labels.keys()), help = "Select the approaches to evaluate; non-existing models will automatically be skipped")
    parser.add_argument("--arch", default='LeNet_pp', choices=['LeNet', 'LeNet_pp'])
    parser.add_argument("--net_type", default='regular', choices=['regular', 'single_fc', 'single_fc_poslin'])
    parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--plot", "-p", default="Evaluate.pdf", help = "Where to write results into")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()

if __name__ == '__main__':

    args = command_line_options()
    if torch.cuda.is_available():
            tools.set_device_gpu(args.gpu if args.gpu is not None else 0)
    else:
        print("Running in CPU mode, might be slow")
        tools.set_device_cpu()

    from Training import Dataset as Dataset

    val_set = Dataset(args.dataset_root, "validation")
    test_set = Dataset(args.dataset_root, "test")

    results = {}
    trained_networks = {
        which: load_network(args.arch, which, args.net_type) for which in args.approaches
    }
    ood_net = load_network(args.arch, "OOD", args.net_type)

    results = {}
    for which, net in trained_networks.items():
        if net is None:
            continue
        print ("Evaluating OOD", which)

        val_gt, val_predicted_sm = extract(val_set, net)
        test_gt, test_predicted_sm = extract(test_set, net)
        _, val_predicted_bc = extract(val_set, ood_net)
        _, test_predicted_bc = extract(test_set, ood_net)

        # compute probabilities
        val_predicted_sm = torch.nn.functional.softmax(torch.tensor(val_predicted_sm), dim=1).detach().numpy()
        test_predicted_sm  = torch.nn.functional.softmax(torch.tensor(test_predicted_sm), dim=1).detach().numpy()
        val_predicted_bc = torch.sigmoid(torch.tensor(val_predicted_bc)).detach().numpy()
        test_predicted_bc = torch.sigmoid(torch.tensor(test_predicted_bc)).detach().numpy()

        # vary thresholds
        ccr, fprv, fprt = [], [], []
        positives = val_predicted_sm[val_gt != -1] * val_predicted_bc[val_gt != -1]
        val = val_predicted_sm[val_gt == -1] * val_predicted_bc[val_gt == -1]
        test = test_predicted_sm[test_gt == -1] * val_predicted_bc[val_gt == -1]
        gt = val_gt[val_gt != -1]
        for tau in sorted(positives[range(len(gt)),gt]):
            # correct classification rate
            ccr.append(numpy.sum(numpy.logical_and(
                numpy.argmax(positives, axis=1) == gt,
                positives[range(len(gt)),gt] >= tau
            )) / len(positives))
            # false positive rate for validation and test set
            fprv.append(numpy.sum(numpy.max(val, axis=1) >= tau) / len(val))
            fprt.append(numpy.sum(numpy.max(test, axis=1) >= tau) / len(test))

        results[which] = (ccr, fprv, fprt)

    writeOSCRCurve(results, labels, args.plot)
