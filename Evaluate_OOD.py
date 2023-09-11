import torch
import numpy
import os
from vast import tools
import networks

import matplotlib
matplotlib.rcParams["font.size"] = 18
from matplotlib import pyplot, patches
from matplotlib.backends.backend_pdf import PdfPages

labels={
  "SoftMax" : "Plain SoftMax",
  "Garbage" : "Garbage Class",
  "EOS" : "Entropic Open-Set",
  "Objectosphere" : "Objectosphere"
}


def command_line_options():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This is the evaluation script for all MNIST experiments. \
                    Where applicable roman letters are used as Known Unknowns. \
                    During training model with best performance on validation set in the no_of_epochs is used.'
    )

    parser.add_argument("--approaches", "-a", nargs="+", default=list(labels.keys()), choices=list(labels.keys()), help = "Select the approaches to evaluate; non-existing models will automatically be skipped")
    parser.add_argument("--arch", default='LeNet_pp', choices=['LeNet', 'LeNet_pp'])
    parser.add_argument("--net_type", default='regular', choices=['regular', 'single_fc', 'single_fc_poslin'])
    parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
    parser.add_argument("--plot", "-p", default="Evaluate.pdf", help = "Where to write results into")
    parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

    return parser.parse_args()


def load_network(args, which, net_type):
    network_file = f"{args.arch}/{args.net_type}/{which}/{which}.model"
    if os.path.exists(network_file):
        net = networks.__dict__[args.arch](network_type=args.net_type, bias = False)
        net.load_state_dict(torch.load(network_file))
        tools.device(net)
        return net
    else:
        return None

def extract(dataset, net):
    gt, logits = [], []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

    with torch.no_grad():
        for (x, y) in data_loader:
            gt.extend(y.tolist())
            logs, feat = net(tools.device(x))
            logits.extend(logs.tolist())

    return numpy.array(gt), numpy.array(logits)


if __name__ == '__main__':

    args = command_line_options()
    if torch.cuda.is_available():
            tools.set_device_gpu(args.gpu if args.gpu is not None else 0)
    else:
        print("Running in CPU mode, might be slow")
        tools.set_device_cpu()

    from Training import Dataset as Dataset

    # negative set
    val_set = Dataset(args.dataset_root, "validation")
    # unknown set
    test_set = Dataset(args.dataset_root, "test")

    results = {}
    networks = {
        which: load_network(args, which, args.net_type) for which in args.approaches
    }
    # FIXME
    # ood_net = load_network(args, args.approaches[0], args.net_type)
    print ("Evaluating OOD method")
    for which, net in networks.items():
        if net is None:
            continue
        val_gt, val_predicted_sm = extract(val_set, net)
        test_gt, test_predicted_sm = extract(test_set, net)
        _, val_predicted_bc = extract(val_set, net)
        _, test_predicted_bc = extract(test_set, net)

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

    pdf = PdfPages(args.plot)

    try:
        # plot with known unknowns (letters 1:13)
        pyplot.figure()
        for which, res in results.items():
            pyplot.semilogx(res[1], res[0], label=labels[which])
        pyplot.legend()
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("Correct Classification Rate")
        pyplot.title("Negative Set")
        pyplot.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)

        # plot with unknown unknowns (letters 14:26)
        pyplot.figure()
        for which, res in results.items():
            pyplot.semilogx(res[2], res[0], label=labels[which])
        pyplot.legend()
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("Correct Classification Rate")
        pyplot.title("Unknown Set")
        pyplot.tight_layout()
        pdf.savefig(bbox_inches='tight', pad_inches=0)

    finally:
        print("Wrote", args.plot)
        pdf.close()
