import torch
import numpy
import os
import Networks
import matplotlib
matplotlib.rcParams["font.size"] = 18

from vast import tools
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

def load_network(arch, approach, net_type):
    network_file = f"{arch}/{net_type}/{approach}/{approach}.model"
    if os.path.exists(network_file):
        net = Networks.__dict__[arch](network_type=net_type, num_classes = 1 if approach == "OOD" else 10, bias = approach == "OOD" or net_type == "regular")
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


def extract(dataset, net):
    gt, logits = [], []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

    with torch.no_grad():
        for (x, y) in data_loader:
            gt.extend(y.tolist())
            logs, feat = net(tools.device(x))
            logits.extend(logs.tolist())

    return numpy.array(gt), numpy.array(logits)

def writeOSCRCurve(results, labels, plot):
    pdf = PdfPages("Evaluation/" + plot)

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
        print("Wrote", plot)
        pdf.close()

