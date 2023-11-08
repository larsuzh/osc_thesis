import torch
import numpy
import argparse
import os
import Networks
import Metrics
import bisect

from vast import tools
from Training import Dataset
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

class Evaluate:
    def __init__(self):
        self.labels = {
            "SoftMax" : "Plain SoftMax",
            "EOS" : "Entropic Open-Set",
            "Objectosphere" : "Objectosphere",
            }
        self.args = self.command_line_options()
        self.set_gpu()
        self.val_set, self.test_set = self.load_data()
        self.results = {}
        self.trained_networks = {
            which: self.load_network(which) for which in self.args.approaches
        }


    def command_line_options(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument("--approaches", "-a", nargs="+", default=list(self.labels.keys()), choices=list(self.labels.keys()), help = "Select the approaches to evaluate; non-existing models will automatically be skipped")
        parser.add_argument("--arch", default='LeNet_pp', choices=['LeNet', 'LeNet_pp'])
        parser.add_argument("--net_type", default='regular', choices=['regular', 'single_fc', 'single_fc_poslin', 'double_fc', 'double_fc_poslin'])
        parser.add_argument("--dataset_root", "-d", default ="/tmp", help="Select the directory where datasets are stored.")
        parser.add_argument("--plot", "-p", default="Evaluate.pdf", help = "Where to write results into")
        parser.add_argument("--gpu", "-g", type=int, nargs="?", const=0, help="If selected, the experiment is run on GPU. You can also specify a GPU index")

        return parser.parse_args()
    

    def set_gpu(self):
        if torch.cuda.is_available():
            tools.set_device_gpu(self.args.gpu if self.args.gpu is not None else 0)
        else:
            print("Running in CPU mode, might be slow")
            tools.set_device_cpu()


    def load_data(self):
        return Dataset(self.args.dataset_root, "validation"), Dataset(self.args.dataset_root, "test")
    

    def load_network(self, approach):
        network_file = f"{self.args.arch}/{self.args.net_type}/{approach}/{approach}.model"
        if os.path.exists(network_file):
            net = Networks.__dict__[self.args.arch](network_type=self.args.net_type, num_classes = 1 if approach == "OOD" else 10, bias = approach == "OOD")
            net.load_state_dict(torch.load(network_file))
            tools.device(net)
            return net
        else:
            return None
    

    def extract(self, dataset, net):
        gt, logits = [], []
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

        with torch.no_grad():
            for (x, y) in data_loader:
                gt.extend(y.tolist())
                logs, _, _= net(tools.device(x))
                logits.extend(logs.tolist())

        return numpy.array(gt), numpy.array(logits)


    def writeOSCRCurve(self):
        pdf = PdfPages("Evaluation/" + self.args.plot)

        try:
            # plot with known unknowns (letters 1:13)
            pyplot.figure()
            for which, res in self.results.items():
                pyplot.semilogx(res[1], res[0], label=self.labels[which])
            pyplot.legend()
            pyplot.xlabel("False Positive Rate")
            pyplot.ylabel("Correct Classification Rate")
            pyplot.title("Negative Set")
            pyplot.ylim(0, 1)  # Set the y-axis limits from 0 to 1
            pyplot.tight_layout()
            pdf.savefig(bbox_inches='tight', pad_inches=0)

            # plot with unknown unknowns (letters 14:26)
            pyplot.figure()
            for which, res in self.results.items():
                pyplot.semilogx(res[2], res[0], label=self.labels[which])
            pyplot.legend()
            pyplot.xlabel("False Positive Rate")
            pyplot.ylabel("Correct Classification Rate")
            pyplot.title("Unknown Set")
            pyplot.ylim(0, 1)  # Set the y-axis limits from 0 to 1
            pyplot.tight_layout()
            pdf.savefig(bbox_inches='tight', pad_inches=0)

        finally:
            print("Wrote", self.args.plot)
            pdf.close()

    
    def calculate_results(self, which, positives, val, test, gt):
        ccr, fprv, fprt = [], [], []
        for tau in sorted(positives[range(len(gt)),gt]):
            # correct classification rate
            ccr.append(numpy.sum(numpy.logical_and(
                numpy.argmax(positives, axis=1) == gt,
                positives[range(len(gt)),gt] >= tau
            )) / len(positives))
            # false positive rate for validation and test set
            fprv.append(numpy.sum(numpy.max(val, axis=1) >= tau) / len(val))
            fprt.append(numpy.sum(numpy.max(test, axis=1) >= tau) / len(test))

        self.results[which] = (ccr, fprv, fprt)

    def find_nearest(self, fpr, t):
        fpr = numpy.asarray(fpr)
        return (numpy.abs(fpr - t)).argmin()

    def evaluate(self):
        for which, net in self.trained_networks.items():
            if net is None:
                continue
            print ("Evaluating", which)

            val_gt, val_predicted = self.extract(self.val_set, net)

            test_gt, test_predicted = self.extract(self.test_set, net)
            test_acc = Metrics.accuracy(torch.from_numpy(test_predicted), torch.from_numpy(test_gt))
            test_conf = Metrics.confidence(torch.from_numpy(test_predicted), torch.from_numpy(test_gt))
            print(f"test accuracy {float(test_acc[0]) / float(test_acc[1]):.5f} "
                  f"test confidence {float(test_conf[0]) / float(test_conf[1]):.5f} ")

            val_predicted = torch.nn.functional.softmax(torch.tensor(val_predicted), dim=1).detach().numpy()
            test_predicted  = torch.nn.functional.softmax(torch.tensor(test_predicted), dim=1).detach().numpy()

            positives = val_predicted[val_gt != -1]
            val = val_predicted[val_gt == -1]
            test = test_predicted[test_gt == -1]
            gt = val_gt[val_gt != -1]

            self.calculate_results(which, positives, val, test, gt)
            for t in [0.001, 0.01, 0.1, 1]:
                print("fpr: ", t, "ccr: ", self.results[which][0][self.find_nearest(self.results[which][2], t)])
        
        self.writeOSCRCurve()



if __name__ == '__main__':
    e = Evaluate()
    e.evaluate()