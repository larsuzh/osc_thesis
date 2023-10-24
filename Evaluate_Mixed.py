import torch
import numpy
import os
import Networks

from Evaluate import Evaluate
from vast import tools

class Evaluate_Mixed(Evaluate):
    def __init__(self):
        super().__init__()


    def load_network(self, approach):
        network_file = f"{self.args.arch}/mixed/{self.args.net_type}/{approach}/{approach}.model"
        if os.path.exists(network_file):
            net = Networks.__dict__[self.args.arch](network_type = self.args.net_type, mixed = True)
            net.load_state_dict(torch.load(network_file))
            tools.device(net)
            return net
        else:
            return None
        
    
    def extract(self, dataset, net):
        gt, logits, logits_2 = [], [], []
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)

        with torch.no_grad():
            for (x, y) in data_loader:
                gt.extend(y.tolist())
                logs, logs_2, _ = net(tools.device(x))
                logits.extend(logs.tolist())
                logits_2.extend(logs_2.tolist())
        return numpy.array(gt), numpy.array(logits), numpy.array(logits_2)
    

    def evaluate(self):
        for which, net in self.trained_networks.items():
            if net is None:
                continue
            print ("Evaluating mixed", which)

            val_gt, val_predicted, val_predicted_bc = self.extract(self.val_set, net)
            test_gt, test_predicted, test_predicted_bc = self.extract(self.test_set, net)

            
            val_predicted = torch.nn.functional.softmax(torch.tensor(val_predicted), dim=1).detach().numpy()
            test_predicted  = torch.nn.functional.softmax(torch.tensor(test_predicted), dim=1).detach().numpy()
            val_predicted_bc = torch.sigmoid(torch.tensor(val_predicted_bc)).detach().numpy()
            test_predicted_bc = torch.sigmoid(torch.tensor(test_predicted_bc)).detach().numpy()

            positives = val_predicted[val_gt != -1] * val_predicted_bc[val_gt != -1]
            val = val_predicted[val_gt == -1] * val_predicted_bc[val_gt == -1]
            test = test_predicted[test_gt == -1] * val_predicted_bc[val_gt == -1]
            gt = val_gt[val_gt != -1]

            self.calculate_results(which, positives, val, test, gt)
            for t in [0.001, 0.01, 0.1, 1]:
                print("fpr: ", t, "ccr: ", self.results[which][0][self.find_nearest(self.results[which][2], t)])
        
        self.writeOSCRCurve()


if __name__ == '__main__':
    e = Evaluate_Mixed()
    e.evaluate()
