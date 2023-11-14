import torch
import numpy
import Networks
import Metrics
from Evaluate import Evaluate
from vast import tools

class Evaluate_OOD(Evaluate):
    def __init__(self):
        super().__init__()
        network_file = f"LeNet/single_fc/OOD/OOD.model"
        self.ood_net = Networks.__dict__["LeNet"](network_type="single_fc", num_classes = 1 , bias = True)
        self.ood_net.load_state_dict(torch.load(network_file))
        tools.device(self.ood_net)
    
    
    def evaluate(self):
        for which, net in self.trained_networks.items():
            if net is None:
                continue
            print ("Evaluating OOD", which)

            val_gt, val_predicted = self.extract(self.val_set, net)
            test_gt, test_predicted = self.extract(self.test_set, net)
            _, val_predicted_bc = self.extract(self.val_set, self.ood_net)
            _, test_predicted_bc = self.extract(self.test_set, self.ood_net)
            test_gt_bc = numpy.array([1 if gt >= 0 else 0 for gt in test_gt])
            test_acc = Metrics.accuracy(torch.from_numpy(test_predicted_bc), torch.from_numpy(test_gt_bc), is_ood = True)
            test_conf = Metrics.confidence(torch.from_numpy(test_predicted_bc), torch.from_numpy(test_gt_bc), is_ood = True)
            print(f"test accuracy {float(test_acc[0]) / float(test_acc[1]):.5f} "
                  f"test confidence {float(test_conf[0]) / float(test_conf[1]):.5f} ")

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
    e = Evaluate_OOD()
    e.evaluate()
