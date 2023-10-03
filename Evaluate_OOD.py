import torch
from Evaluate import Evaluate

class Evaluate_OOD(Evaluate):
    def __init__(self):
        super().__init__()
        self.ood_net = self.load_network("OOD")
    
    
    def evaluate(self):
        for which, net in self.trained_networks.items():
            if net is None:
                continue
            print ("Evaluating OOD", which)

            val_gt, val_predicted = self.extract(self.val_set, net)
            test_gt, test_predicted = self.extract(self.test_set, net)
            _, val_predicted_bc = self.extract(self.val_set, self.ood_net)
            _, test_predicted_bc = self.extract(self.test_set, self.ood_net)

            val_predicted = torch.nn.functional.softmax(torch.tensor(val_predicted), dim=1).detach().numpy()
            test_predicted  = torch.nn.functional.softmax(torch.tensor(test_predicted), dim=1).detach().numpy()
            val_predicted_bc = torch.sigmoid(torch.tensor(val_predicted_bc)).detach().numpy()
            test_predicted_bc = torch.sigmoid(torch.tensor(test_predicted_bc)).detach().numpy()

            
            positives = val_predicted[val_gt != -1] * val_predicted_bc[val_gt != -1]
            val = val_predicted[val_gt == -1] * val_predicted_bc[val_gt == -1]
            test = test_predicted[test_gt == -1] * val_predicted_bc[val_gt == -1]
            gt = val_gt[val_gt != -1]

            self.calculate_results(which, positives, val, test, gt)
        
        self.writeOSCRCurve()



if __name__ == '__main__':
    e = Evaluate_OOD()
    e.evaluate()
