import Networks
import os
from vast import tools
import torch

def check_positive_tensor(input_tensor):
    for row in input_tensor:
        for element in row:
            if element < -0.01:
                return False
    return True

def load_network():
    network_file = f"LeNet/regular/SoftMax/SoftMax.model"
    if os.path.exists(network_file):
        net = Networks.__dict__["LeNet"](network_type="regular", num_classes = 10, bias = False)
        net.load_state_dict(torch.load(network_file))
        tools.device(net)
        return net
    else:
        return None

if __name__ == "__main__":
    net = load_network()
    print(check_positive_tensor(net.fc2.weight))