import torch.nn as nn
import torch

class LeNet_pp(nn.Module):
    def __init__(self, network_type="regular", num_classes=10, bias=False, mixed=False):
        super(LeNet_pp, self).__init__()
        self.conv1_1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=self.conv1_1.out_channels,
            out_channels=32,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.batch_norm1 = nn.BatchNorm2d(self.conv1_2.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2_1 = nn.Conv2d(
            in_channels=self.conv1_2.out_channels,
            out_channels=64,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=self.conv2_1.out_channels,
            out_channels=64,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.batch_norm2 = nn.BatchNorm2d(self.conv2_2.out_channels)
        self.conv3_1 = nn.Conv2d(
            in_channels=self.conv2_2.out_channels,
            out_channels=128,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.conv3_2 = nn.Conv2d(
            in_channels=self.conv3_1.out_channels,
            out_channels=128,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        self.batch_norm3 = nn.BatchNorm2d(self.conv3_2.out_channels)

        self.fc1 = nn.Linear(
            in_features=self.conv3_2.out_channels * 3 * 3, out_features=2, bias=bias
            )
        self.fc2 = nn.Linear(
            in_features=2, out_features=num_classes, bias=bias
            )
        
        self.single_fc = nn.Linear(
            in_features=self.conv3_2.out_channels * 3 * 3, out_features=num_classes, bias=bias
            )
        
        self.single_fc_poslin = PosLinear(
            in_features=self.conv3_2.out_channels * 3 * 3, out_features=num_classes, input_bias=bias
            )
        
        self.double_fc_poslin = PosLinear(
            in_features=2, out_features=num_classes, input_bias=bias
            )
        
        if mixed:
            self.fc2_ood = nn.Linear(
                in_features=2, out_features=1, bias=True
            )

            self.single_fc_ood = nn.Linear(
                in_features=self.conv3_2.out_channels * 3 * 3, out_features=1, bias=True
            )

            self.single_fc_poslin_ood = PosLinear(
                in_features=self.conv3_2.out_channels * 3 * 3, out_features=1, input_bias=True
            )

        self.prelu_act1 = nn.PReLU()
        self.prelu_act2 = nn.PReLU()
        self.prelu_act3 = nn.PReLU()
        self.network_type = network_type
        self.mixed = mixed

    def forward(self, x):
        x = self.prelu_act1(self.pool(self.batch_norm1(self.conv1_2(self.conv1_1(x)))))
        x = self.prelu_act2(self.pool(self.batch_norm2(self.conv2_2(self.conv2_1(x)))))
        x = self.prelu_act3(self.pool(self.batch_norm3(self.conv3_2(self.conv3_1(x)))))
        x = x.view(-1, self.conv3_2.out_channels * 3 * 3)
        if self.network_type == "single_fc":
            y = nn.functional.relu(x)
            x = self.single_fc(y)
        elif self.network_type == "single_fc_poslin":
            y = nn.functional.relu(x)
            x = self.single_fc_poslin(y)
        elif self.network_type == "double_fc":
            y = self.fc1(x)
            y = nn.functional.relu(y)
            x = self.fc2(y)
        elif self.network_type == "double_fc_poslin":
            y = self.fc1(x)
            y = nn.functional.relu(y)
            x = self.double_fc_poslin(y)
        else:
            y = self.fc1(x)
            x = self.fc2(y)
        
        if self.mixed:
            if self.network_type == "single_fc":
                x2 = self.single_fc_ood(y)
            elif self.network_type == "single_fc_poslin":
                x2 = self.single_fc_poslin_ood(y)
            else:
                x2 = self.fc2_ood(y)
            return x, x2, y
        return x, y


    
class LeNet(nn.Module):
    def __init__(self, network_type="regular", num_classes=10, bias=False, mixed=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=20, kernel_size=(5, 5), stride=1, padding=2
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=50,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )

        self.fc1 = nn.Linear(
            in_features=self.conv2.out_channels * 7 * 7, out_features=500, bias=bias
        )
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes, bias=bias)

        self.single_fc = nn.Linear(
            in_features=self.conv2.out_channels * 7 * 7, out_features=num_classes, bias=bias
            )
        
        self.single_fc_poslin = PosLinear(
            in_features=self.conv2.out_channels * 7 * 7, out_features=num_classes, input_bias=bias
            )
        
        self.double_fc_poslin = PosLinear(
            in_features=500, out_features=num_classes, input_bias=bias
            )
        
        if mixed:
            self.fc2_ood = nn.Linear(
                in_features=500, out_features=1, bias=True
            )

            self.single_fc_ood = nn.Linear(
                in_features=self.conv2.out_channels * 7 * 7, out_features=1, bias=True
            )

            self.single_fc_poslin_ood = PosLinear(
                in_features=self.conv2.out_channels * 7 * 7, out_features=1, input_bias=True
            )
        
        self.relu_act = nn.ReLU()
        self.network_type = network_type
        self.mixed = mixed

    def forward(self, x):
        x = self.pool(self.relu_act(self.conv1(x)))
        x = self.pool(self.relu_act(self.conv2(x)))
        x = x.view(-1, self.conv2.out_channels * 7 * 7)
        if self.network_type == "single_fc":
            y = nn.functional.relu(x)
            x = self.single_fc(y)
        elif self.network_type == "single_fc_poslin":
            y = nn.functional.relu(x)
            x = self.single_fc_poslin(y)
        elif self.network_type == "double_fc":
            y = self.fc1(x)
            y = nn.functional.relu(y)
            x = self.fc2(y)
        elif self.network_type == "double_fc_poslin":
            y = self.fc1(x)
            y = nn.functional.relu(y)
            x = self.double_fc_poslin(y)
        else:
            y = self.fc1(x)
            x = self.fc2(y)

        if self.mixed:
            if self.network_type == "single_fc":
                x2 = self.single_fc_ood(y)
            elif self.network_type == "single_fc_poslin":
                x2 = self.single_fc_poslin_ood(y)
            else:
                x2 = self.fc2_ood(y)
            return x, x2, y
        return x, y
    
class PosLinear(nn.Module):
    def __init__(self, in_features, out_features, input_bias=False):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, out_features)))
        
        if input_bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        else:
            self.bias = None

    def forward(self, x):
        linear_result = torch.matmul(x, nn.functional.relu(self.weight))
        
        if self.bias is not None:
            linear_result += nn.functional.relu(self.bias)
            
        return linear_result
    