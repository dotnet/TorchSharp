from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import importsd


NUM_WORDS = 100
EMBEDDING_VEC_LEN = 100
HIDDEN_SIZE = 128


class LSTMModel(nn.Module):
    # The names of properties should be the same in C# and Python
    # otherwise, you have to manually change the key name in the state_dict
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(NUM_WORDS, EMBEDDING_VEC_LEN)
        self.lstm = nn.LSTM(EMBEDDING_VEC_LEN, HIDDEN_SIZE, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_embed = self.embedding(x)
        x_lstm, _ = self.lstm(x_embed)
        x_lstm_last_seq = x_lstm[:, -1, :]
        x_lstm_last_seq = self.dropout(x_lstm_last_seq)
        logits = self.dense(x_lstm_last_seq)
        out = self.sigmoid(logits)
        return out


class LeNet1Model(nn.Module):
    # The names of properties should be the same in C# and Python
    # in this case, we both name the Sequential as layers
    def __init__(self):
        super(LeNet1Model, self).__init__()
        # the names of each layer should also be the same in C# and Python
        modules = OrderedDict([
            ("conv-1", nn.Conv2d(1, 4, 5, padding=2)),
            ("bnrm2d-1", nn.BatchNorm2d(4)),
            ("relu-1", nn.ReLU()),
            ("maxpool-1", nn.MaxPool2d(2, stride=2)),
            ("conv-2", nn.Conv2d(4, 12, 5)),
            ("bnrm2d-2", nn.BatchNorm2d(12)),
            ("relu-2", nn.ReLU()),
            ("maxpool-2", nn.MaxPool2d(2, stride=2)),
            ("flatten", nn.Flatten()),
            ("linear", nn.Linear(300, 10)),
        ])
        self.layers = nn.Sequential(modules)

    def forward(self, x):
        return self.layers.forward(x)


def testLSTM():
    print("testing LSTM")
    mylstm = LSTMModel()
    with open("lstm.dat", "rb") as f:
        sd = importsd.load_state_dict(f)
    # you can change the loaded key names here, for example:
    # sd = {k + "py": v for k, v in sd}
    # you can check the key names of state_dict in python by:
    # print(mylstm.state_dict())
    mylstm.load_state_dict(sd)

    # init values & functions
    torch.manual_seed(0)
    np.random.seed(0)
    labels = torch.tensor(np.random.randint(0, 1, [100, 1]), dtype=torch.float)
    inputs = torch.tensor(np.random.randint(0, 100, [100, 100]))
    opt = optim.Adam(mylstm.parameters(), lr=8e-5)
    loss_func = nn.BCELoss()

    # evaluation before training
    mylstm.eval()
    preds = mylstm.forward(inputs)
    preds = torch.round(preds)
    correct_num = torch.sum(preds == labels).item()
    print(f"before training: {correct_num} corrected")

    # training for 50 steps
    mylstm.train()
    for i in range(50):
        opt.zero_grad()  # Reset the gradient in every iteration
        outputs = mylstm(inputs)
        loss = loss_func(outputs, labels)  # Loss forward pass
        loss.backward()  # Loss backaed pass
        opt.step()  # Update all the parameters by the given learnig rule

    # evaluation after training
    mylstm.eval()
    preds = mylstm.forward(inputs)
    preds = torch.round(preds)
    correct_num = torch.sum(preds == labels).item()
    print(f"after training: {correct_num} corrected")


def testLeNet1():
    print("testing LeNet1")
    mylenet = LeNet1Model()
    with open("lenet1.dat", "rb") as f:
        sd = importsd.load_state_dict(f)
    # you can change the loaded key names here, for example:
    # sd = {k + "py": v for k, v in sd}
    # you can check the key names of state_dict in python by:
    # print(mylenet.state_dict())
    mylenet.load_state_dict(sd)

    # init values & functions
    torch.manual_seed(0)
    np.random.seed(0)
    labels = torch.tensor(np.random.randint(0, 10, [100]))
    inputs = torch.tensor(np.random.randint(0, 255, [100, 1, 28, 28]) / 255.0, dtype=torch.float32)
    opt = optim.Adam(mylenet.parameters(), lr=8e-5)
    loss_func = nn.CrossEntropyLoss()

    # evaluation before training
    mylenet.eval()
    output = mylenet.forward(inputs)
    _, preds = torch.max(output.data, dim=1)
    correct_num = torch.sum(preds == labels).item()
    print(f"before training: {correct_num} corrected")

    # training for 200 steps
    mylenet.train()
    for i in range(200):
        opt.zero_grad()  # Reset the gradient in every iteration
        outputs = mylenet(inputs)
        loss = loss_func(outputs, labels)  # Loss forward pass
        loss.backward()  # Loss backaed pass
        opt.step()  # Update all the parameters by the given learnig rule

    # evaluation after training
    mylenet.eval()
    output = mylenet.forward(inputs)
    _, preds = torch.max(output.data, dim=1)
    correct_num = torch.sum(preds == labels).item()
    print(f"after training: {correct_num} corrected")


if __name__ == '__main__':
    testLSTM()
    testLeNet1()
