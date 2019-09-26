import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    assert False, 'Error in import modules: %s' % sys.argv[0].split("/")[-1]

try:
    from import_cifar100_data import *
except ImportError:
    assert False, 'Error in import local functions.'


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


def get_layers(filename):
    layers = []
    with open(filename) as f:
        C = 3
        flatten = False
        for line in f.readlines():
            layer = line.split(", ")
            layer[-1] = layer[-1][:-1]
            print(layer)

            if layer[0] == 'Conv':
                W, H, F, s = layer[1:]
                W, H, F, s = int(W), int(H), int(F), int(s)
                layers.append(nn.Conv2d(C, F, (W, H), s, padding=(W - 1) // 2))
                layers.append(nn.ReLU())
                C = F
            elif layer[0] == 'MaxPool':
                W, H, s = layer[1:]
                W, H, s = int(W), int(H), int(s)
                layers.append(nn.MaxPool2d((W, H), s))
            elif layer[0] == 'FC':
                W = int(layer[1])
                if not flatten:
                    flatten = True
                    layers.append(Flatten())
                    layers.append(nn.Linear(512, W))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(p=0.5))
                else:
                    layers.append(nn.Linear(1024, W))
                    if W != 100:
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(p=0.5))

    return nn.ModuleList(layers)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.layers = get_layers('vgg-cifar100')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = VGGNet().cuda(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    X_train, X_val, X_test, y_train, y_val, y_test = get_cifar100_data(sys.argv[1])

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        time = 0
        for X_batch, y_batch in zip(X_train, y_train):
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            X_batch, y_batch = torch.tensor(X_batch, device=device).float(), torch.tensor(y_batch, device=device).long()

            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # print statistics
            time += 1
            running_loss += loss.item()
            if time % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, time, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    main()