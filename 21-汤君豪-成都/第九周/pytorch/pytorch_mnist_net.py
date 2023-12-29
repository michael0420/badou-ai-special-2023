import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms



class mnistnet(nn.Module):
    def __init__(self):
        super(mnistnet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])]
    )

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader

class model:
    def __init__(self, net, cost, opti):
        self.net = net
        self.cost = self.create_cost(cost)
        self.opti = self.create_opti(opti)

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_opti(self, opti, **rests):
        support_opti = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_opti[opti]

    def train(self, train_loader, epochs=3):
        # import tqdm
        # pbar = tqdm.trange(epochs)
        # for epoch in pbar:
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.opti.zero_grad()
                predict = self.net.forward(inputs)
                loss = self.cost(predict, labels)
                loss.backward()
                self.opti.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                predict = self.net.forward(inputs)
                predict = torch.argmax(predict, 1)

                total += len(labels)
                correct += (predict == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    net = mnistnet()
    trainloader, testloader = mnist_load_data()
    model = model(net, 'CROSS_ENTROPY', 'RMSP')
    model.train(trainloader)
    model.evaluate(testloader)

