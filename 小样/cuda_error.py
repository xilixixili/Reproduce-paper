from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, kernel_size, in_channels, channel1, channel2):
        super(Net, self).__init__()
        self.theta1, self.Lambda1, self.psi1, self.bias1 = self.generate_parameters(channel1, in_channels)
        self.filter1 = self.whole_filter(in_channels, channel1, kernel_size, self.theta1, self.Lambda1, self.psi1).cuda()

        self.theta2, self.Lambda2, self.psi2, self.bias2 = self.generate_parameters(channel2, channel1)
        self.filter2 = self.whole_filter(channel1, channel2, kernel_size, self.theta2, self.Lambda2, self.psi2).cuda()
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x):
        x = F.conv2d(x, self.filter1, bias=self.bias1)
        x = F.max_pool2d(x, 2, 2)
        x = F.conv2d(x, self.filter2, bias=self.bias2)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    def generate_parameters(self, dim_out, dim_in):
        theta = nn.Parameter(torch.randn(dim_out, dim_in))
        Lambda = nn.Parameter(torch.randn(dim_out, dim_in))
        psi = nn.Parameter(torch.randn(dim_out, dim_in))
        bias = nn.Parameter(torch.randn(dim_out))
        return theta, Lambda, psi, bias


    def whole_filter(self, in_channels, out_channels, kernel_size, theta_column, Lambda_column, psi_column):
        result = torch.zeros(out_channels, in_channels, kernel_size, kernel_size) # \text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW
        for i in range(out_channels):
            result[i] = self.one_filter(in_channels, kernel_size, theta_column[i], Lambda_column[i], psi_column[i])
        return result


    def one_filter(self, in_channels, kernel_size, theta, Lambda, psi):
        result = torch.zeros(in_channels, kernel_size, kernel_size)
        for i in range(in_channels):
            result[i] = self.filter_fn(theta[i], Lambda[i], psi[i], kernel_size)
        return result


    def filter_fn(self, theta, Lambda, psi, kernel_size):
        # Bounding box
        half_size = (kernel_size - 1) // 2
        ymin, xmin = -half_size, -half_size
        ymax, xmax = half_size, half_size
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
        y = torch.FloatTensor(y)
        x = torch.FloatTensor(x)

        # Rotation
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        box = torch.cos(y_theta) * torch.sin(2 * np.pi / Lambda * x_theta + psi)
        return box



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net(5, 1, 20, 50).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for param in model.parameters():
        print(type(param.data), param.size())

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()