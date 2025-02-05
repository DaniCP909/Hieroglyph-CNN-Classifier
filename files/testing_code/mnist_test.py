
import cv2
import matplotlib.pyplot as plt

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #(input, output(n filters), kernel_size, stride)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128) #input features from previous layer, reduce dim
        self.fc2 = nn.Linear(128, 10) #output prev dense layer, n classes

    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def train(args, model, device, train_loader, optimizer, epoch):
        train_lossess = []
        train_counter = []
        test_losses = []
        test_counter = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_lossess.append(loss.item)
                train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(model.state_dict(), '/results/model.pth')
                torch.save(optimizer.state_dict(), '/results/optimizer.pth')
                if args.dry_run:
                    break
    

def main():

    # TRAININIG SETTINGS
    parser = argparse.ArgumentParser(description='Entrenamiento MNIST con PyTorch')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default = 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default = 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default = 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default = 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='learning rate step gamma (default = 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='desables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False, help='disables MacOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default = 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='Saves current Model')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    print(f'*** Training settings: batch_size:{args.batch_size}, test_bath_size:{args.test_batch_size}, epochs:{args.epochs}, lr:{args.lr}, gamma:{args.gamma}, no-cuda:{args.no_cuda}, no_mps:{args.no_mps}, dry_run:{args.dry_run}, seed:{args.seed}, log_interval:{args.log_interval}, save_model:{args.save_model}')


    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kargs.update(cuda_kwargs)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1317,), (0.3081,))
    ])

    dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, **test_kargs)

    examples = enumerate(test_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    example_figure = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title('Ground truth: {}'.format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    example_figure.savefig('./examples/examples1.png')
    

if __name__ == '__main__':
    main()