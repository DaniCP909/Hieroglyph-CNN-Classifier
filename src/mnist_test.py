
import cv2
import matplotlib.pyplot as plt

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from components.HieroglyphCharacterGenerator import HieroglyphCharacterGenerator
from components.HieroglyphAugmentator import HieroglyphAugmentator
from components.HieroglyphDataset import HieroglyphDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1) #(input, output(n filters), kernel_size, stride)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 48 * 48, 1024) #input features from previous layer, reduce dim
        self.fc2 = nn.Linear(1024, 1071) #output prev dense layer, n classes

    
    def forward(self, x):           #28x28                                  |   #64x64 57600                            |   #128x128 246016
        x = self.conv1(x)           #[bs, 1, 28, 28] --> [bs, 32, 26, 26]   |   #[bs, 1, 64, 64] --> [bs, 32, 62, 62]   |   #[bs, 1, 128, 128] --> [bs, 32, 126, 126]
        x = F.relu(x)
        x = self.conv2(x)           #[bs, 32, 26, 26] --> [bs, 64, 24, 24]  |   #[bs, 32, 62, 62] --> [bs, 64, 60, 60]  |   #[bs, 32, 126, 126] --> [bs, 64, 124, 124]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)      #[bs, 64, 24, 24] --> [bs, 64, 12, 12]  |   #[bs, 64, 60, 60] --> [bs, 64, 30, 30]  |   #[bs, 64, 124, 124] --> [bs, 64, 62, 62]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    


def train(args, model, device, train_loader, optimizer, epoch, train_lossess, train_counter):
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
            train_lossess.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), './results/model_results/my_model.pth')
            torch.save(optimizer.state_dict(), './results/model_results/my_optimizer.pth')
            if args.dry_run:
                break
    
def test(model, device, test_loader, test_lossess):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_lossess.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    

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

    paths = [ 
        "./files/fonts/Noto_Sans_Egyptian_Hieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf",
        "./files/fonts/NewGardiner/NewGardinerBMP.ttf",
            ]
    ranges = [ 
        (0x00013000, 0x0001342E),
        (0x0000E000, 0x0000E42E),
            ]

    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    generator = HieroglyphCharacterGenerator(paths[0], ranges[0][0], ranges[0][1], font_size=100)
    augmentator = HieroglyphAugmentator([generator], mask=struct_element)


    dataset_train = HieroglyphDataset(1071, augmentator=augmentator)
    dataset_test = HieroglyphDataset(1071, augmentator=augmentator)

    train_dataloader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, **test_kargs)

    
    train_lossess = []
    train_counter = []
    test_lossess = []
    test_counter = [i * len(train_dataloader.dataset) for i in range(args.epochs + 1)]

    #examples = enumerate(test_dataloader)
    #batch_idx, (example_data, example_targets) = next(examples)
#
    #print(example_data.shape)
#
    #example_figure = plt.figure()
    #for i in range(6):
    #    plt.subplot(2, 3, i+1)
    #    plt.tight_layout()
    #    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #    plt.title('Ground truth: {}'.format(example_targets[i]))
    #    plt.xticks([])
    #    plt.yticks([])
    #example_figure.savefig('./results/my_examples1.png')

    model = Net().to(device)
    print(f'--- Selected: {device}')
    optimizer = optim.Adadelta(model.parameters(), lr = args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test(model, device, test_dataloader, test_lossess)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_dataloader, optimizer, epoch, train_lossess, train_counter)
        test(model, device, test_dataloader, test_lossess) 
        scheduler.step()

    performance_fig = plt.figure()
    plt.plot(train_counter, train_lossess, color='green', zorder=3)
    plt.scatter(test_counter, test_lossess, color='purple', zorder=2)
    plt.legend(['Train loss', 'Test loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('begative log likelihood loss')
    performance_fig.savefig('./results/my_performance.png')

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")   

if __name__ == '__main__':
    main()