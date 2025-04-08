
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

from components.VisualizeTools import plot_predictions_table, compareDatasetPredicts

from ModMnistModel import ModMnistModel
from MnistModel import MnistModel

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

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
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.item()))
            train_lossess.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), f'../results/model_results/my_model_shortfont{args.short_font}_fill{args.fill}.pth')
            torch.save(optimizer.state_dict(), f'../results/model_results/my_optimizer_shortfont{args.short_font}_fill{args.fill}.pth')
            if args.dry_run:
                break
    print(f"Trained Epoch: {epoch}")
    
def test(model, device, test_loader, test_lossess, correct_history):
    model.eval()
    test_loss = 0
    correct = 0
    correct_predictions = {}  # Dictionary to store correct classifications

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get index of max log-probability
            
            all_predictions.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_lossess.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracy = 100. * correct / len(test_loader.dataset)

    return all_predictions, all_targets, accuracy

    

def main():

    # TRAININIG SETTINGS
    parser = argparse.ArgumentParser(description='Entrenamiento MNIST con PyTorch')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default = 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N', help='input batch size for testing (default = 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default = 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default = 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='learning rate step gamma (default = 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='desables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False, help='disables MacOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default = 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='Saves current Model')
    parser.add_argument('--large-mod', action='store_true', default=False, help='Select model False(1024) or True(2048)')
    parser.add_argument('--short-font', action='store_true', default=False, help='Select short filled font False(Noto+Gardenier) or True(Silhousette)')
    parser.add_argument('--fill', action='store_true', default=False, help='Fill contours in generator')
    parser.add_argument('--experiment', type=int, default=0, metavar='N', help='id of experiment for indexing results')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)
    print(f'*** Training settings: batch_size:{args.batch_size}, test_batch_size:{args.test_batch_size}, epochs:{args.epochs}, lr:{args.lr}, gamma:{args.gamma}, no-cuda:{args.no_cuda}, no_mps:{args.no_mps}, dry_run:{args.dry_run}, seed:{args.seed}, log_interval:{args.log_interval}, save_model:{args.save_model}')


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
        "../files/fonts/Noto_Sans_Egyptian_Hieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf",
        "../files/fonts/NewGardiner/NewGardinerBMP.ttf",
        "../files/fonts/JSeshFont/JSeshFont.ttf",
            ]
    ranges = [ 
        (0x00013000, 0x0001342E),
        (0x0000E000, 0x0000E42E),
        (0x00013000, 0x0001342E),
            ]
    
    path_short = [
    "../files/fonts/egyptian-hieroglyphs-silhouette/EgyptianHieroglyphsSilhouet.otf"
    ]
    short_font_tags = [33,36,37,40,41,43,45,49,50,51,52,53,54,55,56,57,64,
                       65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,81,82,
                       83,84,85,86,87,88,89,90,97,98,99,100,101,102,103,104,
                       105,106,107,108,109,110,111,112,113,114,115,116,117,
                       118,119,120,121,122,162,163,165]
    range_short = (0, len(short_font_tags) - 1)

    struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    if args.short_font: 
        generator = HieroglyphCharacterGenerator(path_short[0], range_short[0], range_short[1], font_size=100, short_font=True)
        augmentator = HieroglyphAugmentator([generator], mask=struct_element, fill=False)
        generator_len = generator.getFontLength()
    else: 
        all_generators = []
        for path,hex_range in zip(paths,ranges):
            all_generators.append(HieroglyphCharacterGenerator(path, hex_range[0], hex_range[1], font_size=100))
        augmentator = HieroglyphAugmentator(all_generators, mask=struct_element, fill=args.fill)
        generator_len = all_generators[0].getFontLength()

    print(f"* Generator selected: length = {generator_len}")

    dataset_train = HieroglyphDataset(generator_len, augmentator=augmentator)
    dataset_test = HieroglyphDataset(generator_len, augmentator=augmentator)

    train_dataloader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, **test_kargs)

    
    train_lossess = []
    train_counter = []
    test_lossess = []
    test_counter = [i * len(train_dataloader.dataset) for i in range(args.epochs + 1)]
    correct_predictions_history = {}

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

    if args.large_mod: model = MnistModel(num_classes=generator_len).to(device)
    else: model = MnistModel(num_classes=generator_len).to(device)

    print(f'--- Selected: {device}')
    optimizer = optim.Adadelta(model.parameters(), lr = args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    predictions, targets, accuracy = test(model, device, test_dataloader, test_lossess, correct_predictions_history)
    for epoch in range(1, args.epochs + 1):
        augmentator.incrementSeed(epoch) #starts at 1 or origin_seed + 1 
        train(args, model, device, train_dataloader, optimizer, epoch, train_lossess, train_counter)
        if epoch == 1: augmentator.incrementTestSeed()#sets always 0 or origin_seed
        predictions, targets, accuracy = test(model, device, test_dataloader, test_lossess, correct_predictions_history) 
        print(f"Targets{targets[:20]}")
        print(f"Predicts{predictions[:20]}")
        plot_predictions_table(predictions, targets, epoch)
        scheduler.step()
    compareDatasetPredicts(dataset_test, targets=targets, predicts=predictions, experiment=args.experiment)

    performance_fig = plt.figure()
    plt.plot(train_counter, train_lossess, color='green', zorder=3)
    plt.scatter(test_counter, test_lossess, color='purple', zorder=2)
    plt.legend(['Train loss', 'Test loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('begative log likelihood loss')
    performance_fig.savefig(f'../results/my_performance_shortfont{args.short_font}_fill{args.fill}.png')

    print("Correct classes: ")
    print(correct_predictions_history)


    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")   

if __name__ == '__main__':
    main()