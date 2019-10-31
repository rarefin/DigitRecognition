from DataLoader import DataSet
from torchvision import transforms
import json
import torch
import numpy as np
from Model import DigitNet
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from os.path import join
import os
import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch import nn


def get_loss(length_logit, digit_logits, length_labels, digits_labels):
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
            criterion.cuda()

    loss = criterion(length_logit, length_labels)

    for i in range(len(digit_logits)):
        loss += criterion(digit_logits[i], digits_labels[i])

    return loss


def calculate_correct(digit_logits, digits_labels):

    digit_prediction = []
    for i in range(len(digit_logits)):
        digit_prediction.append(digit_logits[i].max(1)[1])

    num_correct = 0
    num_correct += (digit_prediction[0].eq(digits_labels[0]) &
                    digit_prediction[1].eq(digits_labels[1]) &
                    digit_prediction[2].eq(digits_labels[2]) &
                    digit_prediction[3].eq(digits_labels[3]) &
                    digit_prediction[4].eq(digits_labels[4])).sum()

    return num_correct


def trainAndGetBestModel(model, optimizer, dataloaders, config):
    np.random.seed(11)  # seed all RNGs for reproducibility
    torch.manual_seed(11)

    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]

    subfolder_pattern = 'batch_{}_time_{}'.format(batch_size, f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S-%f}")

    checkpoint_dir = os.path.join(config["paths"]["checkpoint_dir"], 'checkpoints', subfolder_pattern)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logging_dir = os.path.join(config['paths']['tb_log_file_dir'], 'log', subfolder_pattern)
    os.makedirs(logging_dir, exist_ok=True)

    writer = SummaryWriter(logging_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config['training']['lr_decay'],
                                               verbose=True, patience=config['training']['lr_step'])

    best_acc = 0
    for epoch in tqdm(range(1, num_epochs + 1)):

        # Train
        model.train()
        train_loss, train_acc = 0.0,  0.0  # monitor train loss and accuracy

        # Iterate over data
        for img, length_labels, digit_labels in tqdm(dataloaders['train']):

            optimizer.zero_grad()  # zero the parameter gradients

            img = img.float().to(device)
            length_labels = length_labels.to(device)
            digit_labels = [digit.to(device) for digit in digit_labels]

            length_logit, digit_logits = model(img)

            loss = get_loss(length_logit, digit_logits, length_labels, digit_labels)

            # Backprop
            loss.backward()
            optimizer.step()
            epoch_loss = loss.detach().cpu().item() * len(img)
            train_loss += epoch_loss
            train_acc += calculate_correct(digit_logits, digit_labels).detach().cpu().item()


        # Evaluation
        model.eval()
        val_loss, val_acc = 0.0,  0.0  # monitor val loss and accuracy

        for img, length_labels, digit_labels in dataloaders['val']:
            img = img.float().to(device)
            length_labels = length_labels.to(device)
            digit_labels = [digit.to(device) for digit in digit_labels]

            length_logit, digit_logits = model(img)

            loss = get_loss(length_logit, digit_logits, length_labels, digit_labels)

            epoch_loss = loss.detach().cpu().item() * len(img)
            val_loss += epoch_loss
            val_acc += calculate_correct(digit_logits, digit_labels).detach().cpu().item()

        train_loss /= len(dataloaders['train'].dataset)
        val_loss /= len(dataloaders['val'].dataset)

        train_acc /= len(dataloaders['train'].dataset)
        val_acc /= len(dataloaders['val'].dataset)

        if best_acc > val_acc:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'DigitNet.pth'))
            best_acc = val_acc

        print("Epoch--> ", epoch)
        print("train---> loss: ", train_loss, ", accuracy: ", train_acc)
        print("Val---> loss: ", val_loss, ", accuracy: ", val_acc)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        scheduler.step(val_acc)
    writer.close()


def main(config):

    # Reproducibility options
    np.random.seed(0)  # RNG seeds
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize the network based on the network configuration
    model = DigitNet()

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["lr"])

    data_directory = config["paths"]["data_directory"]
    batch_size = config["training"]["batch_size"]
    n_workers = config["training"]["n_workers"]

    transform = transforms.Compose([
        transforms.RandomCrop([54, 54]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = DataSet(join(data_directory, "train.lmdb"), transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)

    val_dataset = DataSet(join(data_directory, "val.lmdb"), transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Train model
    torch.cuda.empty_cache()

    trainAndGetBestModel(model, optimizer, dataloaders, config)


if __name__ == '__main__':

    with open('config.json', "r") as read_file:
        config = json.load(read_file)

    main(config)