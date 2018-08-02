#!/user/bin/env python
####
## train.py
##   - train a new network on a dataset and save the model as a checkpoint
##
## Basic usage: pyton train.py <data_directory>
## Prints out: training loss, validation loss, and validation accuracy
## Options (argparse):
##      Set directory to save checkpoints: --save_dir <save_directory>
##      Choose architecture: --arch "vgg13"
##      Set hyperparameters: --learning_rate 0.01 --hidden_units 512 --epochs 20
##      Use GPU for training: --gpu
####

## Imports
import os
import argparse
import json
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models

## Defaults
d_checkpoints_dir   = 'checkpoints/'
d_categories_json   = 'cat_to_name.json'
d_arch              = 'vgg19_bn'
d_learning_rate     = 0.01
d_hidden_units      = 4096
d_dropout           = 0.5
d_epochs            = 30


## Argument Handling
arg_parser = argparse.ArgumentParser(description='Training arguments parser')

#### Files & Directories
arg_parser.add_argument('data_dir', metavar='DIR', type=str, help='Dataset dir')
arg_parser.add_argument('--save_dir', metavar='DIR', type=str, default=d_checkpoints_dir, help='Save checkpoints dir')
arg_parser.add_argument('--cat_json', metavar='FILE', type=str, default=d_categories_json, help='Categories to name json')

#### Model Selection
arg_parser.add_argument('--arch', metavar='ARCH', type=str, default=d_arch, help='Model architecture')

#### Optional hyperparameters
#### TODO: add in other hyperparameters, e.g., momentum, weight_decay
arg_parser.add_argument('--learning_rate', default=d_learning_rate, type=float, help='Learning rate')
arg_parser.add_argument('--hidden_units', default=d_hidden_units, type=int, help='Hidden units')
arg_parser.add_argument('--dropout', default=d_dropout, type=int, help='Dropout')
arg_parser.add_argument('--epochs', default=d_epochs, type=int, help='Training loops')

#### GPU
arg_parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

####
## Initialize Dataloaders
def init_dataloaders(args):

    #### Data location
    data_root = args.data_dir
    data_dir = {
        'train': os.path.join(data_root, 'train'),
        'validate': os.path.join(data_root, 'valid'),
        'test': os.path.join(data_root, 'test')
    }

    #### Data transform variables
    scale       = 256
    input_shape = 224
    mean        = [0.485, 0.456, 0.406]
    std         = [0.229, 0.224, 0.225]
    batch_size  = 32

    #### Transforms
    data_transforms = {
    'train': transforms.Compose([transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    'validate': transforms.Compose([transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    'test': transforms.Compose([transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}

    #### Load datasets with ImageFolder and Transforms
    image_datasets = {x: datasets.ImageFolder(data_dir[x], data_transforms[x]) for x in ['train', 'validate', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'validate', 'test']}

    return (image_datasets, dataloaders)

####
## Initialize Model
def init_model(args):

    #### Model variables
    pretrained  = args.pretrained
    input_size  = args.input_size
    hidden_size = args.hidden_units
    output_size = args.output_size
    dropout     = args.dropout

    #### Build model
    model = models.__dict__[args.arch](pretrained=pretrained)

    # Freeze params
    for param in model.parameters():
        param.requires_grad = False

    model.features = torch.nn.DataParallel(model.features)

    ##### Define new classifier
    classifier = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(input_size, hidden_size)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(dropout)),
            ('lin2', nn.Linear(hidden_size, hidden_size)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(dropout)),
            ('lin3', nn.Linear(hidden_size, output_size)),
            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier

    return model

####
## Deep Learn
def deep_learn(args, data_sets, data_loaders, model):

    #### Learning hyperparameters
    epochs          = args.epochs
    learning_rate   = args.learning_rate
    momentum        = 0.9       # TODO: these could be args
    decay           = 0.0001    #
    step_size       = 15        #
    gamma           = 0.1       #
    criterion       = nn.CrossEntropyLoss() # or nn.NLLLoss()

    #### If GPU flag set, use GPU if available
    if args.gpu and torch.cuda.is_available():
        args.gpu = True
    else:
        args.gpu = False
    gpu = args.gpu

    if gpu:
        model.cuda()
        criterion.cuda()

    # Adjust via SGD
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay) # or Adam

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Vars for tracking loss and accuracy
    best_epoch_loss     = 1000.0
    best_epoch_acc      = 0.0
    train_datasize      = len(data_sets['train'])
    validation_datasize = len(data_sets['validate'])
    checkpoint          = {}

    #### Epoch loop
    for e in range(epochs):

        # Reset loss and accuracy tracking
        training_loss_avg   = 0.0
        validation_loss_avg = 0.0
        validation_acc      = 0.0

        # Adjust learning rate
        scheduler.step()

        # Training step
        training_loss = model_training(data_loaders['train'], model, criterion, optimizer, gpu)
        training_loss_avg = float(training_loss) / train_datasize

        # Validation step
        (validation_loss, validation_acc) = model_validation(data_loaders['validate'], model, criterion, gpu)
        validation_loss_avg = float(validation_loss) / validation_datasize

        # Print stats
        print('Epoch {}: Training loss avg {:.4f}, Validation loss avg {:.4f}, Acc: {:.4f}'.format(e, training_loss_avg, validation_loss_avg, validation_acc))

        # Epoch stats for best model selection
        epoch_loss = training_loss_avg + validation_loss_avg
        epoch_acc = validation_acc

        # If best accuracy (and loss) save model state
        if (epoch_acc >= best_epoch_acc):
            if (epoch_acc > best_epoch_acc):
                best_epoch_acc = epoch_acc
                best_epoch_loss = epoch_loss
                print('Best model candidate, saving state')
                #best_model = model.state_dict()
                checkpoint = save_checkpoint(args, data_sets, model, optimizer, e, epoch_acc, epoch_loss)
            elif (epoch_loss < best_epoch_loss):
                best_epoch_acc = epoch_acc
                best_epoch_loss = epoch_loss
                print('Best model candidate, saving state')
                #best_model = model.state_dict()
                checkpoint = save_checkpoint(args, data_sets, model, optimizer, e, epoch_acc, epoch_loss)
    print("Done with epoch loop.")

    # load best model
    #model.load_state_dict(best_model)
    #best_model = load_model(args)
    return checkpoint

####
## Save checkpoint
def save_checkpoint(args, data_sets, model, optimizer, epoch, epoch_acc, epoch_loss):
    checkpoint_filename = os.path.join(args.save_dir, 'checkpoint.pth')
    model.class_to_idx = data_sets['train'].class_to_idx
    #print("Our model: \n\n", best_model, '\n')
    #print("The state dict keys: \n\n", best_model.state_dict().keys())
    checkpoint = {'epoch': epoch,
              'accuracy': epoch_acc,
              'loss': epoch_loss,
              'arch': args.arch,
              'input_size': args.input_size,
              'output_size': args.output_size,
              'hidden_size': args.hidden_units,
              'dropout': args.dropout,
              'optim_state': optimizer.state_dict(),
              'model_state': model.state_dict()}
    torch.save(checkpoint, checkpoint_filename)
    return checkpoint

####
## Model training
def model_training(train_loader, model, criterion, optimizer, gpu):
    running_loss = 0.0

    model.train(True)
    # Training loop through training data loader
    for ii, (inputs, labels) in enumerate(train_loader):

        #inputs, labels = inputs.to(device), labels.to(device)
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = torch.autograd.Variable(inputs)
        labels = torch.autograd.Variable(labels)

        # forward input to model and calc loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # zero gradiants, backward for training only, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss

####
## Model evaluation
def model_validation(validation_loader, model, criterion, gpu):

    running_loss    = 0.0
    running_correct = 0
    running_total   = 0
    acc             = 0.0

    model.train(False)
    model.eval()

    # Validation loop through validation data loader
    for jj, (inputs, labels) in enumerate(validation_loader):

        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = torch.autograd.Variable(inputs)
        labels = torch.autograd.Variable(labels)

        # forward input to model and calc loss
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Check how many we got correct
        _, predicted = torch.max(outputs.data, 1)
        running_total += labels.size(0)
        running_correct += (predicted == labels).sum().item()

    acc = 100.0 * (float(running_correct) / running_total)
    return (running_loss, acc)

## main
if __name__ == '__main__':

    #### Process arguments
    args = arg_parser.parse_args()

    # Process Categories JSON file
    with open(args.cat_json, 'r') as f:
        cat_to_name = json.load(f)

    # Tack on a few other arguments to track throughout workflow
    args.output_size    = len(cat_to_name)
    args.input_size     = 25088 # VGG19 feature inputs
    args.pretrained     = True

    #### Initialize data loaders
    dsets, dloaders = init_dataloaders(args)

    #### Initialize model
    model = init_model(args)

    #### Deep Learn
    checkpoint = deep_learn(args, dsets, dloaders, model)
    model.load_state_dict(checkpoint['model_state'])
