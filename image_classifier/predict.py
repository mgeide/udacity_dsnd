#!/user/bin/env python
####
## predict.py
##   - uses a trained network to predict the class for an input image
##
## Basic usage: pyton predict.py </path/to/image> <checkpoint>
## Prints out: predicted flower name(s) along with the probability of that name
## Options (argparse):
##      Top K predicted classes: --top_k <num>
##      Mapping of categories to real names: --category_names <JSON File>
##      Use GPU: --gpu
####

## Imports
import os
import argparse
import json
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image

# Import model initialization function from train.py
from train import init_model

## Defaults
d_checkpoint        = 'checkpoints/checkpoint.pth'
d_categories_json   = 'cat_to_name.json'
d_topk              = 1

## Argument Handling
arg_parser = argparse.ArgumentParser(description='Predict arguments parser')

# Files & Directories, and "top K"
arg_parser.add_argument('image_path', metavar='FILE_PATH', type=str, help='Path to image file')
arg_parser.add_argument('--checkpoint', metavar='FILE_PATH', type=str, default=d_checkpoint, help='Path to checkpoint file')
arg_parser.add_argument('--category_names', metavar='FILE_PATH', type=str, default=d_categories_json, help='Categories to name json')
arg_parser.add_argument('--top_k', metavar='INT', type=int, default=d_topk, help='Number of top predictions to show')
# GPU option
arg_parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

####
## Load model
def load_model(args):
    checkpoint_filename = args.checkpoint
    checkpoint = torch.load(checkpoint_filename)

    # Get args from saved checkpoint for rebuilding model
    args.arch = checkpoint['arch']
    args.input_size = checkpoint['input_size']
    args.output_size = checkpoint['output_size']
    args.hidden_units = checkpoint['hidden_size']
    args.dropout = checkpoint['dropout']
    args.pretrained = True
    
    model = init_model(args)
    model.load_state_dict(checkpoint['model_state'])
    return model

####
## Process Image
def process_image(image_path):
    scale       = 256
    input_shape = 224
    mean        = np.array([0.485, 0.456, 0.406])
    std         = np.array([0.229, 0.224, 0.225])

    # Process a PIL image for use in a PyTorch model
    try:
        pil_image = Image.open(image_path)
        #pil_image.show()
    except IOError:
        print("ERROR opening ", image_path)
        return -1

    pil_loader = transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor()])
    pil_image = pil_loader(pil_image).float()

    np_image = np.array(pil_image)
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    return np_image

####
## Predict
def predict(args, model):

    image_path  = args.image_path
    top_k       = args.top_k

    # Process Categories JSON file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Build pytorch tensor from image for prediction
    image = process_image(image_path)
    image = torch.autograd.Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)

    model.train(False)
    model.eval()

    device = "cpu"
    # Run prediction on GPU if flag is set and GPU is available
    if args.gpu and torch.cuda.is_available():
        device = "cuda"
    image.to(device)
    model.to(device)

    # Run top_k predictions for image against the model
    results = model(image).topk(top_k)
    probs = torch.nn.functional.softmax(results[0].data, dim=1).cpu().numpy()[0]
    classes = results[1].data.cpu().numpy()[0]

    class_names = []
    for x in classes:
        x = str(x)
        if x not in (cat_to_name.keys()):
            #print("Unknown category!")
            class_names.append("Unknown")
        else:
            #print("category name:", cat_to_name[x])
            class_names.append(cat_to_name[x])

    return probs, class_names

## main
if __name__ == '__main__':

    #### Process arguments
    args = arg_parser.parse_args()

    #### Load model from checkpoint
    model = load_model(args)

    #### Make prediction
    probs, class_names = predict(args, model)

    # Print class_names and probabilities
    print('Predicted class names with probabilities:', list(zip(class_names, probs)))
