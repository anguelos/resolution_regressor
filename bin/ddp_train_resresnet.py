#!/usr/bin/env python3


import glob
import json
import re
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50
import random
import tqdm
import fargv
import pandas as pd
import sys


from resresnet import ResResNet, ResolutionDataset, simple_input_transform, create_cropping_transform


p = {
    "mode": [("train", "inference"), "The mode to run in"],
    "epochs": 100,
    "batch_size": 32,
    "lr": 1e-3,
    "image_size": 512,
    "num_workers": 4,
    "device": "cuda",
    "arch": [("resnet50","resnet18", "resnet34"), "The neural network head used"],
    "input_class": [("WritableArea","Seal", "CalibrationCard", "full"),"This is case censitive and should match what is on the filesystem for croped images."],
    "resume_path": "./data/models/res{arch}_{input_class}.pth",
    "weight_decay": 1e-4,
    "outlier_quantile": .2,
    "train_fraction": .8,
    "gt_paths": set({}),
    "gt_glob": "", #"../../1000_CVCharters/*/*/*/*.seals.crops/*.resolution.gt.json",
    "class_glob_regex": "[0-9]+\.resolution\.gt\.json",
    "class_glob_replace": "*.Img_{input_class}.jpg",
    "augment_scale_range": "[1.0, 1.0]",
}


def train_main(args):
    train_input_transform = create_cropping_transform(args.image_size)
    test_input_transform = simple_input_transform

    device = args.device
    (args.class_glob_regex, args.class_glob_replace)
    data = ResolutionDataset(gt_file_list=args.gt_paths,gt_glob=args.gt_glob, 
                             input_glob_subtuple=(args.class_glob_regex, args.class_glob_replace),
                             input_transform=test_input_transform)
    train_dataset, val_dataset = data.split(fraction=.8, shuffle=True, seed=1337)
    train_dataset.input_transform = train_input_transform
    #print(args.gt_paths)
    #print(data)
    #ds_items = sorted([f"{train_dataset.image_paths[n]}\t{train_dataset.ppcms[n]}" for n in range(len(train_dataset))])
    #print("\n".join(ds_items))
    #sys.exit(0)
    #val_dataset = ScaleDataset(resolution_gt_list=val_gt_paths, input_glob_subtuple=(args.class_glob_regex, args.class_glob_replace), input_transform=simple_input_transform, augment_scale_range=[1.,1.])

    #data = ScaleDataset()

    #train_dataset, val_dataset = data.split(fraction=.8, shuffle=True)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model, _, (epoch, hist, arg_hist) = ResResNet.resume(args.resume_path, args.arch, device)

    #model, epoch, hist, arg_hist = ResResNet.resume(args.resume_path, model, device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    while epoch < args.epochs:
        train_loss = 0.0
        val_loss = 0.0
        predictions = []
        gt = []
        model.train()
        for inputs, labels in tqdm.tqdm(train_loader,    desc=f"Epoch {epoch} "):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            predictions.append(outputs.cpu().detach().numpy())
            loss = criterion(outputs, labels.float()[:, None])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        print(f"Train loss: {train_loss/len(train_dataset)}")
        model.eval()
        predictions = []
        clean_predictions = []
        gt = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predictions.append(outputs.view(-1).cpu().detach().numpy())
                clean_predictions.append(model.predict_img(inputs[0,:,:,:], size=args.image_size, model=model, outlier_quantile=args.outlier_quantile))
                gt.append(labels.view(-1).cpu().detach().numpy())
                loss = criterion(outputs, labels.float()[:, None])
                val_loss += loss.item() * inputs.size(0)
        gt = np.concatenate(gt)
        predictions = np.concatenate(predictions)
        clean_predictions = np.array(clean_predictions)
        results = pd.DataFrame(data={"gt": gt, "pred": predictions, "clean_pred": clean_predictions})
        print(results.corr())
        print(f"Val loss: {val_loss/len(val_dataset)}")
        print(f"Val MSE: {((gt-predictions)**2).mean()}")
        print(f"Val Clean MSE: {((gt-clean_predictions)**2).mean()}")
        epoch += 1
        hist["train"].append(train_loss/len(train_dataset))
        hist["val"].append(val_loss/len(val_dataset))
        hist["val_clean"].append(((gt-clean_predictions)**2).mean())
        model.save(args.resume_path, epoch=epoch, hist=hist, args=args, arg_hist=arg_hist)


def inference_main(args):
    raise NotImplementedError("Inference not implemented yet")





try:
    args, _ = fargv.fargv(p)
except KeyError:
    args, _ = fargv.fargv(p, argv=["ipython"])

if args.input_class == "full":
    args.class_glob_regex = "[0-9a-f]+\.seals.crops/[0-9]+\.resolution\.gt\.json"
    args.class_glob_replace = ".img.*"


if __name__ == "__main__":
    if args.mode == "train":
        train_main(args)
    elif args.mode == "inference":
        inference_main(args)