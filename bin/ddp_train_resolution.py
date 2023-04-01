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


class ResResNet(nn.Module):
    def __init__(self, variant="resnet18", pretrained=True):
        super(ResResNet, self).__init__()
        if variant == "resnet18":
            self.resnet = resnet18(pretrained=pretrained)
        elif variant == "resnet34":
            self.resnet = resnet34(pretrained=pretrained)
        elif variant == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


train_input_transform = transforms.Compose([
    transforms.RandomCrop((args.image_size, args.image_size), pad_if_needed=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


simple_input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def img_to_batch(img, size):
    if isinstance(size, int):
        size = (size, size)
    sub_images = []
    x_margin = (img.shape[1] % size[0])
    y_margin = (img.shape[2] % size[1])
    x_margin = random.randint(0, x_margin)
    y_margin = random.randint(0, y_margin)
    #print(f"x_margin={x_margin}, y_margin={y_margin}, img.shape={img.shape}")
    for x in range(x_margin + size[0], img.shape[1], size[0]):
        for y in range(y_margin + size[1], img.shape[2], size[1]):
            sub_images.append(img[None, :, x-size[0]:x, y-size[1]:y])
    if len(sub_images) == 0:
        sub_images.append(img[None,:,:,:])
    return torch.cat(sub_images, dim=0)


def predict_img(img,size, model, outlier_quantile=.2):
    batch = img_to_batch(img, size)
    with torch.no_grad():
        predictions = model(batch)
        predictions = predictions.cpu().numpy()
        predictions = predictions.reshape(-1)
        return np.median(predictions)
        if predictions.shape[0]*outlier_quantile > 1:
            outlier_threshold = np.quantile(predictions, 1-outlier_quantile)
            predictions = predictions[predictions < outlier_threshold]
        else:
            print(f"Warning: too few predictions({predictions}) to filter outliers img.size={img.size()}")
        return predictions.mean()


class ScaleDataset:
    @staticmethod
    def analyse_card(gt_path):
        d = json.load(open(gt_path,"r"))
        card_pixel_width, card_pixel_height = d["image_wh"]
        crude_ppcm = max([card_pixel_width, card_pixel_height]) / 20.3  # Todo (anguelos) measure this with precision
        ppcm_list = []
        for n in range(len(d["rect_LTRB"])):
            LTRB = np.array(d["rect_LTRB"][n])
            diagonal_in_pixels = ((LTRB[[2,3]] - LTRB[[0,1]]) ** 2).sum() ** .5
            class_name = d["class_names"][d["rect_classes"][n]]
            if class_name == "5cm":
                ppcm_list.append(diagonal_in_pixels * .2)
            if class_name == "1cm":
                ppcm_list.append(diagonal_in_pixels)        
            if class_name == "5in":
                ppcm_list.append(diagonal_in_pixels * .2 * 2.54)
            if class_name == "1in":
                ppcm_list.append(diagonal_in_pixels * 2.54)
        if len(ppcm_list):
            ppcm = sum(ppcm_list)/len(ppcm_list)
        else:
            ppcm = -1
        return ppcm, crude_ppcm

    def __init__(self, resolution_gt_list=[],
                resolution_gt_glob="",
                input_glob_subtuple=("",""), #(args.class_glob_regex, args.class_glob_replace),
                input_transform=simple_input_transform, 
                augment_scale_range=(1.,1.)
                ):
        """_summary_

        Args:
            resolution_gt_list (list, optional): _description_. Defaults to [].
            resolution_gt_glob (str, optional): An apropriate value would be "./data/1000_CVCharters/*/*/*/*.seals.crops/*.resolution.gt.json". Defaults to "".
            input_glob_subtuple (tuple, optional): _description_. Defaults to (args.class_glob_regex, args.class_glob_replace).
            input_transform (_type_, optional): _description_. Defaults to simple_input_transform.
            augment_scale_range (tuple, optional): _description_. Defaults to (1.,1.).
        """
        #.Img_WritableArea.jpg
        self.ppcms = []
        self.crude_ppcms = []
        self.image_paths = []
        if resolution_gt_glob != "":
            resolution_gt_list + list(glob.glob(resolution_gt_glob))
        for gt_file in resolution_gt_list:
            ppcm, _ = ScaleDataset.analyse_card(gt_file)
            if ppcm > 0:
                images_glob = re.sub(input_glob_subtuple[0], input_glob_subtuple[1],gt_file)
                images = glob.glob(images_glob)
                if len(images) == 1:
                    self.ppcms.append(ppcm)
                    self.image_paths.append(images[0])
        self.input_transform = input_transform
        self.augment_scale_range = augment_scale_range


    def __getitem__(self, n):
        img = Image.open(self.image_paths[n])
        #w, h = img.size
        #scale = random.uniform(*self.augment_scale_range)
        #img = img.resize((int(w*scale), int(h*scale)))
        #return self.input_transform(img), self.ppcms[n] * scale
        self.input_transform(img), self.ppcms[n]
    
    def __len__(self):
        return len(self.ppcms)
    
    def split(self, fraction=.8, shuffle=False):
        idx = list(range(len(self.ppcms)))
        if shuffle:
            random.shuffle(idx)
        split_idx = int(len(self.ppcms) * fraction)
        ds1 = ScaleDataset(resolution_gt_glob="", input_glob_subtuple=("", ""), input_transform=self.input_transform)
        ds1.crude_ppcms = [self.crude_ppcms[i] for i in idx[:split_idx]]
        ds1.ppcms = [self.ppcms[i] for i in idx[:split_idx]]
        ds1.image_paths=[self.image_paths[i] for i in idx[:split_idx]]
        ds2 = ScaleDataset(resolution_gt_glob="", input_glob_subtuple=("", ""), input_transform=self.input_transform)
        ds2.crude_ppcms = [self.crude_ppcms[i] for i in idx[split_idx:]]
        ds2.ppcms=[self.ppcms[i] for i in idx[split_idx:]]
        ds2.image_paths=[self.image_paths[i] for i in idx[split_idx:]]
        return ds1, ds2


def resume(path, model_arch, device):
    model = ResResNet(variant=model_arch, pretrained=False)
    if not torch.cuda.is_available():
        device = "cpu"
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    hist = checkpoint["hist"]
    arg_hist = checkpoint["arg_hist"]
    return model, epoch, hist, arg_hist


def save(path, model, epoch, hist, args, arg_hist=None):
    if arg_hist is None:
        arg_hist = {epoch: args}
    else:
        arg_hist[epoch]= args
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "hist": hist,
        "arg_hist":arg_hist
    }, path)


def train_main(args):
    device = args.device
    if args.gt_glob != "":
        gt_paths = list(sorted(set(args.gt_paths).union(set(glob.glob(args.gt_glob)))))
    train_gt_paths = gt_paths[:int(len(gt_paths)*args.train_fraction)]
    val_gt_paths = gt_paths[int(len(gt_paths)*args.train_fraction):]
    print(f"Training on {len(train_gt_paths)} samples, validating on {len(val_gt_paths)} samples")
    train_dataset = ScaleDataset(resolution_gt_list=train_gt_paths, input_glob_subtuple=(args.class_glob_regex, args.class_glob_replace), input_transform=train_input_transform, augment_scale_range=eval(args.augment_scale_range))
    val_dataset = ScaleDataset(resolution_gt_list=val_gt_paths, input_glob_subtuple=(args.class_glob_regex, args.class_glob_replace), input_transform=simple_input_transform, augment_scale_range=[1.,1.])

    #data = ScaleDataset()

    #train_dataset, val_dataset = data.split(fraction=.8, shuffle=True)
    train_dataset.input_transform = train_input_transform
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = ResResNet().to(device)

    try:
        model, epoch, hist, arg_hist = resume(args.resume_path, model, device)
        print(f"Resuming from epoch {epoch}")
    except Exception as e:
        print(f"Could not resume from {args.resume_path}: {e}")
        hist = {"train": [], "val": [], "val_clean": []}
        epoch = 0
        arg_hist = {epoch: args}


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
                clean_predictions.append(predict_img(inputs[0,:,:,:], size=args.image_size, model=model, outlier_quantile=args.outlier_quantile))
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
        save(args.resume_path, model, epoch, hist, arg_hist=arg_hist)

def inference_main(args):
    raise NotImplementedError("Inference not implemented yet")


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