import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
import numpy as np
import fargv


class ResResNet(nn.Module):
    @staticmethod
    def resume(path, model_arch, device):
        model = ResResNet(variant=model_arch, pretrained=True)
        if not torch.cuda.is_available():
            device = "cpu"
        try:
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            del checkpoint["model_state_dict"]
            epoch = checkpoint["epoch"]
            hist = checkpoint["hist"]
            arg_hist = checkpoint["arg_hist"]
        except  FileNotFoundError as e:
            fargv.warn(f"Could not resume {model_arch} from {path}: {e}",verbose=1)
            hist = {"train": [], "val": [], "val_clean": []}
            arg_hist = {}
            epoch = 0
            checkpoint = {"hist": hist, "arg_hist": arg_hist, "epoch": epoch}
        model = model.to(device)
        return model, checkpoint, (epoch, hist, arg_hist)


    @staticmethod
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


    def __init__(self, variant="resnet18", pretrained=True):
        super(ResResNet, self).__init__()
        if variant == "resnet18":
            self.resnet = resnet18(pretrained=pretrained)
            self.fc1 = nn.Linear(512, 256)
        elif variant == "resnet34":
            self.resnet = resnet34(pretrained=pretrained)
            self.fc1 = nn.Linear(512, 256)
        elif variant == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            self.fc1 = nn.Linear(2048, 256)
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


    def predict_img(self, img,size, outlier_quantile=.2):
        with torch.no_grad():
            batch = ResResNet.img_to_batch(img, size)
            predictions = self.forward(batch)
            predictions = predictions.cpu().numpy()
            predictions = predictions.reshape(-1)
            return np.median(predictions)
            if predictions.shape[0]*outlier_quantile > 1:
                outlier_threshold = np.quantile(predictions, 1-outlier_quantile)
                predictions = predictions[predictions < outlier_threshold]
            else:
                print(f"Warning: too few predictions({predictions}) to filter outliers img.size={img.size()}")
            return predictions.mean()
    

    def save(self, path, epoch, hist, args, arg_hist):
        arg_hist[epoch] = args
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "hist": hist,
            "arg_hist": arg_hist,
        }, path)