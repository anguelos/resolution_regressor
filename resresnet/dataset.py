import glob
import json
import re
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import fargv
import random


def create_cropping_transform(image_size):
    return transforms.Compose([
        transforms.RandomCrop((image_size, image_size), pad_if_needed=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


simple_input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ResolutionDataset():
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


    def __init__(self, gt_file_list, gt_glob="",
                 input_glob_subtuple=("[0-9]+\.resolution\.gt\.json", "*.Img_Writ*.jpg"),
                 input_transform=simple_input_transform) -> None:
        assert len(gt_file_list) == 0 or gt_glob == ""
        if len(gt_file_list) == 0:
            fargv.warn(f"Loading glob {gt_glob}",3)
            gt_file_list = list(glob.glob(gt_glob))
            fargv.warn(f"Loaded {len(gt_file_list)} files from glob",2)
        else:
            assert gt_glob == ""
            fargv.warn(f"Loaded {len(gt_file_list)} files",2)
        gt_file_list=sorted(gt_file_list)
        self.ppcms = []
        self.image_paths = []
        self.gt_file_list = []
        for gt_file in gt_file_list:
            ppcm, _ = ResolutionDataset.analyse_card(gt_file)
            if ppcm > 0:
                images_glob = re.sub(input_glob_subtuple[0], input_glob_subtuple[1],gt_file)
                images = sorted(glob.glob(images_glob))
                if len(images) != 1:
                    fargv.warn(f"Multiple images for {gt_file}", 2)
                if len(images) == 0:
                    fargv.warn(f"Skipping {gt_file} because it has no image", 2)
                    continue
                self.ppcms.append(ppcm)
                self.image_paths.append(images[0])
                self.gt_file_list.append(gt_file)
            else:
                fargv.warn(f"Skipping {gt_file} because it has no valid ppcm")
        self.input_glob_subtuple = input_glob_subtuple
        self.input_transform = input_transform


    def __getitem__(self, n):
        return self.input_transform(Image.open(self.image_paths[n])), self.ppcms[n]
    

    def __len__(self):
        return len(self.ppcms)


    def __str__(self):
        return f"ResolutionDataset with {len(self)} samples"


    def __repr__(self):
        return f"""ResolutionDataset(gt_file_list={self.gt_file_list},
    input_glob_subtuple={repr(self.input_glob_subtuple)},
    input_transform={repr(self.input_transform)})"""

    def split(self, fraction, shuffle=True, seed=None):
        indexes = list(range(len(self)))
        if shuffle:
            if seed is not None:
                random.Random(seed).shuffle(indexes)
            else:
                random.shuffle(indexes)
        ds1 = ResolutionDataset([], input_glob_subtuple=self.input_glob_subtuple, input_transform=self.input_transform)
        ds2 = ResolutionDataset([], input_glob_subtuple=self.input_glob_subtuple, input_transform=self.input_transform)
        ds1.ppcms = [self.ppcms[i] for i in indexes[:int(len(indexes)*fraction)]]
        ds1.image_paths = [self.image_paths[i] for i in indexes[:int(len(indexes)*fraction)]]
        ds1.gt_file_list = [self.gt_file_list[i] for i in indexes[:int(len(indexes)*fraction)]]
        ds2.ppcms = [self.ppcms[i] for i in indexes[int(len(indexes)*fraction):]]
        ds2.image_paths = [self.image_paths[i] for i in indexes[int(len(indexes)*fraction):]]
        ds2.gt_file_list = [self.gt_file_list[i] for i in indexes[int(len(indexes)*fraction):]]
        return ds1, ds2