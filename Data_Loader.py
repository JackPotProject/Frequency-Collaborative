import torch
from torch.utils.data.dataloader import default_collate
import random
import os
from PIL import Image
from torch import nn


class RandomValSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, sampling_ratio, transforms=None):
        self.data_dir = data_dir
        self.ratio = sampling_ratio
        self.images, self.labels = self._get_image_label()
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def _get_image_label(self):
        file_list = []
        label_list = []
        for root, dir, files in os.walk(self.data_dir):
            if root.endswith('0_real'):
                for file in files:
                    if random.random() <= self.ratio:
                        file_list.append(os.path.join(root, file))
                        label_list.append(0)
            if root.endswith('1_fake'):
                for file in files:
                    if random.random() <= self.ratio:
                        file_list.append(os.path.join(root, file))
                        label_list.append(1)
        return file_list, label_list


class ExperimentSampling(torch.utils.data.Dataset):
    def __init__(self, data_dir, real_dir, sampling_radio=0.1, transforms=None, random_sampling=False):
        self.root = data_dir
        self.real_dir = real_dir
        self.radio = sampling_radio
        self.random_sam = random_sampling
        self.image, self.labels = self._get_image()
        self.transforms = transforms

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def _get_image(self):
        file_list = []
        label_list = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if random.random() <= self.radio:
                    file_list.append(os.path.join(root, file))
                    label_list.append(1)

        fake_img_len = len(file_list)
        all_files = os.listdir(self.real_dir)
        if self.random_sam:
            image_files = random.sample(all_files, fake_img_len)
        else:
            image_files = all_files[:fake_img_len]

        for file in image_files:
            file_name = os.path.join(self.real_dir, file)
            file_list.append(file_name)
            label_list.append(0)

        return file_list, label_list


def load_model(Model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict_func = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    has_module_prefix = any(key.startswith('module.') for key in state_dict_func.keys())

    if has_module_prefix:
        print("Detected 'module.' prefix in state_dict keys. Adjusting keys...")
        new_state_dict = {key.replace('module.', ''): value for key, value in state_dict_func.items()}
    else:
        new_state_dict = state_dict_func

    try:
        Model.load_state_dict(new_state_dict)
        print("Model state dict loaded successfully.")
    except RuntimeError as e:
        if 'Error(s) in loading state_dict' in str(e):
            print("RuntimeError in loading state_dict, attempting to use nn.DataParallel.")
            Model = nn.DataParallel(Model)
            Model.load_state_dict(state_dict_func)
            print("Model state dict loaded successfully with nn.DataParallel.")
        else:
            raise e
    return Model
