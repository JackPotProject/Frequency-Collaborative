import torch
from PIL import Image
import random
from torchvision import transforms
import io
import numpy as np
from torchvision.transforms import functional as F


class ConditionalCenterCrop:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        if img.size[0] < self.size or img.size[1] < self.size:
            return transforms.CenterCrop(self.size)(img)
        return img


def jpeg_compression(img, quality_factor):
    img_io = io.BytesIO()
    img.save(img_io, format='jpeg', quality=quality_factor)
    img_io.seek(0)
    img = Image.open(img_io)
    return img


class BlurAndResize(object):
    def __init__(self, jpeg_prob=0.1, blur_prob=0.1, resize_prob=0.1, **kwargs):
        self.jpeg_prob = jpeg_prob
        self.blur_prob = blur_prob
        self.resize_prob = resize_prob

    def __call__(self, img):
        Padding = ConditionalPadding(256)
        if random.random() < self.jpeg_prob:
            quality = random.randint(70, 100)
            img = jpeg_compression(img, quality)

        if random.random() < self.blur_prob:
            sigma = random.uniform(0, 1)
            kernel_size = int(sigma * 7) | 1
            img = transforms.functional.gaussian_blur(img, kernel_size)

        if random.random() < self.resize_prob:
            scale_factor = random.uniform(0.25, 0.5)
            img = transforms.functional.resize(img, int(img.size[1] * scale_factor))
            img = Padding(img)
        return img


class test_BlurAndResize(object):
    def __init__(self, jpeg_prob, blur_prob, resize_prob, **kwargs):
        self.jpeg_prob = jpeg_prob
        self.blur_prob = blur_prob
        self.resize_prob = resize_prob

    def __call__(self, img):
        Padding = ConditionalPadding(256)
        if random.random() < self.jpeg_prob:
            img = jpeg_compression(img, 95)

        if random.random() < self.blur_prob:
            sigma = 1
            kernel_size = int(sigma * 7) | 1
            img = transforms.functional.gaussian_blur(img, kernel_size)

        if random.random() < self.resize_prob:
            scale_factor = 0.5
            img = transforms.functional.resize(img, int(img.size[1] * scale_factor))
            img = Padding(img)
        return img


class Extract_texture(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        weak_texture, strong_texture = self.extract_texture_regions(img, self.patch_size)
        if len(weak_texture.shape) == 3:
            concat_regions = np.concatenate((weak_texture, strong_texture), axis=0)
        else:
            concat_regions = np.concatenate((weak_texture, strong_texture), axis=1)
        pil_image = Image.fromarray(concat_regions.astype(np.uint8))
        return pil_image

    def ED(self, img):
        r1, r2 = img[:, 0:-1, :], img[:, 1::, :]
        r3, r4 = img[:, :, 0:-1], img[:, :, 1::]
        r5, r6 = img[:, 0:-1, 0:-1], img[:, 1::, 1::]
        r7, r8 = img[:, 0:-1, 1::], img[:, 1::, 0:-1]
        s1 = torch.sum(torch.abs(r1 - r2)).item()
        s2 = torch.sum(torch.abs(r3 - r4)).item()
        s3 = torch.sum(torch.abs(r5 - r6)).item()
        s4 = torch.sum(torch.abs(r7 - r8)).item()
        return s1 + s2 + s3 + s4

    def sort_regions(self, patches):
        patches_with_keys = [(patch[0], patch) for patch in patches]
        patches_with_keys.sort(key=lambda x: x[0], reverse=True)
        sorted_patches = [patch[1] for patch in patches_with_keys]
        return sorted_patches

    def extract_texture_regions(self, image, patch_size=32):
        image = np.array(image)
        h, w, c = image.shape
        sub_high, sub_wide = 128, 256
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size]
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    continue
                patch_tensor = torch.tensor(patch).permute(1, 2, 0)
                l_div = self.ED(patch_tensor)
                patches.append((l_div, patch))
                # patches.sort(key=lambda x: x[0], reverse=True)  # 出错行
                patches = self.sort_regions(patches)
        texture_size = (sub_high, sub_wide, c)
        weak_texture_image = np.zeros(texture_size, dtype=np.uint8)
        strong_texture_image = np.zeros(texture_size, dtype=np.uint8)

        num_strong_patches = 32
        weak_level, weak_vertical = 0, 0
        strong_level, strong_vertical = 0, 0

        strong_patches = patches[:num_strong_patches]
        weak_patches = patches[-1:-num_strong_patches - 1:-1]
        assert len(strong_patches) == 32 and len(weak_patches) == 32, \
            f'get weak_patches{len(weak_patches)}, strong_patches{len(strong_patches)}'

        for idx, (l_div, patch) in enumerate(weak_patches):
            """
            weak_tuxture
            """
            if weak_level >= sub_wide:
                weak_vertical += patch_size
                weak_level = 0
            weak_texture_image[weak_vertical:weak_vertical + patch_size, weak_level:weak_level + patch_size] = patch
            weak_level += patch_size

        for idx, (l_div, patch) in enumerate(strong_patches):
            """
            strong_texture
            """
            if strong_level >= sub_wide:
                strong_vertical += patch_size
                strong_level = 0
            strong_texture_image[strong_vertical:strong_vertical + patch_size,
                                 strong_level:strong_level + patch_size] = patch
            strong_level += patch_size
        return weak_texture_image, strong_texture_image


class ConditionalPadding:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        width, height = img.size
        new_width = max(self.size, ((width + 31) // 32) * 32)
        new_height = max(self.size, ((height + 31) // 32) * 32)
        padding_left = (new_width - width) // 2
        padding_right = new_width - width - padding_left
        padding_top = (new_height - height) // 2
        padding_bottom = new_height - height - padding_top

        img = transforms.functional.pad(img, (padding_left, padding_top, padding_right, padding_bottom),
                                        padding_mode='reflect')
        return img
