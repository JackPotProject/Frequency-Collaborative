import torch
from PIL import Image
from sklearn.metrics import accuracy_score
import os
from CSF import FCACoAttentionNet
from torchvision import transforms
from Data_Augmentation import Extract_texture, test_BlurAndResize


model = FCACoAttentionNet(2)
state_dict = torch.load('./model/9126CSF.pt')


def image_process(is_JPEG=False, is_DownSampling=False, is_Blur=False):
    if is_JPEG:
        img_transforms = transforms.Compose([
            test_BlurAndResize(jpeg_prob=1, blur_prob=0, resize_prob=0),
            transforms.CenterCrop(size=(256, 256)),
            Extract_texture(patch_size=32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif is_DownSampling:
        img_transforms = transforms.Compose([
            test_BlurAndResize(jpeg_prob=0, blur_prob=0, resize_prob=1),
            transforms.CenterCrop(size=(256, 256)),
            Extract_texture(patch_size=32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif is_Blur:
        img_transforms = transforms.Compose([
            test_BlurAndResize(jpeg_prob=0, blur_prob=1, resize_prob=0),
            transforms.CenterCrop(size=(256, 256)),
            Extract_texture(patch_size=32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        img_transforms = transforms.Compose([
            transforms.CenterCrop(size=(256, 256)),
            Extract_texture(patch_size=32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return img_transforms


model.load_state_dict(state_dict)
device = 'cuda:0'
model.to(device)
model.eval()
transforms_img = image_process()


def AI_Draw(save_path):
    img = Image.open(save_path)
    img = img.convert('RGB')
    with torch.no_grad():
        output = model(transforms_img(img)[None, ...].cuda()).softmax(axis=1)
        _, pre_label = torch.max(output, 1)
        return pre_label


def TwoLevels(image_folder):
    predicted_labels = []
    true_labels = []
    for folder_name in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder_name)
        if os.path.isdir(folder_path):
            label = int(folder_name[0])
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
            for image_file in image_files:
                y_label = AI_Draw(image_file)
                y_label = y_label.cpu()
                predicted_labels.append(y_label)
                true_labels.append(label)
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'folder_name:{image_folder}, acc:{accuracy}')
    return accuracy


def ThreeLevels(image_folder):
    predicted_labels = []
    true_labels = []
    for folder_name in os.listdir(image_folder):
        folder_path = os.path.join(image_folder, folder_name)
        for label_folder_name in os.listdir(folder_path):
            label_folder = os.path.join(folder_path, label_folder_name)
            # print(label_folder)
            if os.path.isdir(label_folder):
                label = int(label_folder_name[0])
                image_files = [os.path.join(label_folder, f) for f in os.listdir(label_folder)]
                for image_file in image_files:
                    # print(image_file)
                    y_label = AI_Draw(image_file)
                    y_label = y_label.cpu()
                    predicted_labels.append(y_label)
                    true_labels.append(label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'folder_name:{image_folder}, acc:{accuracy}')
    return accuracy


acc_list = []
flag = 1
file_path = 'your_test_folder_path'
for folder_test in os.listdir(file_path):
    folder_test = os.path.join(file_path, folder_test)
    flag = 1
    for image_folders in os.listdir(folder_test):
        if flag:
            image_folders = os.path.join(folder_test, image_folders)
            for image_or_folder in os.listdir(image_folders):
                image_or_folder = os.path.join(image_folders, image_or_folder)
                if not os.path.isdir(image_or_folder):
                    acc = TwoLevels(folder_test)
                    acc_list.append(acc)
                    flag = 0
                    break
                else:
                    acc = ThreeLevels(folder_test)
                    acc_list.append(acc)
                    flag = 0
                    break
print(f'average_acc: {sum(acc_list) / len(acc_list)}')
