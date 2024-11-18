import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import os
from Different_WeightsNET import FreWeightNet
from torchvision import transforms
from Data_Loader import load_model
import pandas as pd


fre_weight = 0.5
model = FreWeightNet(FreWeight=fre_weight)
state_dict_path = f'../model/Experiment/DWNs_{fre_weight}_ckpt.pt'
model = load_model(model, state_dict_path)
device = 'cuda:0'
model.to(device)
model.eval()

# transforms_img = transforms.Compose([
#             transforms.CenterCrop(size=(256, 256)),
#             Extract_texture(patch_size=32),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
transforms_img = transforms.Compose([
    transforms.CenterCrop(256),
    # Extract_texture(patch_size=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def save_results_to_csv(results, csv_file):
    rows = []
    for result in results:
        cm = result['confusion_matrix']
        cm_flatten = cm.flatten()
        row = [result['dataset'], result['accuracy']] + cm_flatten.tolist()
        rows.append(row)

    # 根据混淆矩阵的尺寸动态生成列名
    num_classes = int(len(rows[0]) - 2) // 4
    columns = ['dataset', 'accuracy'] + [f'cm_{i}{j}' for i in range(num_classes) for j in range(num_classes)]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(csv_file, index=False)


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
            if os.path.isdir(label_folder):
                label = int(label_folder_name[0])
                image_files = [os.path.join(label_folder, f) for f in os.listdir(label_folder)]
                for image_file in image_files:
                    y_label = AI_Draw(image_file)
                    y_label = y_label.cpu()
                    predicted_labels.append(y_label)
                    true_labels.append(label)

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'folder_name:{image_folder}, acc:{accuracy}')
    return accuracy


acc_list = []
flag = 1
file_path = '../test'
for folder_test in os.listdir(file_path):
    folder_test = os.path.join(file_path, folder_test)
    flag = 1
    for image_folders in os.listdir(folder_test):
        # 二级目录时, image_folder是标签, 否则是类别
        if flag:
            image_folders = os.path.join(folder_test, image_folders)
            for image_or_folder in os.listdir(image_folders):
                # 二级目录
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
print(sum(acc_list) / len(acc_list))
