import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from Data_Augmentation import BlurAndResize, Extract_texture, ConditionalCenterCrop, ConditionalPadding
from Data_Loader import RandomValSet
from torchkeras import KerasModel
from torchmetrics import Accuracy
from Experiment.FrequencyVsSpatial import Fre_v_Spa
from Experiment.MultiGPU_Trainer import AcceleratorTrainer

Frequency_model = Fre_v_Spa(is_Fre=True)

transforms_combined = transforms.Compose([
    BlurAndResize(jpeg_prob=0.1, blur_prob=0.1, resize_prob=0.1),
    Extract_texture(patch_size=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_test = transforms.Compose([
    ConditionalPadding(256),
    Extract_texture(patch_size=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_path = '../data_set/train_10k'
val_path = '../test'
data_train = ImageFolder(train_path, transforms_combined)
data_val = RandomValSet(val_path, 0.05, transforms_test)
dl_train = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
dl_val = torch.utils.data.DataLoader(data_val, batch_size=64, shuffle=True)
device = torch.device('cuda:1')
loss = torch.nn.CrossEntropyLoss()
metrics_dict = {"acc": Accuracy(task='multiclass', num_classes=2)}
optimizer = torch.optim.Adam(Frequency_model.parameters(),
                             lr=3e-4)

trainer = AcceleratorTrainer(Frequency_model, optimizer, loss, 'Frequency_model')
trainer.fit(50, dl_train, device)