import torch
from Experiment.Different_WeightsNET import FreWeightNet
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from Data_Augmentation import BlurAndResize, Extract_texture, ConditionalCenterCrop, ConditionalPadding
from Data_Loader import RandomValSet
from torchkeras import KerasModel
from torchmetrics import Accuracy

Fre_Weight = 0.5
model_name = f'DWNs_{Fre_Weight}'
DWNs = FreWeightNet(FreWeight=Fre_Weight)
transforms_combined = transforms.Compose([
    ConditionalPadding(256),
    BlurAndResize(jpeg_prob=0.1, blur_prob=0.1, resize_prob=0.1),
    # Extract_texture(patch_size=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transforms_test = transforms.Compose([
    transforms.CenterCrop(256),
    # Extract_texture(patch_size=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_path = '../data_set/train_10k'
val_path = '../test'
data_train = ImageFolder(train_path, transforms_combined)
data_val = RandomValSet(val_path, 0.05, transforms_test)
dl_train = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
dl_val = torch.utils.data.DataLoader(data_val, batch_size=64, shuffle=True)

loss = torch.nn.CrossEntropyLoss()
metrics_dict = {"acc": Accuracy(task='multiclass', num_classes=2)}
optimizer = torch.optim.Adam(DWNs.parameters(),
                             lr=3e-4)

keras_model = KerasModel(DWNs,
                         loss_fn=loss,
                         metrics_dict=metrics_dict,
                         optimizer=optimizer
                         )

df_history = keras_model.fit(train_data=dl_train,
                             val_data=dl_val,
                             epochs=25,
                             ckpt_path=f'{model_name}_ckpt.pt',
                             patience=20,
                             monitor="val_acc",
                             mode="max",
                             mixed_precision='no',
                             plot=True,
                             quiet=True
                             )

torch.save(DWNs.state_dict(), f'../model/Experiment/{model_name}.pt')
