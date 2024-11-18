import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from Data_Augmentation import BlurAndResize, Extract_texture, ConditionalCenterCrop, ConditionalPadding
from Data_Loader import RandomValSet
from torchkeras import KerasModel
from torchmetrics import Accuracy
from Experiment.FrequencyVsSpatial import Fre_v_Spa

Spatial_model = Fre_v_Spa()

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

loss = torch.nn.CrossEntropyLoss()
metrics_dict = {"acc": Accuracy(task='multiclass', num_classes=2)}
optimizer = torch.optim.Adam(Spatial_model.parameters(),
                             lr=3e-4)

# scheduler = CosineAnnealingLR(optimizer, T_max=30)
keras_model = KerasModel(Spatial_model,
                         loss_fn=loss,
                         metrics_dict=metrics_dict,
                         optimizer=optimizer
                         # scheduler=scheduler
                         )

df_history = keras_model.fit(train_data=dl_train,
                             val_data=dl_val,
                             epochs=50,
                             ckpt_path='spatial_ckpt.pt',
                             patience=20,
                             monitor="val_acc",
                             mode="max",
                             mixed_precision='no',
                             plot=True,
                             quiet=True
                             )

torch.save(Spatial_model.state_dict(), '../model/Experiment/Spatial.pt')
