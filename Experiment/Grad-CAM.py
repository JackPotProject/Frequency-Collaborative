import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from Different_WeightsNET import FreWeightNet


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.feature_maps = None
        self.feature = None
        self.hook = self.model.classifier.layer4.register_full_backward_hook(self.save_gradient)
        self.hook1 = self.model.classifier.layer4.register_forward_hook(self.get_feature_map)

    def get_feature_map(self, module, input, output):
        self.feature_maps = output

    def save_gradient(self, module, gard_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        # 前向传播
        self.feature = self.model(x)[1]
        return self.model(x)[0]

    def backward(self, class_index):
        self.model.zero_grad()
        one_hot_output = torch.zeros((1, 2))
        one_hot_output[0][class_index] = 1
        one_hot_output = one_hot_output.requires_grad_(True)

        output = self.model.classifier(self.feature)
        loss = F.nll_loss(output, one_hot_output.argmax(dim=1))
        loss.backward(retain_graph=True)

    def generate_cam(self):
        """
        :return:cam heatmap (cv2)
        """
        gradients = self.gradients
        feature_maps = self.feature_maps

        weights = F.adaptive_avg_pool2d(gradients, 1)
        cam = torch.sum(weights * feature_maps, dim=1)

        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam[0]
        cam = cv2.normalize(cam, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return cam


def cv_center_crop(img, target_size=(256, 256)):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    crop_x = target_size[0] // 2
    crop_y = target_size[1] // 2
    cropped_img = img[center_y - crop_y:center_y + crop_y, center_x - crop_x:center_x + crop_x]

    return cropped_img


def show_cam_on_image(img, mask, alpha=0.9, threshold=0.1, lower_bound=0.8, upper_bound=0.3):
    """
    :param upper_bound: colormap_max
    :param lower_bound: colormap_min
    :param img: cv2.imread
    :param mask: cam_mask
    :param alpha: transparency
    :param threshold: heatmap_thresholds
    Returns the overlay image
    """
    mask = cv2.GaussianBlur(mask, (21, 21), sigmaX=5, sigmaY=5)

    non_zero_mask = mask[mask > 0]
    min_value = np.min(non_zero_mask)
    max_value = np.max(non_zero_mask)
    normalized_mask = np.zeros_like(mask)
    if max_value > min_value:
        normalized_mask[mask > 0] = (mask[mask > 0] - min_value) / (max_value - min_value)
    scaled_mask = lower_bound + (upper_bound - lower_bound) * normalized_mask
    heatmap_input = np.zeros_like(mask, dtype=np.uint8)
    heatmap_input[mask > 0] = np.uint8(255 * scaled_mask[mask > 0])

    heatmap = cv2.applyColorMap(heatmap_input, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap * (mask > threshold)[:, :, np.newaxis]
    cam = np.float32(img) + heatmap * alpha
    cam = cam / np.max(cam)
    return cam


def get_image_tensor(image_path):
    img = Image.open(image_path)
    transforms_test = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transforms_test(img)
    img_tensor_4d = img_tensor.unsqueeze(0)

    cv_image = cv2.imread(image_path)
    cvt_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    origin_image = np.float32(cvt_image) / 255.0
    origin_image = cv_center_crop(origin_image, (256, 256))
    return img_tensor_4d, origin_image


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = 0.7
model = FreWeightNet(FreWeight=model_name)
model.load_state_dict(torch.load(f'model_path'))
img_path = 'img_path'
input_tensor, original_image = get_image_tensor(img_path)
target_class = 1  # target class index

grad_cam = GradCAM(model)
result = grad_cam.forward(input_tensor)
grad_cam.backward(target_class)
cam = grad_cam.generate_cam()
cam_image = show_cam_on_image(original_image, cam)


plt.figure(figsize=(5, 5))
plt.imshow(cam_image)
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


plt.show()
