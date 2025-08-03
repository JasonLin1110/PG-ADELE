import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import unet_model

def standarize(img):
    return (img - img.mean()) / img.std()

class ImageDataset(Dataset):
    def __init__(self, data_root, image_paths, data_type='train', device="cpu"):

        self.device = device
        self.image_paths = image_paths
        self.data_type = data_type
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.root = data_root


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.root+os.sep+self.image_paths[index]
        image = (cv2.imread(img_path, cv2.IMREAD_UNCHANGED)).astype("uint16")
        clean_mask_path = img_path.replace("images", "masks")
        clean_mask = (cv2.imread(clean_mask_path, cv2.IMREAD_UNCHANGED)).astype("uint8")
        img_pils = Image.fromarray(image)
        image_torch = standarize(self.transform(img_pils).float())
        return image_torch, clean_mask, img_path


def caculate_IoU(c_num, preds, labels, hist):
    numpy_preds = preds.cpu().numpy()
    numpy_labels = labels.cpu().numpy()
    for c in range(c_num):
        hist[c,0] += (numpy_preds==c).sum()
        hist[c,1] += (numpy_labels==c).sum()
        hist[c,2] += ((numpy_labels==c)&(numpy_preds==c)).sum()
    return hist

parser = argparse.ArgumentParser(description="SegTHOR 2020 Challange")
parser.add_argument("--data_dir", type=str, default="datasets/SegThor/train")
parser.add_argument("--model_dir", type=str, default="PGADELE")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--class_num", type=int, default=5)
parser.add_argument("--channel", type=int, default=1)
args = parser.parse_args()

parameters = {}
parameters["data_dir"] = args.data_dir
parameters["model_dir"] = args.model_dir
parameters['device'] = args.device
parameters["batch_size"] = args.batch_size
parameters["class_num"] = args.class_num
parameters["channel"] = args.channel

device = parameters['device'] if torch.cuda.is_available() else "cpu"

test_path = 'test.txt'
with open(test_path, 'r') as file:
    lines = file.readlines()
ts_dl = [line.strip() for line in lines]

dataset = ImageDataset(parameters["data_dir"], img_path=ts_dl, data_type='valid', device=device)
test_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=5,#48
        num_workers=1,
        pin_memory=True,
    )

model_a = unet_model.UNet(n_channels=parameters["channel"], n_classes=parameters["class_num"])

sd = torch.load(os.path.join(parameters["model_dir"]+os.sep+"model_50.pth"), map_location="cpu", weights_only=True)
model_a.load_state_dict(sd, strict=False)
model_a = model_a.cuda()


bestmodel_a = unet_model.UNet(n_channels=1, n_classes=5)
bestsd = torch.load(os.path.join(parameters["model_dir"]+os.sep+'best'+os.sep+"model.pth"), map_location="cpu", weights_only=True)
bestmodel_a.load_state_dict(bestsd, strict=False)
bestmodel_a = bestmodel_a.cuda()


epoch_num=1

model_a.eval()
bestmodel_a.eval()
hist1 = np.zeros((parameters["class_num"], 3))
hist2 = np.zeros((parameters["class_num"], 3))
with torch.no_grad():
    for images, labels, label_path in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.long()
        _, logit1 = model_a(images)
        prob_a = torch.softmax(logit1, dim=1)
        small_images = F.interpolate(images, scale_factor=0.7, mode="bilinear", align_corners=True, recompute_scale_factor=True)
        _, small_logits = model_a(small_images)
        small_logits = F.interpolate(small_logits, size=(256, 256), mode='bilinear', align_corners=True)
        small_prob_a = torch.softmax(small_logits, dim=1)
        large_images = F.interpolate(images, scale_factor=1.5, mode="bilinear", align_corners=True, recompute_scale_factor=True)
        _, large_logits = model_a(large_images)
        large_logits = F.interpolate(large_logits, size=(256, 256), mode='bilinear', align_corners=True)
        large_prob_a = torch.softmax(large_logits, dim=1)
        total_prob_a = (prob_a+small_prob_a+large_prob_a)/3.0
        merge_preds = torch.argmax((total_prob_a), dim=1)

        hist1 = caculate_IoU(parameters["class_num"], merge_preds.to(torch.int), labels.to(torch.int), hist1)

        _, bestlogit1 = bestmodel_a(images)
        bestprob_a = torch.softmax(bestlogit1, dim=1)
        _, bestsmall_logits = bestmodel_a(small_images)
        bestsmall_logits = F.interpolate(bestsmall_logits, size=(256, 256), mode='bilinear', align_corners=True)
        bestsmall_prob_a = torch.softmax(bestsmall_logits, dim=1)
        _, bestlarge_logits = bestmodel_a(large_images)
        bestlarge_logits = F.interpolate(bestlarge_logits, size=(256, 256), mode='bilinear', align_corners=True)
        bestlarge_prob_a = torch.softmax(bestlarge_logits, dim=1)
        besttotal_prob_a = (bestprob_a+bestsmall_prob_a+bestlarge_prob_a)/3.0
        bestmerge_preds = torch.argmax((besttotal_prob_a), dim=1)

        hist2 = caculate_IoU(parameters["class_num"], bestmerge_preds.to(torch.int), labels.to(torch.int), hist2)
all_iou1 = (hist1[:,2])/(hist1[:,0]+hist1[:,1]-hist1[:,2])
miou1 = np.nanmean(all_iou1)
all_iou2 = (hist2[:,2])/(hist2[:,0]+hist2[:,1]-hist2[:,2])
miou2 = np.nanmean(all_iou2)
print('Last Epoch test data')
print("IoU: ", all_iou1)
print('mIoU: ', miou1)
print('Best Epoch test data')
print("IoU: ", all_iou2)
print('mIoU: ', miou2)