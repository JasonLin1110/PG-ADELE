import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
from PIL import Image
import cv2
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import torch.nn.functional as F
from torch.utils.data import DataLoader
import unet_model
from libs import Loss, Prototype, ADELE

import warnings
warnings.simplefilter("ignore", UserWarning)

def standarize(img):
    return (img - img.mean()) / img.std()

def standarize(img):
    return (img - img.mean(dim=(1,2), keepdim=True)) / img.std(dim=(1,2), keepdim=True)

class ImageDataset(Dataset):
    def __init__(self, parameters, image_dir, image_paths, data_type='train', device="cpu"):

        self.device = device
        self.data_type = data_type
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dir = image_dir
        self.image_paths = image_paths
        self.data_dir = parameters["data_dir"]
        self.correct_dir = parameters["correct_dir"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_name = os.path.join(self.dir, self.image_paths[index])
        image = (cv2.imread(img_name, cv2.IMREAD_UNCHANGED)).astype("uint8")
        label_name = os.path.splitext(self.image_paths[index])[0] + '_segmentation.png'
        noisy_name = os.path.join(self.correct_dir, label_name)

        if self.data_type == 'train':
            if os.path.isfile(noisy_name):
                mask = (cv2.imread(noisy_name, cv2.IMREAD_UNCHANGED)).astype("uint8")
            else:
                mask_path = os.path.join((self.data_dir+os.sep+"Training_noisy_resize"), label_name)
                mask = (cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)).astype("uint8")
        else:
            label_dir = (self.dir).replace("_resize", "_GT_resize")
            mask_path = os.path.join(label_dir, label_name)
            mask = (cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)).astype("uint8")
        clean_label_dir = (self.dir).replace("_resize", "_GT_resize")
        clean_mask_path = os.path.join(clean_label_dir, label_name)
        clean_mask = (cv2.imread(clean_mask_path, cv2.IMREAD_UNCHANGED)).astype("uint8")
        img_pils = Image.fromarray(image)
        image_torch = standarize(self.transform(img_pils).float())
        mask = (mask == 255).astype(np.uint8)
        clean_mask = (clean_mask == 255).astype(np.uint8)
        return image_torch, mask, noisy_name, clean_mask

def caculate_IoU(preds, labels, hist,c_num):
    numpy_preds = preds.cpu().numpy()
    numpy_labels = labels.cpu().numpy()
    for c in range(c_num):
        hist[c,0] += (numpy_preds==c).sum()
        hist[c,1] += (numpy_labels==c).sum()
        hist[c,2] += ((numpy_labels==c)&(numpy_preds==c)).sum()
    return hist

def train_model(parameters, dir='model', describe=''):
    if not os.path.exists(parameters["save_dir"]):
        os.makedirs(parameters["save_dir"])
        os.makedirs(parameters["save_dir"]+os.sep+'best')
    with open(parameters["save_dir"]+os.sep+"describe.txt", "w") as file:
        file.write(describe)
    best_miou=0
    best_epoch=0

    model = unet_model.UNet(n_channels=parameters["channel"], n_classes=parameters["class_num"])
    past_model = unet_model.UNet(n_channels=parameters["channel"], n_classes=parameters["class_num"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    torch.cuda.manual_seed(parameters["seed"])
    np.random.seed(parameters["seed"])
    device = torch.device(parameters["device"] if torch.cuda.is_available() else "cpu")

    train_dir = parameters["data_dir"]+os.sep+'Training_resize'
    tr_dl = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    valid_dir = parameters["data_dir"]+os.sep+'Validation_resize'
    val_dl = sorted([f for f in os.listdir(valid_dir) if f.endswith('.jpg')])

    dataset = ImageDataset(parameters, image_dir = train_dir, image_paths=tr_dl, data_type='train', device=device)
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=parameters['batch_size'],
        num_workers=1,
        pin_memory=True,
    )
    val_dataset = ImageDataset(parameters, image_dir = valid_dir, image_paths=val_dl, data_type='valid', device=device)
    valid_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=parameters['batch_size'],
        num_workers=1,
        pin_memory=True,
    )

    model = model.cuda()
    past_model = past_model.cuda()

    with torch.no_grad():
        class_features=Prototype.inital(model, train_loader, parameters,  max_samples_per_class = [1000000, 1000000]).cuda()
        past_class_features=class_features.detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=parameters["lr"], momentum=parameters["momentum"], weight_decay=parameters["weight_decay"])
    past_optimizer = torch.optim.SGD(past_model.parameters(), lr=parameters["lr"], momentum=parameters["momentum"], weight_decay=parameters["weight_decay"])
    past_model.load_state_dict(model.state_dict())
    past_class_features = class_features.detach().clone()
    past_optimizer.load_state_dict(optimizer.state_dict())

    class_iou_records=[]
    valid_record=[]
    correct_record=[]
    need_label_correction_dict = {0:True, 1:False}
    already_label_correction_dict = {0:True, 1:False}
    first_correct_dict = {0:0, 1:0}

    now_epoch=0

    # train + valid
    model_path = 'model.pth'
    past_model.eval()
    for epoch in tqdm(range(parameters["epochs"])):
        now_epoch+=1
        train_hist = np.zeros((parameters["class_num"], 3), dtype=np.uint64)
        correct_hist = np.zeros((parameters["class_num"],3), dtype=np.uint64)
        past_model.eval()
        past_model.load_state_dict(model.state_dict())
        past_class_features = class_features.detach().clone()
        past_optimizer.load_state_dict(optimizer.state_dict())
        model.train()
        print('Epoch ', now_epoch)
        for images, masks, _, clean_masks in tqdm(train_loader):
            images, masks, clean_masks = images.cuda(), masks.cuda(), clean_masks.cuda()
            masks = masks.long()
            clean_masks = clean_masks.long()
            o_features1, logits = model(images)
            small_images = F.interpolate(images, scale_factor=0.7, mode="bilinear", align_corners=True, recompute_scale_factor=True)
            small_features1, small_logits = model(small_images)
            small_logits = F.interpolate(small_logits, size=(256, 256), mode='bilinear', align_corners=True)
            small_features1 = F.interpolate(small_features1, size=(256, 256), mode='bilinear', align_corners=True)
            large_images = F.interpolate(images, scale_factor=1.5, mode="bilinear", align_corners=True, recompute_scale_factor=True)
            large_features1, large_logits = model(large_images)
            large_logits = F.interpolate(large_logits, size=(256, 256), mode='bilinear', align_corners=True)
            large_features1 = F.interpolate(large_features1, size=(256, 256), mode='bilinear', align_corners=True)
            features1 = (o_features1+small_features1+large_features1)/3.0

            prob = torch.softmax(logits, dim=1)
            small_prob = torch.softmax(small_logits, dim=1)
            large_prob = torch.softmax(large_logits, dim=1)
            total_prob = (prob+small_prob+large_prob)/3.0
            total_prob = torch.clamp(total_prob, 1e-3, 1-1e-3)
            with torch.no_grad():
                proto_ids = Prototype.distribute(features1, class_features, masks, parameters=parameters)
            loss = Loss.all_loss(small_logits,logits,large_logits, masks, prob, small_prob, large_prob, total_prob, features1, class_features, proto_ids, parameters)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            with torch.no_grad():
                class_features = Prototype.update(features1, class_features, masks, proto_ids, parameters=parameters)
                merge_pred = torch.argmax(total_prob, dim=1)
                correct_hist = caculate_IoU(masks.to(torch.int), clean_masks.to(torch.int), correct_hist, parameters["class_num"])
                train_hist = caculate_IoU(merge_pred.to(torch.int), masks.to(torch.int), train_hist, parameters["class_num"])

        model.eval()
        with torch.no_grad():
            all_iou = (train_hist[:,2])/(train_hist[:,0]+train_hist[:,1]-train_hist[:,2])
            miou = np.nanmean(all_iou)
            print('Train  : ', all_iou, ', miou: ', miou)
            class_iou_records.append(all_iou.tolist())
            iou_history = np.array(class_iou_records)

            all_iou = (correct_hist[:,2])/(correct_hist[:,0]+correct_hist[:,1]-correct_hist[:,2])
            miou = np.nanmean(all_iou)
            print('Correct: ', all_iou, ', miou: ', miou)
            correct_record.append([miou]+ all_iou.tolist())
            if (now_epoch)%parameters["correct_freq"]==0:
                already_label_correction_dict={0:True, 1:False}
            for c in [1]:
                class_correct = ADELE.if_update(iou_history[:,c], now_epoch-1, n_epoch=parameters["epochs"], threshold=parameters["r"])
                need_label_correction_dict[c]=class_correct
                if already_label_correction_dict[c]:
                    need_label_correction_dict[c]=False
            val_hist = np.zeros((parameters["class_num"], 3), dtype=np.uint64)
            for images, labels, _, _ in (valid_loader):
                images, labels = images.cuda(), labels.cuda()
                _, logit1 = model(images)
                prob = torch.softmax(logit1, dim=1)
                small_images = F.interpolate(images, scale_factor=0.7, mode="bilinear", align_corners=True, recompute_scale_factor=True)
                _, small_logits = model(small_images)
                small_logits = F.interpolate(small_logits, size=(256, 256), mode='bilinear', align_corners=True)
                small_prob = torch.softmax(small_logits, dim=1)
                large_images = F.interpolate(images, scale_factor=1.5, mode="bilinear", align_corners=True, recompute_scale_factor=True)
                _, large_logits = model(large_images)
                large_logits = F.interpolate(large_logits, size=(256, 256), mode='bilinear', align_corners=True)
                large_prob = torch.softmax(large_logits, dim=1)
                total_prob = (prob+small_prob+large_prob)/3.0
                merge_preds = torch.argmax((total_prob), dim=1)
                val_hist = caculate_IoU(merge_preds.to(torch.int), labels.to(torch.int), val_hist, parameters["class_num"])
            all_iou = (val_hist[:,2])/(val_hist[:,0]+val_hist[:,1]-val_hist[:,2])
            miou = np.nanmean(all_iou)
            valid_record.append([miou]+ all_iou.tolist())
            print('Valid  : ', all_iou, ', miou: ', miou)
            if (now_epoch==parameters["epochs"]):
                torch.save(model.state_dict(), parameters["save_dir"]+os.sep+"model_"+str(now_epoch)+".pth")
                torch.save(class_features, parameters["save_dir"]+os.sep+"prototype_"+str(now_epoch)+".pth")
            if miou>best_miou:
                best_miou=miou
                best_epoch=now_epoch
                torch.save(model.state_dict(), parameters["save_dir"]+os.sep+'best'+os.sep+model_path)
                torch.save(class_features, parameters["save_dir"]+os.sep+'best'+os.sep+"prototype.pth")
            print('best_epoch: ', best_epoch, ", best_miou = ", best_miou)

        if any(need_label_correction_dict[key] for key in need_label_correction_dict if key != 0) and now_epoch!=parameters["epochs"]:
            past_model.train()
            for images, masks, img_path, clean_masks in (train_loader):
                images = images.cuda()
                mask_shape = masks.shape
                masks, clean_masks = masks.cuda(), clean_masks.cuda()
                masks = masks.long()

                o_features1, logits = past_model(images)
                small_images = F.interpolate(images, scale_factor=0.7, mode="bilinear", align_corners=True, recompute_scale_factor=True)
                small_features1, small_logits = past_model(small_images)
                small_logits = F.interpolate(small_logits, size=(256, 256), mode='bilinear', align_corners=True)
                small_features1 = F.interpolate(small_features1, size=(256, 256), mode='bilinear', align_corners=True)
                large_images = F.interpolate(images, scale_factor=1.5, mode="bilinear", align_corners=True, recompute_scale_factor=True)
                large_features1, large_logits = past_model(large_images)
                large_logits = F.interpolate(large_logits, size=(256, 256), mode='bilinear', align_corners=True)
                large_features1 = F.interpolate(large_features1, size=(256, 256), mode='bilinear', align_corners=True)
                features1 = (o_features1+small_features1+large_features1)/3.0

                prob = torch.softmax(logits, dim=1)
                small_prob = torch.softmax(small_logits, dim=1)
                large_prob = torch.softmax(large_logits, dim=1)
                total_prob = (prob+small_prob+large_prob)/3.0
                total_prob = torch.clamp(total_prob, 1e-3, 1-1e-3)

                with torch.no_grad():
                    proto_ids = Prototype.distribute(features1, past_class_features, masks, parameters=parameters)
                loss = Loss.all_loss(small_logits,logits,large_logits, masks, prob, small_prob, large_prob, total_prob, features1, past_class_features, proto_ids, parameters)
                past_optimizer.zero_grad()
                loss.backward()
                past_optimizer.step()
                
                with torch.no_grad():
                    pred = torch.argmax(total_prob, dim=1)
                    mask = torch.zeros((mask_shape[0],256,256), dtype=torch.bool).cuda()
                    if need_label_correction_dict[1]:
                        label_belong_correct = (masks==0)
                        after_belong = (pred == 1)
                        confident = Prototype.dist_mask(total_prob, past_class_features, features1, 1, 0, masks)
                        mask |= (confident & label_belong_correct & after_belong)
                        label_belong_correct = (masks==1)
                        after_belong = (pred == 0)
                        confident = Prototype.ist_mask(total_prob, past_class_features, features1, 0, 1, masks)
                        mask |= ((confident) & label_belong_correct & after_belong)
                    new_labels = torch.where(mask, pred, masks)
                    correct_hist = caculate_IoU(new_labels.to(torch.int), clean_masks.to(torch.int), correct_hist, parameters["class_num"])
                    for i in range(mask_shape[0]):
                        array = (new_labels[i].cpu().numpy()*255).astype(np.uint8)
                        image = Image.fromarray(array, mode="L")
                        directory = os.path.dirname(img_path[i])
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        image.save(img_path[i])

                with torch.no_grad():
                    past_class_features = Prototype.update(features1, past_class_features, masks, proto_ids, parameters=parameters)

        for c in [1]:
            if need_label_correction_dict[c]==True:
                already_label_correction_dict[c]=True
                if first_correct_dict[c]==0:
                    first_correct_dict[c]=now_epoch
        
        print('correct: ', need_label_correction_dict)
        print('first: ', first_correct_dict)
    print("-----------------------------------------------------------------")
    print('valid best_epoch: ', best_epoch)
    print('valid best_mIoU : ', best_miou)
    print(first_correct_dict)
    with open(parameters["save_dir"]+os.sep+"describe.txt", "w") as file:
        file.write(describe+'\n')
        file.write('valid best_epoch: '+str(best_epoch)+', best_mIoU : '+str(best_miou)+'\n')
        file.write('valid last epoch: '+', mIoU = '+str(miou)+'\n')
    valid_history = np.array(valid_record)
    correct_history = np.array(correct_record)
    csv_name = parameters["save_dir"]+os.sep+'valid_IoU.csv'
    np.savetxt(csv_name, valid_history, delimiter=",", fmt="%.6f")
    csv_name = parameters["save_dir"]+os.sep+'Correct_IoU.csv'
    np.savetxt(csv_name, correct_history, delimiter=",", fmt="%.6f")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2017 Challange")
    parser.add_argument("--data_dir", type=str, default="ISIC2017")
    parser.add_argument("--correct_dir", type=str, default="correct_ISIC")
    parser.add_argument("--save_dir", type=str, default="PGADELE_ISIC")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--number_epochs", type=int, default=50)
    parser.add_argument("--correct_freq", type=int, default=10)
    parser.add_argument("--class_num", type=int, default=2)
    parser.add_argument("--channel", type=int, default=3)

    # ADELE + label correction 
    parser.add_argument("--beta_fg", type=float, default=0.95, help="confidence threshold for the foreground")
    parser.add_argument("--beta_bg", type=float, default=0.95, help="confidence threshold for the background")
    parser.add_argument("--delta", type=float, default=0, help="similarity threshold(-1~1)")
    parser.add_argument("--r", type=float, default=0.9, help="the r for the label correction")
    parser.add_argument("--effect_learn", type=float, default=0.01, help="the effect leraning threshold")

    # prototype set
    parser.add_argument("--p_num", type=int, default=5, help="prototypes number of each class")

    # loss parameters
    parser.add_argument("--rho", type=float, default=0.8, help='the threshold when select the target for JSD')
    parser.add_argument("--alpha", type=float, default=0.01, help='loss weight for prototype learning (HAPMC+PPA)')
    parser.add_argument("--m", type=float, default=20, help='HAPMC positive margin (degree)')
    parser.add_argument("--n_m", type=float, default=0.5, help='HAPMC negative margin (0~1)')
    parser.add_argument("--tau", type=float, default=0.01, help='HAPMC sharp')
    parser.add_argument("--eps", type=float, default=0.0, help='hard prototype selection threshold')

    args = parser.parse_args()

    parameters = {}
    parameters["data_dir"] = args.data_dir
    parameters["correct_dir"] = args.correct_dir
    parameters["save_dir"] = args.save_dir
    parameters["seed"] = args.seed
    parameters['device'] = args.device
    parameters["batch_size"] = args.batch_size
    parameters["lr"] = args.lr
    parameters["momentum"] = args.momentum
    parameters["weight_decay"] = args.weight_decay
    parameters["epochs"] = args.number_epochs
    parameters["class_num"] = args.class_num
    parameters["channel"] = args.channel

    parameters["effect_learn"] = args.effect_learn
    parameters["r"] = args.r
    parameters["conf_threshold"] = args.beta_fg
    parameters["conf_threshold_bg"] = args.beta_bg
    parameters["similarity_threshold"] = args.delta
    parameters["correct_freq"] = args.correct_freq

    parameters["p_num"] = args.p_num

    parameters["rho"] = args.rho
    parameters["aplha"] = args.alpha
    parameters["m"] = args.m
    parameters["n_m"] = args.n_m
    parameters["tau"] = args.tau
    parameters["eps"] = args.eps
    train_model(parameters, describe='')