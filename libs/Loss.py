import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
from . import Prototype as P
warnings.simplefilter("ignore", UserWarning)

def prototype_loss(features, prototypes, labels, proto_ids, parameters):
    tau=parameters["tau"]
    labels_flat = labels.view(-1)
    _,C,_,_ = features.shape
    features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
    
    all_cosine = F.cosine_similarity(features_flat.unsqueeze(1).unsqueeze(1), prototypes.unsqueeze(0), dim=3) #B*H*W, class_num, p_num
    indices = torch.arange(labels_flat.size(0), device=labels_flat.device)
    belong_cosine = all_cosine[indices, labels_flat, proto_ids]

    margin_rad = torch.tensor(parameters["m"] * np.pi / 180, device=belong_cosine.device)
    belong_cosine_with_margin = torch.cos(torch.acos(belong_cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) + margin_rad)
    all_cosine_with_n_margin = all_cosine+parameters["n_m"]
    all_cosine_with_n_margin[indices, labels_flat, proto_ids] = belong_cosine_with_margin

    #all_cosine
    all_cosine_mask = (all_cosine>parameters["eps"]) | (all_cosine>=belong_cosine.unsqueeze(1).unsqueeze(2))
    all_cosine_mask[indices, labels_flat, :] = False
    all_cosine_mask[indices, labels_flat, proto_ids]=True

    HAPMC_loss = (-torch.log(
        torch.exp(belong_cosine_with_margin/tau) /
        (torch.sum((torch.exp(all_cosine_with_n_margin/tau)*all_cosine_mask), dim=(1,2)))
    ))
    PPA_loss = ((1-belong_cosine)**2)
    return torch.mean(PPA_loss)+torch.mean(HAPMC_loss)

def dice_loss(prob1, prob2, prob3, target, eps=1e-6):
    target_onehot = F.one_hot(target, num_classes=prob1.shape[1])
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()
    intersection1 = (prob1 * target_onehot).sum(dim=(2,3))
    union1 = prob1.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    intersection2 = (prob2 * target_onehot).sum(dim=(2,3))
    union2 = prob2.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    intersection3 = (prob3 * target_onehot).sum(dim=(2,3))
    union3 = prob3.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    dice = (((2 * intersection1 + eps) / (union1 + eps)) +  ((2 * intersection2 + eps) / (union2 + eps)) + ((2 * intersection3 + eps) / (union3 + eps)))/3# [B, C]
    return 1 - dice.mean()

def all_loss(s_logits,logits,l_logits, masks, prob, s_prob, l_prob, total_prob, features, class_features, proto_ids, parameters):

    # Cross-Entropy
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criterion2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criterion3 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    CE_loss = torch.mean((criterion(logits/3.0, masks)+criterion2(s_logits/3.0, masks)+criterion3(l_logits/3.0, masks)))/3.0

    # Dice
    Dice_loss = dice_loss(F.softmax(logits/3.0, dim=1), F.softmax(s_logits/3.0, dim=1), F.softmax(l_logits/3.0, dim=1), masks)

    # Kullback-Divergence
    max_prob, _ = total_prob.max(dim=1)
    total_mask = max_prob.ge(parameters["r"]).float().unsqueeze(1)
    kl_loss = torch.mean(torch.sum(F.kl_div(total_prob.log(), torch.clamp(prob, 1e-3, 1-1e-3), reduction='none') * total_mask, dim=1)\
            + torch.sum(F.kl_div(total_prob.log(), torch.clamp(s_prob, 1e-3, 1-1e-3), reduction='none') * total_mask, dim=1)\
            + torch.sum(F.kl_div(total_prob.log(), torch.clamp(l_prob, 1e-3, 1-1e-3), reduction='none') * total_mask, dim=1))/3.0
    
    # HAPMC+PPA
    PC_loss = prototype_loss(features, class_features, masks, proto_ids, parameters)

    return 0.5*(CE_loss+Dice_loss)+kl_loss+parameters["alpha"]*PC_loss
