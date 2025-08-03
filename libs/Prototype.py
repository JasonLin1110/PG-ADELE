import torch
from tqdm import tqdm
import torch.nn.functional as F
import warnings
from collections import defaultdict
warnings.simplefilter("ignore", UserWarning)

def dist_mask(pred_prob, prototype_buffer, feature, preds_c, parameters):
    cosine = F.cosine_similarity(feature.unsqueeze(1).unsqueeze(1), prototype_buffer.unsqueeze(0).unsqueeze(-1).unsqueeze(-1), dim=3)
    pred_cosine = cosine[:,preds_c,:,:,:]
    p_max_value, _ = pred_cosine.max(dim=1)
    if preds_c==0:
        return (pred_prob[:,preds_c,:,:]>parameters["conf_threshold_bg"]) & (p_max_value>parameters["similarity_threshold"])
    return (pred_prob[:,preds_c,:,:]>parameters["conf_threshold"]) & (p_max_value>parameters["similarity_threshold"])

def online_clustering(prototypes, features, iters=3, kappa=0.05):
    K, D = prototypes.shape
    N = features.shape[0]
    k_cosine = F.cosine_similarity(prototypes.unsqueeze(1), features.unsqueeze(0), dim=2)
    L = torch.exp(k_cosine / kappa)
    L /= torch.sum(L)
    for _ in range(iters):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K
        L /= torch.sum(L, dim=0, keepdim=True)
        L /= N
    L *= N
    return L.T

def distribute(features, prototypes, labels, parameters):
    B, C, H, W = features.shape
    features_reshaped = features.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = labels.view(-1)
    if parameters["p_num"]==1:
        proto_ids = torch.zeros(B*H*W, dtype=torch.long, device=features.device)
        return proto_ids
    proto_ids = torch.empty(B*H*W, dtype=torch.long, device=features.device)
    for cls in range(parameters["class_num"]):
        mask = (labels_flat==cls)
        if mask.sum()==0:
            continue
        cls_features = features_reshaped[mask]
        cls_prototypes = prototypes[cls,:,:]
        k_probs = online_clustering(cls_prototypes, cls_features)  # (Nc, p_num)
        cluster_idx = torch.argmax(k_probs, dim=1)
        proto_ids[mask] = cluster_idx.to(proto_ids.device)
    return proto_ids
    
def update(features, prototypes, labels, proto_ids, parameters):
    B, C, H, W = features.shape
    features_reshaped = features.permute(0, 2, 3, 1).reshape(-1, C)
    labels_flat = labels.view(-1)
    for c in range(parameters["class_num"]):
        label_cnt=(labels_flat==c).sum()
        if label_cnt< parameters["p_num"]:
            continue
        for k in range(parameters["p_num"]):
            k_mask = (labels_flat==c) & (proto_ids==k)
            if k_mask.sum()==0:
                continue
            selected_feats = features_reshaped[k_mask]
            mean_features = F.normalize(selected_feats.mean(dim=0, keepdim=True), p=2, dim=1)
            prototypes[c,k,:] = 0.999 * prototypes[c,k,:] + (1 - 0.999) * mean_features
    return prototypes

def spherical_kmeans(X, num_clusters=10, num_iters=20, device='cuda'):
    X = X.to(device)
    X = F.normalize(X, p=2, dim=1)

    N, D = X.shape
    indices = torch.randperm(N)[:num_clusters]
    centers = X[indices]

    for _ in range(num_iters):
        sim = torch.matmul(X, centers.T)  # [N, K]

        labels = sim.argmax(dim=1)  # [N]

        new_centers = []
        for k in range(num_clusters):
            assigned = X[labels == k]
            if assigned.shape[0] == 0:
                new_center = X[torch.randint(0, N, (1,))]
            else:
                new_center = assigned.mean(dim=0)
            new_centers.append(F.normalize(new_center, p=2, dim=0))
        centers = torch.stack(new_centers, dim=0)  # [K, D]
    return centers

def inital(model, dataloader, parameters, max_samples_per_class):
    with torch.no_grad():
        features_per_class = defaultdict(list)
        for images, masks, _, _ in tqdm(dataloader):
            images, masks = images.cuda(), masks.cuda()
            o_features, _ = model(images)
            small_images = F.interpolate(images, scale_factor=0.7, mode="bilinear", align_corners=True, recompute_scale_factor=True)
            small_features, _ = model(small_images)
            small_features = F.interpolate(small_features, size=(256, 256), mode='bilinear', align_corners=True)
            large_images = F.interpolate(images, scale_factor=1.5, mode="bilinear", align_corners=True, recompute_scale_factor=True)
            large_features, _ = model(large_images)
            large_features = F.interpolate(large_features, size=(256, 256), mode='bilinear', align_corners=True)
            features = (o_features+small_features+large_features)/3.0
            features = features.permute(0, 2, 3, 1)
            for c in range(parameters["class_num"]):
                mask = (masks == c)  # [B, H, W]
                selected_features = features[mask]  # shape: [N, 64]
                if selected_features.numel() > 0:
                    features_per_class[c].append(selected_features.cpu())
        for c in range(parameters["class_num"]):
            features_per_class[c] = torch.cat(features_per_class[c], dim=0)
        all_prototypes = []
        if parameters["p_num"]==1:
            for c in range(parameters["class_num"]):
                features_c = features_per_class[c]
                if features_c.shape[0] > max_samples_per_class[c]:
                    idx = torch.randperm(features_c.shape[0])[:max_samples_per_class[c]]
                    features_c = features_c[idx]
                features_c = features_c.cpu()
                cluster_centers = features_c.mean(dim=0)
                cluster_centers = F.normalize(cluster_centers, p=2, dim=0).unsqueeze(0)
                all_prototypes.append(cluster_centers.cpu())
        else:
            for c in range(parameters["class_num"]):
                features_c = features_per_class[c]
                if features_c.shape[0] > max_samples_per_class[c]:
                    idx = torch.randperm(features_c.shape[0])[:max_samples_per_class[c]]
                    features_c = features_c[idx]
                features_c = features_c.cpu()
                cluster_centers = spherical_kmeans(features_c, num_clusters=parameters["p_num"], num_iters=20, device='cuda')
                cluster_centers = F.normalize(cluster_centers, p=2, dim=1)
                all_prototypes.append(cluster_centers.cpu())
        all_prototypes = torch.stack(all_prototypes, dim=0)
    return all_prototypes