# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch

def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64, device=image_features.device)

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm  = text_features  / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * (image_norm @ text_norm.t())
    logits_per_text  = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) * 0.5


# def compute_sdm(image_fetures, text_fetures, pid, logit_scale,
#                 image_id=None, factor=0.3, epsilon=1e-8):
#     """
#     Similarity Distribution Matching (SDM)
#     """
#     batch_size = image_fetures.shape[0]

#     pid = pid.view(batch_size, 1)               # [B,1]
#     labels = (pid.eq(pid.t())).float()          # [B,B] 同 pid 为 1，否则 0

#     if image_id is not None:
#         image_id = image_id.view(batch_size, 1)
#         image_id_mask = (image_id.eq(image_id.t())).float()
#         # soft label mix
#         labels = (labels - image_id_mask) * factor + image_id_mask

#     image_norm = image_fetures / (image_fetures.norm(dim=1, keepdim=True) + epsilon)
#     text_norm  = text_fetures  / (text_fetures.norm(dim=1, keepdim=True) + epsilon)

#     # cosine sim
#     t2i_cosine_theta = text_norm @ image_norm.t()     # [B,B]
#     i2t_cosine_theta = t2i_cosine_theta.t()

#     text_proj_image = logit_scale * t2i_cosine_theta  # [B,B]
#     image_proj_text = logit_scale * i2t_cosine_theta  # [B,B]

#     # ✅ normalize true distribution (避免除零)
#     denom = labels.sum(dim=1, keepdim=True).clamp_min(epsilon)
#     labels_distribute = labels / denom                # [B,B]

#     i2t_pred = F.softmax(image_proj_text, dim=1)
#     i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))

#     t2i_pred = F.softmax(text_proj_image, dim=1)
#     t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

#     loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
#     return loss

def compute_sdm(image_fetures, text_fetures, pid, logit_scale,
                image_id=None, factor=0.3, epsilon=1e-8):

    device = image_fetures.device
    batch_size = image_fetures.shape[0]

    # ✅ 统一设备 + dtype
    pid = pid.to(device).view(batch_size, 1).long()  # [B,1]

    # labels on same device
    labels = pid.eq(pid.t()).float()                 # [B,B]

    if image_id is not None:
        image_id = image_id.to(device).view(batch_size, 1).long()
        image_id_mask = image_id.eq(image_id.t()).float()
        labels = (labels - image_id_mask) * factor + image_id_mask

    # ✅ logit_scale 也必须在同一设备（尤其你用 self.logit_scale 时很容易在 cpu）
    if not torch.is_tensor(logit_scale):
        logit_scale = torch.tensor(logit_scale, device=device)
    else:
        logit_scale = logit_scale.to(device)

    # normalize features
    image_norm = image_fetures / (image_fetures.norm(dim=1, keepdim=True) + epsilon)
    text_norm  = text_fetures  / (text_fetures.norm(dim=1, keepdim=True) + epsilon)

    # cosine sim
    t2i_cosine_theta = text_norm @ image_norm.t()   # [B,B]
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    denom = labels.sum(dim=1, keepdim=True).clamp_min(epsilon)
    labels_distribute = labels / denom

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))

    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    return loss

# def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
#     """
#     Cross-Modal Projection Matching Loss (CMPM)
#     """
#     batch_size = image_embeddings.shape[0]

#     labels = labels.view(batch_size, 1)
#     labels_mask = labels.eq(labels.t()).float()      # [B,B] 同 pid=1，否则0

#     # normalize
#     image_norm = image_embeddings / (image_embeddings.norm(dim=1, keepdim=True) + epsilon)
#     text_norm  = text_embeddings  / (text_embeddings.norm(dim=1, keepdim=True) + epsilon)

#     # cross-modal projection scores
#     # （这里我用更常见的 cosine: image_norm @ text_norm.t()，你原来是 “未归一化 @ 归一化”，也可以保留）
#     image_proj_text = image_norm @ text_norm.t()     # [B,B]
#     text_proj_image = image_proj_text.t()            # [B,B]

#     # ✅ normalize the true matching distribution（按行sum归一化，避免NaN）
#     denom = labels_mask.sum(dim=1, keepdim=True).clamp_min(epsilon)
#     labels_dist = labels_mask / denom                # [B,B]

#     i2t_pred = F.softmax(image_proj_text, dim=1)
#     i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_dist + epsilon))

#     t2i_pred = F.softmax(text_proj_image, dim=1)
#     t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_dist + epsilon))

#     cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
#     return cmpm_loss

def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss (CMPM) - device-safe
    """
    device = image_embeddings.device
    batch_size = image_embeddings.shape[0]

    # ✅ labels 搬到同一设备 + long
    labels = labels.to(device).view(batch_size, 1).long()
    labels_mask = labels.eq(labels.t()).float()  # [B,B]

    # normalize
    image_norm = image_embeddings / (image_embeddings.norm(dim=1, keepdim=True) + epsilon)
    text_norm  = text_embeddings  / (text_embeddings.norm(dim=1, keepdim=True) + epsilon)

    # cosine sim matrix
    image_proj_text = image_norm @ text_norm.t()  # [B,B]
    text_proj_image = image_proj_text.t()

    # ✅ normalize true distribution（避免除0）
    denom = labels_mask.sum(dim=1, keepdim=True).clamp_min(epsilon)
    labels_dist = labels_mask / denom

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_dist + epsilon))

    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_dist + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    return cmpm_loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss (ID loss)
    """
    loss_img = F.cross_entropy(image_logits, labels)
    loss_txt = F.cross_entropy(text_logits, labels)
    return 0.5 * (loss_img + loss_txt)

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        # ✅ 加入 image_features / text_features / logit_scale（用于ITC）
        def loss_func(
            score, feat, target,
            target_cam=None,
            i2tscore=None,
            image_features=None,
            text_features=None,
            pids=None,
            logit_scale=torch.tensor(50.),
            image_logits=None,
            text_logits=None
        ):
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    # -------- 1) ID LOSS --------
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    # -------- 2) TRIPLET LOSS --------
                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, target)[0]
                    ID_LOSS = 0.0
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                    # -------- 3) 你原来的 I2T（可选）--------
                    # if i2tscore != None:
                    #     I2TLOSS = xent(i2tscore, target)
                    #     loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                    
                    # -------- 4) ✅ ITC（可选）--------
                    # 这里的“满足 itc 条件”我按 cfg.MODEL.IF_ITC == 'on' 来写
                    # 并要求 image_features/text_features/logit_scale 都传进来
                    if cfg.MODEL.IF_ITC_LOSS == 'on':
                        if (image_features is None) or (text_features is None) or (logit_scale is None):
                            # 你也可以选择直接不加；或者 raise，让调用方必须传
                            raise ValueError("IF_ITC is on, but image_features/text_features/logit_scale is None.")
                        itc_loss = compute_itc(image_features, text_features, logit_scale)
                        loss = loss + cfg.MODEL.ITC_LOSS_WEIGHT * itc_loss
                    
                    # -------- ✅ SDM（可选）--------
                    if cfg.MODEL.IF_SDM_LOSS == "on":
                        if (image_features is None) or (text_features is None) or (pids is None) or (logit_scale is None):
                            raise ValueError("IF_SDM is on, but i_feats/t_feats/pid/logit_scale is None.")

                        SDMLOSS = compute_sdm(image_features, text_features, pids, logit_scale)
                        loss = loss + cfg.MODEL.SDM_LOSS_WEIGHT * SDMLOSS

                    # -------- ✅ CMPM（可选）--------
                    if cfg.MODEL.IF_CMPM_LOSS == "on":
                        if (image_features is None) or (text_features is None) or (pids is None):
                            raise ValueError("IF_CMPM is on, but image_features/text_features/pids is None.")
                        CMPMLOSS = compute_cmpm(image_features, text_features, pids)
                        loss = loss + cfg.MODEL.CMPM_LOSS_WEIGHT * CMPMLOSS
                                        
                    # -------- ✅ Cross-modal ID（可选）--------
                    if cfg.MODEL.IF_CM_ID_LOSS == "on":
                        if (image_logits is None) or (text_logits is None):
                            raise ValueError("IF_ID is on, but image_logits/text_logits is None.")

                        if cfg.MODEL.IF_LABELSMOOTH == 'on':
                            id_loss = compute_id(image_logits, text_logits, target)
                        else:
                            id_loss = compute_id(image_logits, text_logits, target)

                        loss = loss + cfg.MODEL.ID2_LOSS_WEIGHT * id_loss   # 这里用个新权重名避免和原ID冲突

                    return loss
                else:
                    # -------- 1) ID LOSS --------
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    # -------- 2) TRIPLET LOSS --------
                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    # -------- 3) 你原来的 I2T（可选）--------
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    # -------- 4) ✅ ITC（可选）--------
                    

                    return loss
            # else:
            #     print('expected METRIC_LOSS_TYPE should be triplet'
            #           'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


