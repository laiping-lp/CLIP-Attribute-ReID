# -*- encoding: utf-8 -*-
'''
@File    :   processor_clipreid_stage1_attr.py
@Time    :   2025/12/17 21:10:45
@Author  :   laiping
@Version :   1.0
@Contact :   laiping2001@gmail.com
'''

# here put the import lib

import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

def do_train_stage1_attr(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device) 
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    gender_labels = []
    ucc_labels = []
    ucs_labels = []
    lcc_labels = []
    lcs_labels = []
    hat_labels = []
    backpack_labels = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view,attributes) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True)
                for i, img_feat,attribute in zip(target, image_feature,attributes):
                    labels.append(i)
                    
                    gender_labels.append(torch.as_tensor(attribute['gender'],   dtype=torch.long, device=i.device))
                    hat_labels.append(torch.as_tensor(attribute['hat'],        dtype=torch.long, device=i.device))
                    backpack_labels.append(torch.as_tensor(attribute['backpack'], dtype=torch.long, device=i.device))
                    ucc_labels.append(torch.as_tensor(attribute['upper_color'], dtype=torch.long, device=i.device))
                    ucs_labels.append(torch.as_tensor(attribute['upper_style'], dtype=torch.long, device=i.device))
                    lcs_labels.append(torch.as_tensor(attribute['lower_style'], dtype=torch.long, device=i.device))
                    lcc_labels.append(torch.as_tensor(attribute['lower_color'], dtype=torch.long, device=i.device))

                    image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0).cuda() #N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        gender_labels_list   = torch.stack(gender_labels, dim=0).long().cuda()  # [N]
        hat_labels_list      = torch.stack(hat_labels, dim=0).long().cuda()     # [N]
        backpack_labels_list = torch.stack(backpack_labels, dim=0).long().cuda()# [N]
        ucc_labels_list      = torch.stack(ucc_labels, dim=0).long().cuda()     # [N]
        ucs_labels_list      = torch.stack(ucs_labels, dim=0).long().cuda()     # [N]
        lcs_labels_list      = torch.stack(lcs_labels, dim=0).long().cuda()     # [N]
        lcc_labels_list      = torch.stack(lcc_labels, dim=0).long().cuda()     # [N]

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features
    del gender_labels, hat_labels, backpack_labels
    del ucc_labels, ucs_labels, lcs_labels, lcc_labels

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            gender_batch   = gender_labels_list[b_list]
            hat_batch      = hat_labels_list[b_list]
            backpack_batch = backpack_labels_list[b_list]
            ucc_batch      = ucc_labels_list[b_list]
            ucs_batch      = ucs_labels_list[b_list]
            lcs_batch      = lcs_labels_list[b_list]
            lcc_batch      = lcc_labels_list[b_list]
            with amp.autocast(enabled=True):
                text_features = model(label = target, get_text = True,gender_label=gender_batch,hat_label=hat_batch,backpack_label= backpack_batch,ucc_label=ucc_batch,lcc_label=lcc_batch,ucs_label=ucs_batch,lcs_label=lcs_batch)
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)

            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
