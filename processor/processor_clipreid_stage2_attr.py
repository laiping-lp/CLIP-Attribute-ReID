# -*- encoding: utf-8 -*-
'''
@File    :   processor_clipreid_stage2_attr.py
@Time    :   2025/12/17 21:10:50
@Author  :   laiping
@Version :   1.0
@Contact :   laiping2001@gmail.com
'''

# here put the import lib

import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss

def do_train_stage2_attr(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    gender_num_classes = 2
    hat_num_classes = 5
    backpack_num_classes = 5
    ucc_num_classes = 12
    ucs_num_classes = 4
    lcc_num_classes = 12
    lcs_num_classes = 4
    # 初始化时，指定 dtype 为 torch.int64 并与 l_list 的类型一致
    gender_per_class = torch.zeros(num_classes, dtype=torch.int64, device=device)
    hat_per_class = torch.zeros(num_classes, dtype=torch.int64, device=device)
    backpack_per_class = torch.zeros(num_classes, dtype=torch.int64, device=device)
    ucc_per_class = torch.zeros(num_classes, dtype=torch.int64, device=device)
    ucs_per_class = torch.zeros(num_classes, dtype=torch.int64, device=device)
    lcc_per_class = torch.zeros(num_classes, dtype=torch.int64, device=device)
    lcs_per_class = torch.zeros(num_classes, dtype=torch.int64, device=device)

    # 从 train_loader_stage2 中提取属性标签，并确保赋值时类型一致
    for img, vid, target_cam, target_view, attributes in train_loader_stage2:
        for i, attribute in zip(vid, attributes):
            gender_per_class[i] = torch.tensor(attribute['gender'], dtype=torch.int64, device=device)
            ucc_per_class[i] = torch.tensor(attribute['upper_color'], dtype=torch.int64, device=device)
            ucs_per_class[i] = torch.tensor(attribute['upper_style'], dtype=torch.int64, device=device)
            lcc_per_class[i] = torch.tensor(attribute['lower_color'], dtype=torch.int64, device=device)
            lcs_per_class[i] = torch.tensor(attribute['lower_style'], dtype=torch.int64, device=device)
            hat_per_class[i] = torch.tensor(attribute['hat'], dtype=torch.int64, device=device)
            backpack_per_class[i] = torch.tensor(attribute['backpack'], dtype=torch.int64, device=device)


    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            gender_batch   = gender_per_class[l_list]      # [B]
            ucc_batch      = ucc_per_class[l_list]         # [B]
            ucs_batch      = ucs_per_class[l_list]         # [B]
            lcc_batch      = lcc_per_class[l_list]         # [B]
            lcs_batch      = lcs_per_class[l_list]         # [B]
            hat_batch      = hat_per_class[l_list]         # [B]
            backpack_batch = backpack_per_class[l_list]    # [B]
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True,
                                        gender_label=gender_batch,
                                        ucc_label=ucc_batch,
                                        ucs_label=ucs_batch,
                                        lcc_label=lcc_batch,
                                        lcs_label=lcs_batch,
                                        hat_label=hat_batch,
                                        backpack_label=backpack_batch)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view,attributes) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _,attributes) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _,attributes) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference_attr(cfg,
                 model,
                 val_loader,
                 query_loader, 
                 gallery_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_val = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()
    evaluator_val.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    img_path_list_val = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath,attributes) in enumerate(query_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            gender_labels = []
            ucc_labels = []
            ucs_labels = []
            lcc_labels = []
            lcs_labels = []
            hat_labels = []
            backpack_labels = []
            for i,attribute in zip(pid,attributes):
                gender_labels.append(torch.as_tensor(attribute['gender'],   dtype=torch.long, device=img[0].device))
                hat_labels.append(torch.as_tensor(attribute['hat'], dtype=torch.long, device=img[0].device))
                backpack_labels.append(torch.as_tensor(attribute['backpack'], dtype=torch.long, device=img[0].device))
                ucc_labels.append(torch.as_tensor(attribute['upper_color'], dtype=torch.long, device=img[0].device))
                ucs_labels.append(torch.as_tensor(attribute['upper_style'], dtype=torch.long, device=img[0].device))
                lcs_labels.append(torch.as_tensor(attribute['lower_style'], dtype=torch.long, device=img[0].device))
                lcc_labels.append(torch.as_tensor(attribute['lower_color'], dtype=torch.long, device=img[0].device))
            gender_labels_list   = torch.stack(gender_labels, dim=0).long().cuda()  # [N]
            hat_labels_list      = torch.stack(hat_labels, dim=0).long().cuda()     # [N]
            backpack_labels_list = torch.stack(backpack_labels, dim=0).long().cuda()# [N]
            ucc_labels_list      = torch.stack(ucc_labels, dim=0).long().cuda()     # [N]
            ucs_labels_list      = torch.stack(ucs_labels, dim=0).long().cuda()     # [N]
            lcs_labels_list      = torch.stack(lcs_labels, dim=0).long().cuda()     # [N]
            lcc_labels_list      = torch.stack(lcc_labels, dim=0).long().cuda()     # [N]
            feat = model(img, cam_label=camids, view_label=target_view,get_expansion = True,gender_label=gender_labels_list,hat_label=hat_labels_list,backpack_label= backpack_labels_list,ucc_label=ucc_labels_list,lcc_label=lcc_labels_list,ucs_label=ucs_labels_list,lcs_label=lcs_labels_list)
            # feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
    for n_iter, (img, pid, camid, camids, target_view, imgpath,_ ) in enumerate(gallery_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            
            feat = model(img, cam_label=camids, view_label=target_view,get_image=True)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    for n_iter, (img, pid, camid, camids, target_view, imgpath,_ ) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            
            # feat = model(img, cam_label=camids, view_label=target_view)
            feat = model(img, cam_label=camids, view_label=target_view,get_image =True)
            evaluator_val.update((feat, pid, camid))
            img_path_list_val.extend(imgpath)



    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        
    cmc_val, mAP_val, _, _, _, _, _ = evaluator_val.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP_val))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_val[r - 1]))   
    
    return cmc[0], cmc[4]

