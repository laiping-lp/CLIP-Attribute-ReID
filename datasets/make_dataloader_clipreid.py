import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .uavhuman import UAVHuman
from .uavhuman_attr import UAVHumanAttr
from .common import CommDataset
import collections.abc as container_abcs
int_classes = int
string_classes = str
from torch.utils.data import Subset
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    "uavhuman":UAVHuman,
    "uavhuman_attr":UAVHumanAttr,
}

def train_collate_fn(batch,dataset_name):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    #has attributes
    if dataset_name == "uavhuman_attr":
        imgs, pids, camids, viewids , _,attributes  = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, viewids,attributes
    else:
        imgs, pids, camids, viewids , _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, viewids,


def val_collate_fn(batch,dataset_name):
    #has attributes
    if dataset_name == "uavhuman_attr":
        imgs, pids, camids, viewids, img_paths,attributes = zip(*batch)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids_batch = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths,attributes
    else:
        imgs, pids, camids, viewids, img_paths = zip(*batch)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        camids_batch = torch.tensor(camids, dtype=torch.int64)
        return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg,get_test=None):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    
    #""" mmvrac-reid
    # train_items = list()
    # domain_names = []
    # train_items.extend(dataset.train)
    # domain_names.append(dataset.dataset_name)

    # train_set = CommDataset(cfg, train_items, train_transforms, relabel=True, domain_names=domain_names)
    # train_set_normal = CommDataset(cfg, train_items, val_transforms, relabel=True, domain_names=domain_names)
    # """
    
    train_set = ImageDataset(dataset.train, train_transforms,cfg.DATASETS.NAMES)
    train_set_normal = ImageDataset(dataset.train, val_transforms,cfg.DATASETS.NAMES)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    # view_num = dataset.num_train_vids
    view_num = 0

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(cfg,dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_stage2 = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=lambda batch: train_collate_fn(batch, dataset_name=cfg.DATASETS.NAMES),
                pin_memory=True,
            )
        else:
            train_loader_stage2 = DataLoader(
                train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(cfg,dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=lambda batch: train_collate_fn(batch, dataset_name=cfg.DATASETS.NAMES),
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=lambda batch: train_collate_fn(batch, dataset_name=cfg.DATASETS.NAMES),
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms,cfg.DATASETS.NAMES)

    num_val = len(val_set)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=lambda batch: val_collate_fn(batch, dataset_name=cfg.DATASETS.NAMES)
    )
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=lambda batch: train_collate_fn(batch, dataset_name=cfg.DATASETS.NAMES),
    )
    if get_test == None:
        return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num
    else:
        query_set = Subset(val_set,list(range(min(len(dataset.query),num_val))))
        gallery_set = Subset(val_set,list(range(min(len(dataset.query),num_val),num_val)))
        query_loader = DataLoader(
            query_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=lambda batch: val_collate_fn(batch, dataset_name=cfg.DATASETS.NAMES)
        )

        gallery_loader = DataLoader(
            gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=lambda batch: val_collate_fn(batch, dataset_name=cfg.DATASETS.NAMES)
        )
        return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num,query_loader,gallery_loader

    



def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs


def make_sampler(train_set, num_batch, num_instance, num_workers,
                 mini_batch_size, drop_last=True, flag1=True, flag2=True, seed=None, train_pids=None, cfg=None, model=None):

    if cfg.DATALOADER.SAMPLER == 'single_domain':
        data_sampler = samplers.DomainIdentitySampler(train_set.img_items,
                                                      mini_batch_size, num_instance,train_pids)
    elif cfg.DATALOADER.SAMPLER == 'SHS':
        test_transforms = build_transforms(cfg, is_train=False)
        data_sampler = samplers.SHS(cfg=cfg, train_set=train_set,
                                     batch_size=mini_batch_size,
                                     model=model,
                                     transform=test_transforms)
    elif flag1:
        data_sampler = samplers.RandomIdentitySampler(train_set.img_items,
                                                      mini_batch_size, num_instance)
    else:
        data_sampler = samplers.DomainSuffleSampler(train_set.img_items,
                                                     num_batch, num_instance, flag2, seed, cfg)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader