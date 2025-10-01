import logging
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)



class COCOImageTextDataset(Dataset):
    def __init__(self, file_pt):
        data = torch.load(file_pt)
        self.img = data.get("z_img",None) 
        self.txt = data.get("z_text",None) 
        self.img_ids = data.get("img_ids",None) 
        self.caption_ids = data.get("caption_ids", None)  
  
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
         return self.img[idx], self.txt[idx], self.img_ids[idx], self.caption_ids[idx]


def dataset_class(args,file_train, file_test):
   
    train_dataset = COCOImageTextDataset(file_train)
    test_dataset = COCOImageTextDataset(file_test)

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        test_sampler  = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        sampler = train_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return train_loader, test_loader

