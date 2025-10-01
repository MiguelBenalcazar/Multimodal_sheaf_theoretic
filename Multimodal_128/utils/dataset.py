import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


def make_collate_fn(tokenizer, img_processor):
    def collate_fn(batch):
        imgs, texts = zip(*batch)  # unzip list of (img, text)

        imgs = img_processor(images=list(imgs), return_tensors="pt")

        encodings = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        return imgs, {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        }

    return collate_fn


# def make_collate_fn(tokenizer, img_processor):
#     def collate_fn(batch):
#         imgs, texts = zip(*batch)  # unzip list of (img, text)

#         # Process images one by one, return pixel_values, then stack
#         img_tensors = []
#         for img in imgs:
#             processed = img_processor(img, return_tensors="pt")["pixel_values"]  # shape: (1, C, H, W)
#             img_tensors.append(processed)
#         imgs = torch.cat(img_tensors, dim=0)  # shape: (B, C, H, W)

#         # Tokenize texts with minimal padding
#         encodings = tokenizer(
#             list(texts),
#             return_tensors="pt",
#             padding="longest",   # only pad to longest in batch
#             truncation=True
#         )

#         return imgs, {
#             "input_ids": encodings["input_ids"],
#             "attention_mask": encodings["attention_mask"]
#         }

#     return collate_fn



class COCOImageTextDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
  
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path, caption = row["image_name"], row["caption"]

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            logger.warning(f"Failed to open {img_path}, returning black image")
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            caption = ""

        return img, caption


def dataset_class(args, img_processor, tokenizer, file_train, file_test):
   
    train_dataset = COCOImageTextDataset(file_train)
    test_dataset = COCOImageTextDataset(file_test)

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    test_sampler  = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)


    train_loader = DataLoader(
        train_dataset,
        sampler = train_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=make_collate_fn(tokenizer, img_processor)
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=make_collate_fn(tokenizer, img_processor)
    )

    return train_loader, test_loader


# def get_COCO_dataloaders(file_train, file_test, tokenizer, img_processor,
#                          batch_size=32, num_workers=4, pin_memory=True):
#     train_dataset = COCOImageTextDataset(file_train)
#     test_dataset = COCOImageTextDataset(file_test)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         collate_fn=make_collate_fn(tokenizer, img_processor)
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         collate_fn=make_collate_fn(tokenizer, img_processor)
#     )

#     return train_loader, test_loader


# def dataset_class(args, img_processor, tokenizer):
#     random.seed(args.seed)

#     current_folder = os.getcwd()
#     dataset_path = os.path.abspath(os.path.join(current_folder, "..", "..", "data", args.dataset_name))

#     if not (os.path.exists(dataset_path) and os.path.isdir(dataset_path)):
#         raise FileNotFoundError(f"Folder does not exist: {dataset_path}")

#     # Find parquet files
#     all_parquets = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".parquet")]
#     if not all_parquets:
#         raise RuntimeError(f"No parquet files found in {dataset_path}")

#     random.shuffle(all_parquets)
#     split = int(0.8 * len(all_parquets))
#     train_files, test_files = all_parquets[:split], all_parquets[split:]

#     train_dataset = ParquetImageTextDataset(train_files, max_files=1)
#     test_dataset = ParquetImageTextDataset(test_files, max_files=1)

#     train_loader = DataLoader(
#         tproj_embedding_imgrain_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=True,
#         collate_fn=make_collate_fn(tokenizer, img_processor)
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=False,
#         collate_fn=make_collate_fn(tokenizer, img_processor)
#     )

#     return train_loader, test_loader



# class ParquetImageTextDataset(Dataset):
#     def __init__(self, parquet_files, transform=None, max_files=None):
#         """
#         parquet_files: list of parquet file paths
#         transform: optional image transform
#         max_files: limit number of parquet files for testing
#         """
#         if max_files:
#             parquet_files = parquet_files[:max_files]

#         # Load all into one dataframe
#         dfs = [pd.read_parquet(f, columns=["URL", "TEXT"]) for f in parquet_files]
#         self.df = pd.concat(dfs, ignore_index=True)
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         url, text = row["URL"], row["TEXT"]

#         try:
#             img = Image.open(BytesIO(requests.get(url, timeout=5).content)).convert("RGB")
#         except:
#             text = ""
#             img = Image.new("RGB", (224, 224), color=(0, 0, 0))

#         if self.transform:
#             img = self.transform(img)

#         return img, text


# def dataset_class(args, img_processor, tokenizer):
#     random.seed(args.seed)

#     current_folder = os.getcwd()
#     dataset_path = os.path.abspath(os.path.join(current_folder, "..", "..", "data", args.dataset_name))

#     if not (os.path.exists(dataset_path) and os.path.isdir(dataset_path)):
#         raise FileNotFoundError(f"Folder does not exist: {dataset_path}")

#     # Find parquet files
#     all_parquets = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".parquet")]
#     if not all_parquets:
#         raise RuntimeError(f"No parquet files found in {dataset_path}")

#     random.shuffle(all_parquets)
#     split = int(0.8 * len(all_parquets))
#     train_files, test_files = all_parquets[:split], all_parquets[split:]

#     train_dataset = ParquetImageTextDataset(train_files, max_files=1)
#     test_dataset = ParquetImageTextDataset(test_files, max_files=1)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=True,
#         collate_fn=make_collate_fn(tokenizer, img_processor)
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=False,
#         collate_fn=make_collate_fn(tokenizer, img_processor)
#     )

#     return train_loader, test_loader






# import os
# import logging
# import pyarrow.dataset as ds
# import pyarrow as pa
# import torch
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from io import BytesIO
# import requests
# import random

# from torchvision import transforms



# logger = logging.getLogger(__name__)

# def make_collate_fn(tokenizer, img_processor):
#     def collate_fn(batch):
#         imgs, texts = zip(*batch)  # unzip list of (img, text)

#         imgs = img_processor(images=imgs, return_tensors="pt")


#         # Tokenize text batch
#         encodings = tokenizer(
#             list(texts),
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         )

        
#         # Return dict instead of BatchEncoding for safety
#         return imgs, {
#             "input_ids": encodings["input_ids"],
#             "attention_mask": encodings["attention_mask"]
#         }

#     return collate_fn


# def load_dataset_from_folder(random, dataset_path:str="./data/Laion_400M", format:str="parquet", train_split:float=0.8):
#     # Create a dataset from all parquet shards
#     dataset = ds.dataset(dataset_path, format= format)
   
#     logger.info(dataset.schema)

#     # Get all fragments (each Parquet file is a fragment)
#     fragments = list(dataset.get_fragments())
#     num_fragments = len(fragments)

#     # Shuffle for randomness
    
#     random.shuffle(fragments)

#     # Split 80/20
#     train_fragments = fragments[:int(train_split * num_fragments)]
#     test_fragments  = fragments[int(train_split * num_fragments):]

#     logger.info(f"Train shards: {len(train_fragments)}, Test shards: {len(test_fragments)}")

#     return train_fragments, test_fragments\
    

# class ArrowImageTextDataset(Dataset):
#     def __init__(self, fragments, transform=None, max_fragments:int=None):
#         """
#         fragments: list of PyArrow dataset fragments (parquet shards)
#         transform: optional torchvision transform
#         max_fragments: optional int to limit number of shards for faster testing
#         """
#         if max_fragments:
#             fragments = fragments[:max_fragments]

#         # Load tables from fragments
#         self.table = pa.concat_tables([
#             f.to_table(columns=["URL", "TEXT"]) for f in fragments
#         ])
#         self.transform = transform
    
#     def __len__(self):
#         return self.table.num_rows

#     def __getitem__(self, idx):
#         row = self.table.slice(idx, 1)
#         url = row.column("URL")[0].as_py()
#         text = row.column("TEXT")[0].as_py()

        
      
#         try:
#             img = Image.open(BytesIO(requests.get(url, timeout=5).content)).convert("RGB")
#         except:
#             text = ''
#             img = Image.new("RGB", (224, 224), color=(0, 0, 0))

#         # if self.transform:
#         #     img = self.transform(images=img, return_tensors="pt")
      
#         return img, text


# def dataset_class(args, img_processor, tokenizer):
#     random.seed(args.seed)

#     current_folder = os.getcwd()
#     dataset_path = os.path.abspath(os.path.join(current_folder,"..", "..","data", args.dataset_name))
    
#     if not (os.path.exists(dataset_path) and os.path.isdir(dataset_path)):
#         raise FileNotFoundError(f"Folder does not exist: {dataset_path}")
    
#     train_dataset, test_dataset = load_dataset_from_folder(random, dataset_path)

#     # transform = transforms.Compose([
#     #     transforms.Resize((224, 224)),
#     #     transforms.ToTensor(),
#     # ])

#     # train_dataset = ArrowImageTextDataset(train_dataset,transform=img_processor, max_fragments=1)
#     # test_dataset = ArrowImageTextDataset(test_dataset, transform=img_processor,  max_fragments=1)

#     train_dataset = ArrowImageTextDataset(train_dataset, max_fragments=1)
#     test_dataset = ArrowImageTextDataset(test_dataset, max_fragments=1)

#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=True, 
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=True,
#         collate_fn=make_collate_fn(tokenizer, img_processor )
#     )
#     test_loader  = DataLoader(
#         test_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=False, 
#         num_workers=args.num_workers,
#         pin_memory=args.pin_mem,
#         drop_last=False,
#         collate_fn=make_collate_fn(tokenizer, img_processor)
#     )

#     return train_loader, test_loader

