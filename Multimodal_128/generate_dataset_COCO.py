from pycocotools.coco import COCO
from tqdm import tqdm
import pandas as pd
import os

def create_dataset_from_COCO(
        annotations_path: str, 
        images_path: str,
        output_dir: str = None, 
        prefix: str = "train"
    ):
    # Initialize COCO captions API
    coco_caps = COCO(annotations_path)

    # Get all image ids
    imgIds = coco_caps.getImgIds()
    print(f"Total images: {len(imgIds)}")
    
    data = []

    for img_id in tqdm(imgIds, desc="Processing images"):
        img_info = coco_caps.loadImgs(img_id)[0]
        img_name = img_info["file_name"]

        annIds = coco_caps.getAnnIds(imgIds=img_info['id'])
        anns = coco_caps.loadAnns(annIds)

        for annotation in anns:
            info = {
                "ann_id": annotation["id"],
                "image_id": annotation["image_id"],
                "image_name":os.path.join(images_path, img_name),
                "caption": annotation["caption"]
            }
            data.append(info)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    print(f"Total rows: {len(df)}")

    # Save if output_dir is provided
    if output_dir:
        out_file = f"{output_dir}/coco_{prefix}_captions.csv"
        df.to_csv(out_file, index=False)
        print(f"Saved dataset to {out_file}")

    return df




curretn_folder = os.getcwd()
dataset = os.path.abspath(os.path.join(curretn_folder,"..", "..","data","COCO", "annotations_trainval2017", "annotations"))
dataset_img = os.path.abspath(os.path.join(curretn_folder,"..", "..","data","COCO", "val2017"))

annFile = os.path.join(dataset, "captions_val2017.json")

out = "/home/home/Desktop/research/data/COCO/joint/"

create_dataset_from_COCO(annFile, dataset_img, out,  prefix= "test" )