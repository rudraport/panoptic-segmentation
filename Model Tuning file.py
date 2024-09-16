# mask2former_panoptic_segmentation.py
# dependencies
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.9/index.html

# # Install Mask2Former
# git clone https://github.com/facebookresearch/Mask2Former
# cd Mask2Former
# python setup.py install

# Install pycocotools
pip
install
pycocotools
import os
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_panoptic_separated
from mask2former import add_maskformer2_config
from detectron2.utils.logger import setup_logger
from pycocotools.coco import COCO
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator

# Set up the logger
setup_logger()


# Register COCO Panoptic Segmentation dataset using pycocotools
def register_coco_panoptic_datasets():
    data_root = "/path/to/coco/dataset"

    # Register train and validation datasets for COCO Panoptic Segmentation
    register_coco_panoptic_separated(
        "coco_panoptic_train", {},
        os.path.join(data_root, "annotations/panoptic_train2017.json"),
        os.path.join(data_root, "train2017"),
        os.path.join(data_root, "annotations/panoptic_train2017")
    )

    register_coco_panoptic_separated(
        "coco_panoptic_val", {},
        os.path.join(data_root, "annotations/panoptic_val2017.json"),
        os.path.join(data_root, "val2017"),
        os.path.join(data_root, "annotations/panoptic_val2017")
    )


# Define the Mask2Former Trainer for panoptic segmentation
class Mask2FormerPanopticTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)


# Load data using pycocotools
def load_coco_data():
    data_root = "/path/to/coco/dataset"
    ann_file_train = os.path.join(data_root, "annotations/panoptic_train2017.json")
    coco_train = COCO(ann_file_train)

    ann_file_val = os.path.join(data_root, "annotations/panoptic_val2017.json")
    coco_val = COCO(ann_file_val)

    # Return COCO objects for train and validation
    return coco_train, coco_val


# Setup the configuration
def setup_cfg():
    cfg = get_cfg()
    add_maskformer2_config(cfg)

    # Load the config from Mask2Former repository
    cfg.merge_from_file("./configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml")

    # Register the datasets
    cfg.DATASETS.TRAIN = ("coco_panoptic_train",)
    cfg.DATASETS.TEST = ("coco_panoptic_val",)

    # Hyperparameters for training
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/maskformer/mask2former_panoptic_r50.pth"  # Pre-trained model
    cfg.SOLVER.IMS_PER_BATCH = 8  # Images per batch
    cfg.SOLVER.BASE_LR = 0.0001  # Learning rate
    cfg.SOLVER.MAX_ITER = 50000  # Number of iterations
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 133  # Number of classes for COCO dataset
    cfg.OUTPUT_DIR = "./output"

    # Ensure the output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


# Main function to run training
def main():
    # Register datasets
    register_coco_panoptic_datasets()

    # Load COCO data using pycocotools
    coco_train, coco_val = load_coco_data()

    # Setup the configuration
    cfg = setup_cfg()

    # Initialize the trainer and start training
    trainer = Mask2FormerPanopticTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
import torch
from transformers import Mask2FormerConfig, Mask2FormerForImageSegmentation, Trainer, TrainingArguments
from datasets import load_dataset
from torchvision import transforms
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader


# Custom Dataset to handle COCO Panoptic Segmentation data
class COCOPanopticDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.img_dir}/{img_metadata['file_name']}"
        img = Image.open(img_path).convert("RGB")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        labels = []
        for ann in anns:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            labels.append(ann['category_id'])

        masks = torch.stack([torch.tensor(mask) for mask in masks])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, masks, labels


# Define the dataset paths (You need to provide correct paths)
train_img_dir = 'train/images'
train_ann_file = 'train/annotations.json'

# Define data transformations (Resize, Normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the COCO dataset
train_dataset = COCOPanopticDataset(img_dir=train_img_dir, ann_file=train_ann_file, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Define the Mask2Former model and its configuration
config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
model = Mask2FormerForImageSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")

# Setup Trainer for model training
training_args = TrainingArguments(
    output_dir="./mask2former-output",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("facebook/mask2former-swin-large-coco-panoptic")
# trying to keep same names fro saving us from hassles
