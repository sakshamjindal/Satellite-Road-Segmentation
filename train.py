import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import models
import inspect
import math
from utils import losses
from utils import Logger
from datasets import MassuchusettsDataset
from utils.torchsummary import summary
from trainer import Trainer

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def augmentations(image_size = 512):
    
    import albumentations as A
    return A.Compose([
        A.RandomCrop(width = image_size, height = image_size, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        A.Blur(p=0.01, blur_limit = 3),
    ], p = 1)

def main(config, resume):
    train_logger = Logger()
    
    # DATASETS
    root_dir = config["train_loader"]["data_dir"]
    train_file = os.path.join(root_dir, "train.csv")
    val_file = os.path.join(root_dir, "val.csv")
    transforms = augmentations(image_size = config["train_loader"]["crop_size"])
    train_dataset = MassuchusettsDataset(file_path = train_file, transforms = transforms)
    val_dataset = MassuchusettsDataset(file_path = val_file, transforms = None)
    
    # DATA LOADERS
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size = config["train_loader"]["batch_size"], pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=4, batch_size= config["val_loader"]["batch_size"], pin_memory=True)
    train_loader.MEAN = train_dataset.MEAN
    train_loader.STD = train_dataset.STD

    # MODEL
    model = get_instance(models, 'arch', config, config["trainer"]["num_classes"])
    print(f'\n{model}\n')

    # LOSS
    if "CrossEntropyLoss" in config['loss']:
        loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'], weight = config["weights"])
    else:
        loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)
