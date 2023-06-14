# Satellite Road Segmentation

This repo contains a PyTorch an implementation of semantic segmentation models for Massuchusetts Roads Dataset

## Installation

```bash
pip install -r requirements.txt
```

## Training
To train a model, set the corresponding configuration file, then simply run:

```bash
python train.py --config config.json
```

The log files will be saved in `saved\runs` and the `.pth` chekpoints in `saved\`

```bash
tensorboard --logdir saved
```


## Code structure

  ```
  main-repository/
  │
  ├── train.py - main script to start training
  ├── trainer.py - the main trained
  ├── config.json - holds configuration for training
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │
  ├── dataloader/ - loading the data for different segmentation datasets
  │
  ├── models/ - contains semantic segmentation models
  │
  ├── saved/
  │   ├── runs/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │  
  └── utils/ - small utility functions
      ├── losses.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      └── lr_scheduler - learning rate schedulers 
  ```


## Acknowledgement

The repository is a derivative of [pytorch-segmentation](https://github.com/yassouali/pytorch-segmentation) and has been used for experimentation of Satellite Image Segmentation on Massuchussets Datset. 