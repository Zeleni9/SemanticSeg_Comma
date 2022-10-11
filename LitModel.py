import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np 
import argparse
from pathlib import Path
from typing import  Union
import torch
import pytorch_lightning as pl
from typing import Callable, List
import segmentation_models_pytorch as smp

class LitModel(pl.LightningModule):
    """Transfer Learning
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 backbone: str = 'efficientnet-b0',
                 augmentation_level: str = 'light',
                 batch_size: int = 32,
                 lr: float = 1e-4,
                 eps: float = 1e-7,
                 height: int = 14*32,
                 width: int = 18*32,
                 num_workers: int = 6, 
                 epochs: int = 50, 
                 gpus: int = 1, 
                 weight_decay: float = 1e-3,
                 class_values: List[int] = [41,  76,  90, 124, 161, 0] # 0 added for padding
                 ,**kwargs) -> None:
        
        super().__init__()
        self.data_path = Path(data_path)
        self.epochs = epochs
        self.backbone = backbone
        self.batch_size = batch_size
        self.lr = lr
        self.height = height
        self.width = width
        self.num_workers = num_workers
        self.gpus = gpus
        self.weight_decay = weight_decay
        self.eps = eps
        self.class_values = class_values 
        self.class_colors = np.array([[32, 32, 64], [0, 0, 255], [255, 0, 204], [ 96, 128, 128], [102, 255, 0], [255, 255, 255]])
        self.augmentation_level = augmentation_level 
        
        self.save_hyperparameters()

        self.train_custom_metrics = {'train_acc': smp.utils.metrics.Accuracy(activation='softmax2d')}
        self.validation_custom_metrics = {'val_acc': smp.utils.metrics.Accuracy(activation='softmax2d')}

        self.preprocess_fn = smp.encoders.get_preprocessing_fn(self.backbone, pretrained='imagenet')
        
        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""
        # 1. net:
        self.net = smp.Unet(self.backbone, classes=len(self.class_values), 
                            activation=None, encoder_weights='imagenet')
        # 2. Loss:
        self.loss_func = lambda x, y: torch.nn.CrossEntropyLoss()(x, torch.argmax(y,axis=1))

    def forward(self, x):
        """Forward pass. Returns logits."""
        x = self.net(x)
        return x


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='efficientnet-b0',
                            type=str,
                            metavar='BK',
                            help='Name as in segmentation_models_pytorch')
        parser.add_argument('--augmentation-level',
                            default='light',
                            type=str,
                            help='Training augmentation level c.f. retiriever')
        parser.add_argument('--data-path',
                            default='/home/yyousfi1/commaai/comma10k',
                            type=str,
                            metavar='dp',
                            help='data_path')
        parser.add_argument('--epochs',
                            default=30,
                            type=int,
                            metavar='N',
                            help='total number of epochs')
        parser.add_argument('--batch-size',
                            default=32,
                            type=int,
                            metavar='B',
                            help='batch size',
                            dest='batch_size')
        parser.add_argument('--gpus',
                            type=int,
                            default=1,
                            help='number of gpus to use')
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=1e-4,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--eps',
                            default=1e-7,
                            type=float,
                            help='eps for adaptive optimizers',
                            dest='eps')
        parser.add_argument('--height',
                            default=14*32,
                            type=int,
                            help='image height')
        parser.add_argument('--width',
                            default=18*32,
                            type=int,
                            help='image width')
        parser.add_argument('--num-workers',
                            default=6,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--weight-decay',
                            default=1e-3,
                            type=float,
                            metavar='wd',
                            help='Optimizer weight decay')

        return parser
