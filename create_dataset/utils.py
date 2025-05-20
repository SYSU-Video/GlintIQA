from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torchvision
from PIL import Image
from torch import nn
import argparse


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    NORMALIZE_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
    NORMALIZE_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)
    DEVICE: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    IMAGE_EXTENSIONS: Tuple[str, ...] = (
        '.png', '.bmp', '.BMP', '.jpeg', '.JPG', '.jpg', '.gif', '.tiff'
    )

    def __init__(self, args: Optional[argparse.Namespace] = None):
        """Initialize configuration with optional command line arguments."""
        self.output_dir = getattr(args, 'output_dir', './info_save')
        self.model_name = getattr(args, 'model_name', 'resnet50')
        self.dataset_paths = self._initialize_dataset_paths(args)

    def _initialize_dataset_paths(self, args: Optional[argparse.Namespace]) -> Dict[str, str]:
        """Initialize dataset paths with potential command line overrides."""
        base_paths = {
            'live': 'E:\iqadataset\LIVE',
            'csiq': 'E:\iqadataset\CSIQ',
            'tid2013': 'E:\iqadataset\TID2013',
            'kadid-10k': 'E:\iqadataset\kadid10k',
        }

        if args and hasattr(args, 'dataset_paths'):
            base_paths.update(args.dataset_paths)

        return base_paths

    @property
    def reference_folders(self) -> Dict[str, str]:
        """Define reference folder names for each dataset."""
        return {
            'live': 'refimgs',
            'csiq': 'src_imgs',
            'tid2013': 'reference_images',
            'kadid-10k': 'ref_imgs'
        }


class ImageProcessor:
    """Base class for image loading and processing."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.transforms = self._get_transforms()

    def _get_transforms(self) -> torchvision.transforms.Compose:
        """Get the image transformation pipeline."""
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=self.config.NORMALIZE_MEAN,
                std=self.config.NORMALIZE_STD
            )
        ])

    @staticmethod
    def load_image(path: str) -> Image.Image:
        """Load an image from path and convert to RGB."""
        with Image.open(path) as img:
            return img.convert('RGB')

    def process_image(self, path: str) -> torch.Tensor:
        """Load and transform an image."""
        img = self.load_image(path)
        return self.transforms(img)


class ModelBase:
    """Base class for neural network models."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.model = self._initialize_model()
        self.feature_dim = 2048 if config.model_name in ['resnet101', 'resnet50'] else 512

    def _initialize_model(self) -> nn.Module:
        """Initialize and configure the ResNet model."""
        resnet = getattr(torchvision.models, self.config.model_name)(pretrained=True)
        model = nn.Sequential(*list(resnet.children())[:-1])
        model.eval()
        return model.to(self.config.DEVICE)


def get_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with common arguments."""
    parser = argparse.ArgumentParser(description='Image Processing Arguments')
    parser.add_argument('--output_dir', type=str, default='./info_save',
                        help='Directory to save output files')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        choices=['resnet34', 'resnet50', 'resnet101'],
                        help='Name of the ResNet model to use')
    parser.add_argument('--dataset', type=str, default='kadid-10k',
                        help='Dataset to process')
    parser.add_argument('--kadis_ref_imgs_path', type=str,
                        default='/path/to/kadis/dataset',
                        help='Path to KADIS dataset')
    return parser