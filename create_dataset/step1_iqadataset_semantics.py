# step1_iqadataset_semantics.py
from utils import BaseConfig, ImageProcessor, ModelBase, get_argument_parser
import torch
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm


class FeatureExtractor(ModelBase):
    """Class for extracting semantic features from images."""

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.image_processor = ImageProcessor(config)

    def extract_features(self, dataset: str) -> None:
        """Extract features from dataset reference images."""
        ref_path = Path(self.config.dataset_paths[dataset]) / self.config.reference_folders[dataset]

        image_files = [
            f for f in sorted(ref_path.iterdir())
            if f.is_file() and f.suffix.lower() in self.config.IMAGE_EXTENSIONS
        ]

        print(f"Processing {len(image_files)} images from {dataset} dataset...")

        features = np.zeros((len(image_files), self.feature_dim))

        for i, img_path in tqdm(enumerate(image_files)):
            with torch.no_grad():
                img_tensor = self.image_processor.process_image(str(img_path))
                features[i] = F.normalize(
                    self.model(img_tensor.unsqueeze(0).to(self.config.DEVICE))
                    .squeeze().flatten().unsqueeze(0),
                    p=2, dim=1
                ).cpu().numpy()

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output_path / f'{dataset}_ref_semantics.csv',
            features,
            fmt='%f',
            delimiter=' '
        )
        print(f"Features saved to {output_path / f'{dataset}_ref_semantics.csv'}")


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    config = BaseConfig(args)
    extractor = FeatureExtractor(config)
    extractor.extract_features(args.dataset)


if __name__ == "__main__":
    main()
    '''
    # features extraction for kadid-10k
    python step1_iqadataset_semantics.py --dataset kadid-10k --model_name resnet50 --output_dir ./info_save
    '''


