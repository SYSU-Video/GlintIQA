import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class LabelGeneratorConfig:
    """Configuration for label generation process."""
    kadid_path: Path
    similarity_file: Path
    output_dir: Path
    batch_size: int = 1000
    dist_types: int = 25
    dist_levels: int = 5
    reference_shape: Tuple[int, int] = (81, 125)  # Shape for KADID-10K reference matrix

    def __post_init__(self):
        """Convert string paths to Path objects and create output directory."""
        self.kadid_path = Path(self.kadid_path)
        self.similarity_file = Path(self.similarity_file)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class LabelGenerator:
    """Generates labels for distorted images based on semantic similarity."""

    def __init__(self, config: LabelGeneratorConfig):
        """
        Initialize label generator with configuration.

        Args:
            config: Configuration object containing necessary parameters
        """
        self.config = config
        self.dmos_scores = self._load_and_normalize_dmos()
        self.similarity_info = self._load_similarity_info()

    @staticmethod
    def normalize_minmax(data: np.ndarray) -> np.ndarray:
        """
        Perform min-max normalization on input data.

        Args:
            data: Input array to normalize

        Returns:
            Normalized array in range [0, 1]
        """
        data_range = np.max(data) - np.min(data)
        return (data - np.min(data)) / data_range

    def _load_and_normalize_dmos(self) -> np.ndarray:
        """
        Load and normalize DMOS scores from KADID-10K dataset.

        Returns:
            Reshaped matrix of normalized DMOS scores
        """
        dmos_path = self.config.kadid_path / 'dmos.csv'
        dmos = np.asarray(pd.read_csv(dmos_path)['dmos'])
        normalized_dmos = 9 * self.normalize_minmax(dmos)
        return normalized_dmos.reshape(*self.config.reference_shape)

    def _load_similarity_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load similarity information between KADIS and KADID images.

        Returns:
            Tuple of (selected KADIS images, selected KADID reference images)
        """
        similarity_data = np.asarray(pd.read_csv(self.config.similarity_file))
        return similarity_data[:, 0], similarity_data[:, 1]

    def _generate_distorted_image_name(
            self,
            ref_image: str,
            folder_idx: int,
            dist_type: int,
            level: int
    ) -> str:
        """Generate filename for distorted image."""
        img_ref = ref_image.split('.')[0]
        return f'{folder_idx:03d}/{img_ref}_{dist_type:02d}_{level:02d}.bmp'

    def _process_batch(
            self,
            start_idx: int,
            batch_size: int
    ) -> Tuple[List[str], np.ndarray]:
        """
        Process a batch of images and generate their labels.

        Args:
            start_idx: Starting index for the batch
            batch_size: Size of the batch to process

        Returns:
            Tuple of (distorted image names, corresponding labels)
        """
        dist_imgs = []
        dist_labels = np.zeros(batch_size * self.config.dist_types * self.config.dist_levels)

        for i in range(start_idx, start_idx + batch_size):
            if i >= len(self.similarity_info[0]):
                break

            folder_idx = int(np.ceil((i + 1) / self.config.batch_size))
            kadid_ref_index = int(self.similarity_info[1][i][1:3]) - 1
            img_labels = self.dmos_scores[kadid_ref_index, :]

            # Generate distorted image names
            for dist_type in range(1, self.config.dist_types + 1):
                for level in range(1, self.config.dist_levels + 1):
                    dist_imgs.append(
                        self._generate_distorted_image_name(
                            self.similarity_info[0][i],
                            folder_idx,
                            dist_type,
                            level
                        )
                    )

            # Assign labels
            batch_offset = (i % self.config.batch_size) * (self.config.dist_types * self.config.dist_levels)
            dist_labels[batch_offset:batch_offset + len(img_labels)] = img_labels

        return dist_imgs, dist_labels

    def generate_labels(self):
        """Generate and save labels for all images in batches."""
        total_images = len(self.similarity_info[0])
        num_batches = int(np.ceil(total_images / self.config.batch_size))

        print(f"Generating labels for {total_images} images in {num_batches} batches...")

        for batch in tqdm(range(num_batches)):
            start_idx = batch * self.config.batch_size
            dist_imgs, dist_labels = self._process_batch(start_idx, self.config.batch_size)

            # Save batch results
            if dist_imgs:
                df = pd.DataFrame({
                    'dist_img': dist_imgs,
                    'label': dist_labels
                })
                output_file = self.config.output_dir / f'dataset_{batch + 1:03d}.csv'
                df.to_csv(output_file, index=False)
                print(f"Saved batch {batch + 1} to {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate labels for distorted images')
    parser.add_argument('--kadid_path', type=str, required=True,
                        help='Path to KADID-10K dataset')
    parser.add_argument('--similarity_file', type=str, required=True,
                        help='Path to similarity information CSV file')
    parser.add_argument('--output_dir', type=str, default='./labels',
                        help='Directory to save generated labels')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size for processing')
    return parser.parse_args()


def main():
    """Main function to run label generation."""
    args = parse_args()

    config = LabelGeneratorConfig(
        kadid_path=args.kadid_path,
        similarity_file=args.similarity_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

    generator = LabelGenerator(config)
    generator.generate_labels()


if __name__ == "__main__":
    main()

    '''
    python step3_labels_transfer.py  --kadid_path E:\iqadataset\kadid10k  --similarity_file ./info_save/similarity_img_in_kadid-10k.csv  --output_dir ./labels --batch_size 1000
    '''
