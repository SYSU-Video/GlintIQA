import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class ProcessingConfig:
    """Configuration for similarity data processing."""
    input_file: Path
    output_file: Path
    dataset_selection_counts: Dict[str, int]
    total_images: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ProcessingConfig':
        """Create configuration from command line arguments."""
        return cls(
            input_file=Path(args.input_file),
            output_file=Path(args.output_file),
            dataset_selection_counts={
                'tid2013': args.tid_count,
                'csiq': args.csiq_count,
                'pipal': args.pipal_count
            },
            total_images=args.total_images
        )


class SimilarityDataProcessor:
    """Process and select images based on similarity scores."""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize processor with configuration.

        Args:
            config: Processing configuration parameters
        """
        self.config = config
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load similarity data from CSV file."""
        if not self.config.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")
        return pd.read_csv(self.config.input_file)

    def _get_sorted_indices(self, similarity_scores: np.ndarray) -> np.ndarray:
        """Get indices sorted by similarity scores."""
        return np.argsort(similarity_scores)

    def _select_top_images(self, source_images: np.ndarray,
                           sorted_indices: np.ndarray,
                           count: int) -> List[str]:
        """Select top N images based on sorted similarity indices."""
        return source_images[sorted_indices[-count:][::-1]].tolist()

    def process_similarities(self) -> Tuple[List[str], List[str], List[float]]:
        """
        Process similarity data and select images.

        Returns:
            Tuple containing selected KADIS images, reference images, and similarity scores
        """
        # Extract data columns
        source_images = self.data['kadis'].values
        dataset_data = {
            dataset: {
                'ref': self.data[dataset].values,
                'sim': self.data[f'sim_value_{dataset}'].values
            }
            for dataset in self.config.dataset_selection_counts.keys()
        }

        # Get KADID-10K specific data
        kadid_refs = self.data['kadid-10k'].values
        kadid_similarities = self.data['sim_value_kadid-10k'].values

        # Process each dataset
        selected_images = []
        for dataset, count in self.config.dataset_selection_counts.items():
            sorted_indices = self._get_sorted_indices(dataset_data[dataset]['sim'])
            selected = self._select_top_images(source_images, sorted_indices, count)
            selected_images.extend(selected)

        # Get unique images
        unique_images = list(np.unique(selected_images))

        # Add remaining images from KADID-10K until reaching total_images
        sorted_kadid_indices = self._get_sorted_indices(kadid_similarities)[::-1]
        for idx in sorted_kadid_indices:
            img = source_images[idx]
            if img not in unique_images:
                unique_images.append(img)
                if len(unique_images) == self.config.total_images:
                    break

        # Get corresponding KADID references and similarity values
        kadid_refs_selected = []
        kadid_sims_selected = []
        for img in tqdm(unique_images, desc="Collecting KADID references"):
            idx = (source_images == img)
            kadid_refs_selected.append(kadid_refs[idx].item())
            kadid_sims_selected.append(kadid_similarities[idx].item())

        return unique_images, kadid_refs_selected, kadid_sims_selected

    def save_results(self, selected_images: List[str],
                     reference_images: List[str],
                     similarity_scores: List[float]) -> None:
        """Save processing results to CSV file."""
        results = np.stack((selected_images, reference_images, similarity_scores)).T
        df = pd.DataFrame(results, columns=['kadis', 'kadid-10k', 'sim_value'])

        self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.config.output_file, index=False)
        print(f"Results saved to {self.config.output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process image similarity data')

    parser.add_argument('--input_file', type=str,
                        default='./info_save/kadis_similarity_img_in_iqadataset.csv',
                        help='Path to input similarity CSV file')
    parser.add_argument('--output_file', type=str,
                        default='./info_save/similarity_img_in_iqadataset.csv',
                        help='Path to output CSV file')
    parser.add_argument('--tid_count', type=int, default=10000,
                        help='Number of images to select from TID2013')
    parser.add_argument('--csiq_count', type=int, default=10000,
                        help='Number of images to select from CSIQ')
    parser.add_argument('--pipal_count', type=int, default=10000,
                        help='Number of images to select from PIPAL')
    parser.add_argument('--total_images', type=int, default=50000,
                        help='Total number of images to select')

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    config = ProcessingConfig.from_args(args)

    processor = SimilarityDataProcessor(config)
    selected_images, reference_images, similarity_scores = processor.process_similarities()
    processor.save_results(selected_images, reference_images, similarity_scores)


if __name__ == "__main__":
    main()
    '''
    python step2_kadis_sim_img_in_iqadatasets_v2.py  --input_file ./info_save/kadis_similarity_img_in_iqadataset.csv --output_file ./info_save/similarity_img_in_iqadataset.csv --tid_count 10000  --csiq_count 10000 --pipal_count 10000 --total_images 50000
    '''

