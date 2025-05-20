# step2_kadis_sim_img_in_iqadataset.py
from utils import BaseConfig, ImageProcessor, ModelBase, get_argument_parser
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import List, Tuple


class SimilarityAnalyzer(ModelBase):
    """Class for analyzing semantic similarity between images."""

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self.image_processor = ImageProcessor(config)

    @staticmethod
    def calculate_similarity(
            semantic_vector: torch.Tensor,
            semantic_matrix: torch.Tensor,
            ref_imgs: List[str]
    ) -> Tuple[str, float]:
        """Calculate similarity between vector and matrix."""
        cosine_sim = torch.mm(semantic_vector, semantic_matrix)
        top1_value, top1_idx = torch.topk(cosine_sim, k=1, dim=1)
        sim_img = ref_imgs[int(top1_idx.cpu().numpy())]
        return sim_img, top1_value.cpu().numpy().item(0)

    def analyze_dataset_similarity(
            self,
            kadis_images: np.ndarray,
            kadis_ref_imgs_path: str,
            output_path: str
    ) -> None:
        """Analyze similarity between KADIS images and reference dataset."""
        # Load reference features
        ref_features = np.loadtxt(
            os.path.join(self.config.output_dir, f'{self.config.dataset}_ref_semantics.csv'),
            delimiter=' '
        )
        ref_features = torch.Tensor(ref_features).to(self.config.DEVICE).t()

        # Get reference images
        ref_path = os.path.join(
            self.config.dataset_paths[self.config.dataset],
            self.config.reference_folders[self.config.dataset]
        )
        ref_images = sorted(os.listdir(ref_path))

        results = []
        print(f"Analyzing similarity for {len(kadis_images)} images...")

        for kadis_img in tqdm(kadis_images):
            with torch.no_grad():
                img_tensor = self.image_processor.process_image(
                    os.path.join(kadis_ref_imgs_path, kadis_img)
                )
                features = self.model(img_tensor.unsqueeze(0).to(self.config.DEVICE))
                features = F.normalize(
                    features.squeeze().flatten().unsqueeze(0),
                    p=2, dim=1
                )

                sim_img, sim_value = self.calculate_similarity(
                    features, ref_features, ref_images
                )
                results.append([kadis_img, sim_img, sim_value])

        df = pd.DataFrame(
            results,
            columns=['kadis', f'{self.config.dataset}', 'sim_value']
        )
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


def main():
    parser = get_argument_parser()
    parser.add_argument('--kadis_list', type=str, required=True,
                        help='Path to file containing KADIS image list')
    args = parser.parse_args()

    config = BaseConfig(args)
    analyzer = SimilarityAnalyzer(config)

    kadis_images = np.loadtxt(args.kadis_list, dtype=str)
    print(f"Loaded {len(kadis_images)} KADIS images")
    config.dataset = args.dataset
    output_path = os.path.join(
        config.output_dir,
        f'similarity_img_in_{config.dataset}.csv'
    )

    analyzer.analyze_dataset_similarity(
        kadis_images,
        args.kadis_ref_imgs_path,
        output_path
    )


if __name__ == "__main__":
    main()
    ''' 
    
    python step2_kadis_sim_img_in_iqadataset.py --dataset kadid-10k --kadis_ref_imgs_path F:\kadis\kadis700k\ref_imgs --kadis_list ./info_save/selected_kadis_imgs50k.txt
    '''
