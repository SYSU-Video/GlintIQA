# SAQT-IQA Dataset Generation

This folder contains the dataset generation pipeline for the
Semantic-Aligned Quality Transfer (SAQT) method used by GlintIQA.

The main pipeline builds SAQT-IQA labels in three Python steps:

1. Extract semantic features from KADID-10K reference images.
2. Retrieve the semantically closest KADID-10K reference image for each selected KADIS image.
3. Transfer KADID-10K quality labels to the selected KADIS images under the same distortion type and level.

An optional pre-selection script can be used before Step 2 when the selected
KADIS image list comes from a multi-dataset similarity table.

Distorted image rendering is still handled by the existing MATLAB scripts in
`create_dataset/gengerate_dis_img`. 

## Folder Layout

```text
dataset_generation/
+-- create_dataset/
    +-- data/
    |   +-- selected_kadis_imgs50k.txt
    |   +-- kadis_similarity_img_in_iqadataset.csv
    +-- saqt/                         
    +-- scripts/
    |   +-- 00_select_kadis_from_iqa_similarity.py # Optional
    |   +-- 01_extract_reference_features.py
    |   +-- 02_retrieve_semantic_matches.py
    |   +-- 03_generate_saqt_labels.py
    |   +-- 04_validate_saqt_outputs.py
    +-- gengerate_dis_img/            # MATLAB distortion generation
```

`scripts/00_select_kadis_from_iqa_similarity.py` is the renamed optional
selection utility for processing multi-dataset similarity files.

## Expected Data

Prepare the following datasets before running the pipeline:

```text
/path/to/kadid10k/
+-- ref_imgs/
+-- dmos.csv

/path/to/kadis700k/
+-- ref_imgs/
```

If you want to process only a selected subset of KADIS images, create a text
file with one image name per line:

```text
selected_kadis_images.txt
```

Example:

```text
flower-883226.png
city-123456.png
...
```

## Recommended Output Layout

Use a writable output root on a disk with enough space:

```text
outputs/
+-- features/
|   +-- kadid-10k/
|       +-- resnet101/
|           +-- reference_features.npy
|           +-- reference_images.csv
+-- retrieval/
|   +-- kadis700k_to_kadid-10k/
|       +-- resnet101/
|           +-- top1_semantic_matches.csv
+-- labels/
    +-- saqt-iqa/
        +-- top1/
            +-- labels_batch_0001.csv
            +-- labels_batch_0002.csv
            +-- manifest.json
```

## Environment

The default semantic backbone is `resnet101`, matching the paper description.
Other torchvision ResNet variants are also supported: `resnet18`, `resnet34`,
`resnet50`, `resnet101`, and `resnet152`.

## Optional Step 0: Select KADIS Images

Use this script only if you already have a multi-dataset similarity CSV and
want to select a fixed number of KADIS images before semantic retrieval:

```bash
python scripts/00_select_kadis_from_iqa_similarity.py \
  --input-csv ./data/kadis_similarity_img_in_iqadataset.csv \
  --output-csv ./outputs/retrieval/selected_similarity_img_in_iqadataset.csv \
  --tid-count 10000 \
  --csiq-count 10000 \
  --pipal-count 10000 \
  --total-images 50000
```

The script also accepts the old argument names, such as `--input_file`,
`--output_file`, and `--total_images`, for compatibility.

## Step 1: Extract KADID-10K Reference Features

Run this command from `dataset_generation/create_dataset`:

```bash
python scripts/01_extract_reference_features.py \
  --dataset kadid-10k \
  --dataset-root /path/to/kadid10k \
  --output-root ./outputs \
  --model-name resnet101 \
  --batch-size 32
```

Output:

```text
outputs/features/kadid-10k/resnet101/reference_features.npy
outputs/features/kadid-10k/resnet101/reference_images.csv
```

`reference_features.npy` stores L2-normalized semantic features.
`reference_images.csv` stores the image names and full paths in the same order.

## Step 2: Retrieve Semantic Matches

Run nearest-neighbor retrieval from selected KADIS images to KADID-10K references:

```bash
python scripts/02_retrieve_semantic_matches.py \
  --source-image-root /path/to/kadis700k/ref_imgs \
  --source-list ./data/selected_kadis_images.txt \         # or use "selected_similarity_img_in_iqadataset.csv"
  --reference-feature-dir ./outputs/features/kadid-10k/resnet101 \
  --output-root ./outputs \
  --model-name resnet101 \
  --top-k 1 \
  --batch-size 32
```

If `--source-list` is omitted, all images directly under `--source-image-root`
are processed. Add `--recursive` if images are stored in nested folders.

Output:

```text
outputs/retrieval/kadis700k_to_kadid-10k/resnet101/top1_semantic_matches.csv
```

CSV columns:

```text
source_image,source_path,target_reference,target_reference_path,semantic_distance,cosine_similarity,rank
```

## Step 3: Generate SAQT-IQA Labels

Transfer KADID-10K MOS values to selected KADIS images:

```bash
python scripts/03_generate_saqt_labels.py \
  --kadid-dmos /path/to/kadid10k/dmos.csv \
  --matches-csv ./outputs/retrieval/kadis700k_to_kadid-10k/resnet101/top1_semantic_matches.csv \
  --output-dir ./outputs/labels/saqt-iqa/top1 \
  --batch-size 1000
```

Output files:

```text
outputs/labels/saqt-iqa/top1/labels_batch_0001.csv
outputs/labels/saqt-iqa/top1/manifest.json
```

Label CSV columns:

```text
distorted_image,quality_score,source_image,matched_reference,distortion_type,distortion_level
```

`quality_score` is normalized to the range `[0, 9]`. Each source image produces
`25 x 5 = 125` label rows by default.

## Step 4: Generate Distorted Images

Use the existing MATLAB scripts in:

```text
create_dataset/gengerate_dis_img
```

The generated distorted image names should match the `distorted_image` column
from the label CSV files:

```text
001/source_image_name_01_01.bmp
001/source_image_name_01_02.bmp
...
```

## Validate Outputs

You can validate any combination of feature, retrieval, and label outputs:

```bash
python scripts/04_validate_saqt_outputs.py \
  --feature-dir ./outputs/features/kadid-10k/resnet101 \
  --matches-csv ./outputs/retrieval/kadis700k_to_kadid-10k/resnet101/top1_semantic_matches.csv \
  --label-dir ./outputs/labels/saqt-iqa/top1
```

The validator checks required files, required CSV columns, feature-image count
alignment, and label row counts recorded in `manifest.json`.