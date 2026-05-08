# step 0 (optional) ==============================
# python scripts/00_select_kadis_from_iqa_similarity.py \
#   --input-csv ./data/kadis_similarity_img_in_iqadataset.csv \
#   --output-csv ./outputs/retrieval/selected_similarity_img_in_iqadataset.csv \
#   --tid-count 10000 \
#   --csiq-count 10000 \
#   --pipal-count 10000 \
#   --total-images 50000

# step 1 ==============================
# python scripts/01_extract_reference_features.py \
#   --dataset kadid-10k \
#   --dataset-root /public/datasets/iqadataset/kadid10k \
#   --output-root ./outputs \
#   --model-name resnet101 \
#   --batch-size 32


# step 2 ==============================
# python scripts/02_retrieve_semantic_matches.py \
#   --source-image-root /public/server_huang/dataset/kadis700k/kadis700k/ref_imgs \
#   --source-list data/selected_kadis_imgs50k.txt \
#   --reference-feature-dir ./outputs/features/kadid-10k/resnet101 \
#   --model-name resnet101 \
#   --output-root ./outputs \
#   --top-k 1 \
#   --batch-size 1

# step 3 ==============================
# python scripts/03_generate_saqt_labels.py \
#   --kadid-dmos /public/datasets/iqadataset/kadid10k/dmos.csv \
#   --matches-csv ./outputs/retrieval/kadis700k_to_kadid-10k/resnet101/top1_semantic_matches.csv \
#   --output-dir ./outputs/labels/saqt-iqa/top1 \
#   --batch-size 1000