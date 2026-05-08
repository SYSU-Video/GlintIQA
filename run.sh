python scripts/train.py \
    --dataset clive \
    --dataset-root /public/datasets/iqadataset/LIVEC \
    --output-dir ./outputs/live_glintiqa \
    --epochs 300 \
    --train-test-round 10 \
    --seed 12345

###############################################
# python scripts/cross_dataset.py \
#     --train-dataset kadid-10k \
#     --train-dataset-root /path/to/kadid10k \
#     --test-datasets live csiq tid2013 \
#     --test-dataset-roots /path/to/LIVE /path/to/CSIQ /path/to/TID2013 \
#     --output-dir ./outputs/cross_kadid \
#     --epochs 50 \
#     --batch-size 32 \
#     --eval-batch-size 32

###############################################
#   torchrun --nproc_per_node=4 scripts/train_distributed.py \
#     --dataset-root /path/to/SAQT_IQA/images \
#     --label-path /path/to/SAQT_IQA/labels \
#     --similarity-csv /path/to/similarity_img_in_kadid-10k_resnet50.csv \
#     --output-dir ./outputs/pretrain_saqt \
#     --epochs 50 \
#     --batch-size 128 \
#     --val-size 1000 \
#     --amp

#   If you need to run additional tests on KADID-10K, you can explicitly specify:
#   --test-dataset kadid-10k \
#   --test-dataset-root /path/to/kadid10k \
#   --test-every 5
