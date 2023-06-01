python -m torch.distributed.launch --use_env main.py \
    --dataset_file coco \
    --coco_path ./datasets/path/to/coco \
    --output_dir ./datasets/path/to/output \
    --epochs 10 \
    --lr 1e-2 \
    --weight_decay 1e-2 \
    --lr_drop 100 \
    --batch_size 4

