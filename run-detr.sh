python -m torch.distributed.launch --use_env main.py \
    --dataset_file coco \
    --coco_path ./datasets/path/to/coco \
    --output_dir ./datasets/path/to/output \
    --epochs 10
