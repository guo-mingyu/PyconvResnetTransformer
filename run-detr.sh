python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --coco_path /dataset/path/to/coco