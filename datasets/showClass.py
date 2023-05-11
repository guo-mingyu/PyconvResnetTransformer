import argparse
from pycocotools.coco import COCO

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--ann_file', type=str, required=True, help='Path to COCO annotation file')
args = parser.parse_args()

# 加载COCO 2017数据集的标注文件
coco = COCO(args.ann_file)

# 获取数据集中所有类别的名称
cats = coco.loadCats(coco.getCatIds())
cat_names = [cat['name'] for cat in cats]

# 打印类别名称
print("COCO 2017数据集中的类别：")
print(cat_names)
