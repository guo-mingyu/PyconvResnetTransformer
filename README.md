# PyconvResnetTransformer
========
- 下载 COCO 2017 数据集，并将其拆分为训练集和验证集。

- 在训练和验证集中只保留你想要的 10 个类别的数据。你可以使用 COCO 数据集提供的标签信息（例如，包含类别信息的 JSON 文件）来完成这一步。

- 在 DETR 模型中，找到 datasets/coco.py 文件。这是 COCO 数据集的处理代码。在这个文件中，你可以找到一个名为 CocoDetection 的类，它用于读取 COCO 数据集。

- 在 CocoDetection 类的 __getitem__ 方法中，只保留你想要的 10 个类别的数据。你可以在这个方法中使用 COCO 数据集提供的标签信息来实现这一步。

- 在 DETR 模型中，找到 models/detr.py 文件。这是 DETR 模型的主要代码文件。在这个文件中，你需要修改分类器的输出大小，以便仅包含你想要的 10 个类别。你可以在 DETR 类的构造函数中修改分类器的输出大小。

- 在训练 DETR 模型之前，你需要为你的数据集创建一个新的标签映射。你可以在 datasets/coco.py 文件中找到一个名为 get_coco_api_from_dataset 的函数。在这个函数中，你可以创建一个新的标签映射，仅包含你想要的 10 个类别。

- 最后，你可以使用修改后的代码来训练 DETR 模型。你可以使用 PyTorch 提供的训练代码或编写自己的训练脚本来完成这一步。请注意，你可能需要根据你的数据集大小和硬件资源修改训练参数。

进入到WSL的bash环境
在PowerShell中，你可以使用以下命令进入到WSL的bash环境：

Copy code
wsl
然后，你可以在bash中使用以下命令进入到工作目录：

bash
Copy code
cd /home/guomingyu/PyconvResnetTransformer
接着，你可以使用以下命令启动Docker容器，并指定工作目录：

bash
Copy code
docker run --rm -it -v "$(pwd):/workspace" -w /workspace detr

docker run --gpus all --rm -it --shm-size=80g -v "$(pwd):/workspace" -w /workspace detr



其中，$(pwd)表示当前目录，也就是/home/guomingyu/PyconvResnetTransformer。-v选项用于挂载当前目录到容器中的/workspace目录下，-w选项用于指定容器的工作目录为/workspace。




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



