# MobileNetV3 in PyTorch

An implementation of `MobileNetV3` in PyTorch. `MobileNetV3` is an efficient
convolutional neural network architecture for mobile devices. For more information check the paper:
[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

## Usage

Clone the repo:
```bash
git clone https://github.com/Randl/MobileNetV3-pytorch
pip install -r requirements.txt
```

Use the model defined in `MobileNetV3.py` to run ImageNet example:
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 imagenet.py --dataroot "/path/to/imagenet/" --sched clr -b 128 --seed 42 --world-size 8 --sync-bn```
```

To continue training from checkpoint
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "/path/to/checkpoint/folder"
```

## Results

WIP

|Classification Checkpoint | MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|  Inference time|
|--------------------------|------------|---------------|---------------|---------------|---------------|---------------|----------------|
|MobileNetV3 Large x1.0 224|219.80      |5.481          |          73.53|          91.14|           75.2|              -|               ~258ms|
|  [mobilenet_v2_1.0_224](https://github.com/Randl/MobileNetV2-pytorch/)|300         |3.47           |          72.10|          90.48|           71.8|           91.0|               ~461ms|

Inference time is for single 1080 ti per batch of 128.

You can test it with
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "results/mobilenetv3large-v1/model_best0.pth.tar" -e
```

## Other implementations
- https://github.com/d-li14/mobilenetv3.pytorch : 73.152% top-1, with more FLOPs
- https://github.com/xiaolai-sqlai/mobilenetv3 : 75.45% top-1, even more FLOPs
- https://github.com/rwightman/gen-efficientnet-pytorch : 75.634% top-1, seems to be right FLOPs

## Code used
- [DropBlock implementation](https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py) by [miguelvr](https://github.com/miguelvr)
- [FLOPS calculator](https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/utils/flops_benchmark.py) by [warmspringwinds](https://github.com/warmspringwinds)
- [Utility function for divisibility](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py) by [Tensorflow](https://github.com/tensorflow)
- [Squeeze-Excitation block](https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py) by [jonnedtc](https://github.com/jonnedtc)
- [Custom cross-entropy](https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py) by [eladhoffer](https://github.com/eladhoffer)
- [Shadow weights implementation](https://github.com/eladhoffer/utils.pytorch/blob/ca6a47a7766c50930a607d8425216d39104b7664/optim.py) by [eladhoffer](https://github.com/eladhoffer)
