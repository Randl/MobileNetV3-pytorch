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
python3 -m torch.distributed.launch --nproc_per_node=8 imagenet.py --dataroot "/path/to/imagenet/" --warmup 5 --sched cosine -lr 0.2 -b 128 -d 5e-5 --world-size 8 --seed 42
```

To continue training from checkpoint
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "/path/to/checkpoint/folder"
```
## Results

TODO

|Classification Checkpoint | MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|  Inference time|
|--------------------------|------------|---------------|---------------|---------------|---------------|---------------|----------------|
|MobileNetV3 Large x1.0 224|214.70      |5.145          |          70.88|          89.53|           75.2|              -|               -|
|  [mobilenet_v2_1.0_224](https://github.com/Randl/MobileNetV2-pytorch/)|300         |3.47           |          72.10|          90.48|           71.8|           91.0|               -|
You can test it with
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "" -e
```

## Other implementations
- https://github.com/d-li14/mobilenetv3.pytorch : 73.152% top-1, but with more FLOPs
## Code used
- [DropBlock implementation](https://github.com/miguelvr/dropblock/blob/master/dropblock/dropblock.py) by [miguelvr](https://github.com/miguelvr)
- [FLOPS calculator](https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/utils/flops_benchmark.py) by [warmspringwinds](https://github.com/warmspringwinds)
- [Utility function for divisibility](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py) by [Tensorflow](https://github.com/tensorflow)
- [Squeeze-Excitation block](https://github.com/jonnedtc/Squeeze-Excitation-PyTorch/blob/master/networks.py) by [jonnedtc](https://github.com/jonnedtc)
- [Custom cross-entropy](https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py) by [eladhoffer](https://github.com/eladhoffer)
- [Shadow weights implementation](https://github.com/eladhoffer/utils.pytorch/blob/ca6a47a7766c50930a607d8425216d39104b7664/optim.py) by [eladhoffer](https://github.com/eladhoffer)
