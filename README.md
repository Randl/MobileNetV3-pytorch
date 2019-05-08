# MobileNetV3 in PyTorch

An implementation of `MobileNetV3` in PyTorch. `MobileNetV3` is an efficient
convolutional neural network architecture for mobile devices. For more information check the paper:
[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

## Usage

Clone the repo:
```bash
git clone https://github.com/Randl/MNASNet-pytorch
pip install -r requirements.txt
```

Use the model defined in `model.py` to run ImageNet example:
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 imagenet.py --dataroot "/path/to/imagenet/" --warmup 5 --sched cosine -lr 0.2 -b 128 -d 5e-5 --world-size 8 --seed 42
```

To continue training from checkpoint
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "/path/to/checkpoint/folder"
```
## Results


|Classification Checkpoint| MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|
|-------------------------|------------|---------------|---------------|---------------|---------------|---------------|

You can test it with
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "results/shufflenet_v2_0.5/model_best.pth.tar" -e
```

## Other implementations

