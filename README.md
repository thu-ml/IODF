# Integer-Only Discrete Flows (ICML 2022) [arxiv]

This repository contains Pytorch implementation of experiments from the paper [Fast Lossless Neural Compression with Integer-Only Discrete Flows](http://101.6.240.88:9000/project/62970673d23e15008fcc652d). The implementation is based on [Integer Discrete Flows](https://github.com/jornpeters/integer_discrete_flows). rANS entropy coding in C language is based on [local bits back](https://github.com/hojonathanho/localbitsback).

## Main Dependency
* Python >= 3.7
* Pytorch 1.9.0
* TensorRT 8.2.0.6 + CUDA 10.2

## Usage
Turn LSQ modules in IODF into FakeQuantize modules:
```
python scripts/init_fakequant_model.py --dataset imagenet64 --batchsize 512 --nn_type resnet --resume your_path  --from_lsq
```

Build engine: 
```
python exp_build_engine.py --nn_type resnet --dataset imagenet64 --batchsize 32 --resume your_path --quantize
```

## Contact 

Please open an issue.

## Cite
to do. 