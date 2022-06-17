# Integer-Only Discrete Flows (ICML 2022) [arxiv]

This repository contains Pytorch implementation of experiments from the paper [Fast Lossless Neural Compression with Integer-Only Discrete Flows](http://101.6.240.88:9000/project/62970673d23e15008fcc652d). The implementation is based on [Integer Discrete Flows](https://github.com/jornpeters/integer_discrete_flows). rANS entropy coding in C language is based on [local bits back](https://github.com/hojonathanho/localbitsback).

## Main Dependency
* Python >= 3.7
* Pytorch 1.9.0
* TensorRT 8.2.0.6 + CUDA 10.2

## Usage
<!-- Basic training IODF and coding with IODF:
```
python run_train.py --nn_type resnet --dataset imagenet32 --batchsize 256

python run_coding.py --nn_type resnet --dataset imagenet32 --batchsize 500 --resume base --no_decode
``` -->
* Follow training procedure described by Algorithm.1 in the paper. Refer to [commands.sh](./commands.sh) for detailed scripts. 

* For TensorRT Implementation, switch to branch trt.

## Contact 

Please open an issue. 

## Cite
to do. 