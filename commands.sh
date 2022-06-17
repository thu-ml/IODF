## train 
python run_train.py --nn_type resnet --dataset imagenet32 --batchsize 256

## initialize pruning model: add binary gates into convolution layers, and train the gated model. 
python scripts/init_pruning_conv1_model.py --nn_type resnet --dataset imagenet32 --resume your_path
python run_train.py --nn_type resnet --dataset imagenet32 --batchsize 256 --pruning --resume your_path

## remove binary gates and disabled filters and finetune 
python scripts/remove_masks_in_pruning_model.py --nn_type resnet --dataset imagenet32 --resume your_path --batchsize 500 --epoch 1 --pmode conv1
python run_train.py --nn_type resnet --dataset imagenet32 --batchsize 256 --pruned --resume your_path

## initialize activation quantized model and finetune 
python scripts/init_a_lsq_model.py --nn_type resnet --dataset imagenet32 --resume your_path
python run_train.py --nn_type resnet --dataset imagenet32 --batchsize 256 --quantize --resume your_path

## initialize activation and weights both quantized model and finetune 
python scripts/init_aw_lsq_model.py --nn_type resnet --dataset imagenet32 --resume your_path 
python run_train.py --nn_type resnet --dataset imagenet32 --batchsize 256 --quantize --resume your_path

## evaluate coding
python run_coding.py --nn_type resnet --dataset imagenet32 --batchsize 500 --resume base --no_decode