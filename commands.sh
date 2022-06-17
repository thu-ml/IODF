# evaluate bpd
python run_evaluate_bpd.py --nn_type resnet --dataset imagenet64 --batchsize 500 --resume IODF --pruned 

# evaluate torch inference 
python exp_torch_latency.py --nn_type densenet --dataset imagenet32 --batchsize 16 --resume base_8bit

## build trt engine
python exp_build_engine.py --nn_type resnet --dataset imagenet64 --batchsize 16 --resume base_8bit --quantize
 
## test inference latency 
python run_test_latency_inference.py --dataset imagenet32 --batchsize 16 --engine_path resnet_q_np-base_asym 

## load fakequant model
python scripts/init_fakequant_model.py --dataset imagenet64 --batchsize 512 --nn_type densenet --resume base  