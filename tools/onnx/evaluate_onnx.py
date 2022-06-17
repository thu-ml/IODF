import torch
import onnxruntime as ort
import numpy as np 
import tqdm

def evaluate_onnx(onnx_path, data_loader, z_idx=0):
    zs = []
    bpds = []
    
    sess = ort.InferenceSession(onnx_path)
    ip_name = 'x'
    op_name = [o.name for o in sess.get_outputs()]
    
    for i, (x, _) in tqdm.tqdm(enumerate(data_loader), desc='Evaluating onnx...'):
        if i == 10:
            break
        x = x.numpy().astype('float32')
        bs = x.shape[0]
        res = sess.run(op_name, {ip_name: x})
        z = res[z_idx].reshape(bs, -1)
        bpd = res[-1].reshape(bs, -1)
        zs.append(z)
        bpds.append(bpd)

    zs = np.concatenate(zs, axis=0) # used for checking if model is correct.
    np.set_printoptions(threshold=np.inf)
    print(zs[2] * 256)

    bpds = np.concatenate(bpds, axis=0)
    return bpds