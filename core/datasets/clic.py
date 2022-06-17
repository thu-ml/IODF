from concurrent.futures import process
import numpy as np 
import os
from PIL import Image
from tqdm import tqdm

def read_img(dp):
    img = np.asarray(Image.open(dp)).transpose(2, 0, 1).copy()
    w,h,c = img.shape
    # imageio.imsave('./clic_orig.png', img)
    # exit()
    return img, w, h

def padding(img_array, reso):
    c, w, h = img_array.shape
    m, n = w // reso, h // reso
    r_w, r_h = reso - (w - m * reso),  reso - ( h - n * reso)
    if r_w < reso:
        img_array = np.concatenate([img_array, np.zeros([3, r_w, h])], axis = 1)
        w = w + r_w
    if r_h < reso:
        img_array = np.concatenate([img_array, np.zeros([3, w, r_h])], axis = 2)
    
    return img_array

def patch(img_array, reso):
    patches = []
    c, w, h = img_array.shape
    m, n = w // reso, h // reso
    assert w == m * reso and h == n * reso, isinstance(img_array, np.ndarray)

    for i in range(m):
        for j in range(n):
            patches.append(img_array[:, i*reso:(i+1)*reso, j*reso:(j+1)*reso])
    patches=np.stack(patches, axis=0)

    return patches

def load_clic(reso):
    assert reso in [32, 64], 'not compatible resolution'

    datapath = f'../DATASETS/clic_profession_test_2021/reso_{reso}'
    patch_names = os.listdir(datapath)
    patches = []
    for name in patch_names:
        patches.append(np.load(os.path.join(datapath, name)))
    return patches

def process_clic(reso):
    assert reso in [32, 64], 'not compatible resolution'

    datapath = '../DATASETS/clic_profession_test_2021'
    img_names = [s for s in os.listdir(datapath) if s.endswith('.png')]
    print(len(img_names), 'images')
    npy_path = os.path.join(datapath, f'reso_{reso}')
    os.makedirs(npy_path, exist_ok=True)
    for name in tqdm(img_names):
        img_array, w, h = read_img(os.path.join(datapath, name))
        patches = patch(padding(img_array, reso), reso)
        np.save(os.path.join(npy_path, name.replace('.png', '')), patches)
    
if __name__ == '__main__':
    # p = load_clic(32)
    # print(p[0].shape)
    process_clic(64)