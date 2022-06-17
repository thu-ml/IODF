from PIL import Image

import numpy as np 

def read_img(s):
    dp = '../DATASETS/%s_wild.ppm' % s
    img = Image.open(dp)
    w,h = img.size
    img = np.array(img.getdata()).reshape(w,h,3).transpose(2,0,1)
    return img, w, h 

def load_wild_32(reso):
    img, w, h = read_img(reso)

    w, h = img.shape[1:]
    m, n = w // 32, h // 32

    patches = []
    for i in range(m):
        for j in range(n):
            patch = img[:, i*32:(i+1)*32, j*32:(j+1)*32]
            # print(patch.shape)
            patches.append(patch)

    res = img[:,:,n*32:h]
    cat_imgs = []
    for i in range(m):
        cat_imgs.append(res[:,i*32:(i+1)*32, :])
    cat_imgs = np.concatenate(cat_imgs, axis=-1)

    n = cat_imgs.shape[-1] // 32
    for i in range(n):
        patches.append(cat_imgs[:,:,i*32:(i+1)*32])

    patches = np.stack(patches, axis = 0)

    return patches

def load_wild_64(reso):
    img, w, h = read_img(reso)

    m, n = w // 64, h // 64

    r = 64 - (h - n * 64)

    print(r)

    img = np.concatenate([img, np.zeros([3, w, r])], axis=-1)

    n += 1

    patches = []
    for i in range(m):
        for j in range(n):
            patch = img[:, i*64:(i+1)*64, j*64:(j+1)*64]
            # print(patch.shape)
            patches.append(patch)

    patches = np.stack(patches, axis = 0)

    return patches

def load_wild(reso, size):
    if size == 32:
        return load_wild_32(reso)
    
    elif size == 64:
        return load_wild_64(reso)


if __name__ == '__main__':
    p = load_wild('4k', 64)
    print(p.shape)