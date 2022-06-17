target_layers_32 = [
    2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30, 32, 34, 38, 40, 42, 44, 46, 48, 50, 52, 56, 58, 60, 62, 64, 66, 68, 70, 74, 76, 78, 80, 82, 84, 86, 88, 92, 94, 96, 98, 100, 102, 104, 106, 110, 112, 114, 116, 118, 120, 122, 124, 128, 130, 132, 134, 136, 138, 140, 142, 146, 148, 150, 152, 154, 156, 158, 160, 164, 166, 168, 170, 172, 174, 176, 178, 182, 184, 186, 188, 190, 192, 194, 196, 200, 202, 204, 206, 208, 210, 212, 214, 218, 220, 222, 224, 226, 228, 230, 232, 236, 238, 240, 242, 244, 246, 248, 250, 254, 256, 258, 260, 262, 264, 266, 268, 272, 274, 276, 278, 280, 282, 284, 286, 290, 292, 294, 296, 298, 300, 302, 304, 308, 310, 312, 314, 316, 318, 320, 322, 326, 328, 330, 332, 334, 336, 338, 340, 344, 346, 348, 350, 352, 354, 356, 358, 362, 364, 366, 368, 370, 372, 374, 376, 380, 382, 384, 386, 388, 390, 392, 394, 398, 400, 402, 404, 406, 408, 410, 412, 416, 418, 420, 422, 424, 426, 428, 430, 434, 436, 438, 440, 442, 444, 446, 448, 452, 454, 456, 458, 460, 462, 464, 466
]

target_layers_64 = [
    2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30, 32, 34, 38, 40, 42, 44, 46, 48, 50, 52, 56, 58, 60, 62, 64, 66, 68, 70, 74, 76, 78, 80, 82, 84, 86, 88, 92, 94, 96, 98, 100, 102, 104, 106, 110, 112, 114, 116, 118, 120, 122, 124, 128, 130, 132, 134, 136, 138, 140, 142, 146, 148, 150, 152, 154, 156, 158, 160, 164, 166, 168, 170, 172, 174, 176, 178, 182, 184, 186, 188, 190, 192, 194, 196, 200, 202, 204, 206, 208, 210, 212, 214, 218, 220, 222, 224, 226, 228, 230, 232, 236, 238, 240, 242, 244, 246, 248, 250, 254, 256, 258, 260, 262, 264, 266, 268, 272, 274, 276, 278, 280, 282, 284, 286, 290, 292, 294, 296, 298, 300, 302, 304, 308, 310, 312, 314, 316, 318, 320, 322, 326, 328, 330, 332, 334, 336, 338, 340, 344, 346, 348, 350, 352, 354, 356, 358, 362, 364, 366, 368, 370, 372, 374, 376, 380, 382, 384, 386, 388, 390, 392, 394, 398, 400, 402, 404, 406, 408, 410, 412, 416, 418, 420, 422, 424, 426, 428, 430, 434, 436, 438, 440, 442, 444, 446, 448, 452, 454, 456, 458, 460, 462, 464, 466, 470, 472, 474, 476, 478, 480, 482, 484, 488, 490, 492, 494, 496, 498, 500, 502, 506, 508, 510, 512, 514, 516, 518, 520, 524, 526, 528, 530, 532, 534, 536, 538, 542, 544, 546, 548, 550, 552, 554, 556, 560, 562, 564, 566, 568, 570, 572, 574, 578, 580, 582, 584, 586, 588, 590, 592, 596, 598, 600, 602, 604, 606, 608, 610, 614, 616, 618, 620, 622, 624, 626, 628
]

resolution_scale_32 = {
    1: 1/4., 2: 1/2., 3: 1.
}

resolution_scale_64 = {
    1: 1/8., 2: 1/4, 3: 1/2., 4: 1.
}

conv1_filters_per_level_32 = [9216, 9216, 8192]
conv2_filters_per_level_32 = [9216, 9216, 8192]

conv1_filters_per_level_64 = [9216, 9216, 9216, 8192]
conv2_filters_per_level_64 = [9216, 9216, 9216, 8192]

def resolution_scale(dataset):
    if dataset == 'cifar10' or dataset =='imagenet32':
        return resolution_scale_32
    elif dataset == 'imagenet64':
        return resolution_scale_64

def specify_resolution_for_each_conv(conv_idx, num_nn_in_each_level = 18 * 9):
    if conv_idx < num_nn_in_each_level:
        return 1
    elif conv_idx < num_nn_in_each_level * 2:
        return 2 
    elif conv_idx < num_nn_in_each_level * 3:
        return 3
    else:
        return 4

def specify_resolution_for_each_mask_32(mask_idx):
    # mask should be indexed from 1 
    mask_per_block = 3
    num_resblock_in_each_level = 8 * 9
    if mask_idx <= num_resblock_in_each_level * mask_per_block:
        return 1
    elif mask_idx <= num_resblock_in_each_level * 2 * mask_per_block:
        return 2 
    else:
        return 3  

def specify_resolution_for_each_mask_64(mask_idx):
    # mask should be indexed from 1 
    mask_per_block = 3
    num_resblock_in_each_level = 8 * 9
    if mask_idx <= num_resblock_in_each_level * mask_per_block:
        return 1
    elif mask_idx <= num_resblock_in_each_level * 2 * mask_per_block:
        return 2 
    elif mask_idx <= num_resblock_in_each_level * 3 * mask_per_block:
        return 3 
    else:
        return 4

def specify_resolution_for_each_mask(mask_idx, dataset):
    if dataset == 'imagenet32' or dataset == 'cifar10':
        return specify_resolution_for_each_mask_32(mask_idx)
    elif dataset == 'imagenet64':
        return specify_resolution_for_each_mask_64(mask_idx)

def specify_pos_for_each_mask(mask_idx):
    return (mask_idx - 1) % 3
