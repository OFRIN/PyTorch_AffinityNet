import torch

def decompose_2D_for_features(features, num_extensions):
    _, _, h, w = features.size()

    h_per_patch = h // num_extensions
    w_per_patch = w // num_extensions
    
    """
    # 2x2
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+

    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """

    patches = []
    for splitted_features in torch.split(features, h_per_patch, dim=2):
        for patch in torch.split(splitted_features, w_per_patch, dim=3):
            patches.append(patch)

    return patches

def combine_2D_for_images(images_list, num_extensions, use_cuda=True):
    index = 0
    ext_h_list = []
    for _ in range(num_extensions):
        ext_w_list = []
        for _ in range(num_extensions):
            ext_w_list.append(images_list[index])
            index += 1
        ext_h_list.append(torch.cat(ext_w_list, dim=3))
    images = torch.cat(ext_h_list, dim=2)
    if use_cuda: images = images.cuda()
    return images

def combine_for_labels(labels_list, use_cuda=True):
    labels = torch.max(torch.stack(labels_list), dim=0)[0]
    if use_cuda: labels = labels.cuda()
    return labels