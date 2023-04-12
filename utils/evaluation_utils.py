from utils.utils import prepare_model
from utils.utils import gram_matrix
import numpy as np
import torch
import cv2

def calc_cosine_similarity(i1,i2):
    temp = torch.sum(torch.mul(i1,i2)).item()
    n1 = torch.linalg.norm(i1).item()
    n2 = torch.linalg.norm(i2).item()
    return (temp)/(n1*n2)

def content_fidelity(stylized_img, content_img, config, device):
    neural_net, content_feature_maps_index_name, _ = prepare_model(config['model'], device)
    content_img_set_of_feature_maps = neural_net(content_img)
    stylized_img_set_of_feature_maps = neural_net(stylized_img)
    
    content_img_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    stylized_img_representation = stylized_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)

    CF = calc_cosine_similarity(content_img_representation, stylized_img_representation)

    return CF

def global_effects(stylized_img,style_img,config,device):
    neural_net, _, style_feature_maps_indices_names = prepare_model(config['model'], device)
    style_img_set_of_feature_maps = neural_net(style_img.unsqueeze(0))
    target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    stylised_img_set_of_feature_maps = neural_net(stylized_img.unsqueeze(0))
    stylised_gram_matrix = [gram_matrix(x) for cnt, x in enumerate(stylised_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    channels=[0,1,2]
    GC=0
    for channel in channels:
        s1 = cv2.calcHist([style_img.cpu().numpy().transpose((1,2,0))], [channel], None, [256], [0, 256])
        s = cv2.calcHist([stylized_img.detach().cpu().numpy().transpose((1,2,0))], [channel], None, [256], [0, 256])
        GC=GC+(np.sum(s*s1)/(np.linalg.norm(s)*np.linalg.norm(s1)))
    GC=(1/3)*GC
    HT = 0
    for sgm,tsr in zip(stylised_gram_matrix, target_style_representation):
        temp = np.multiply(sgm.detach().cpu(),tsr.cpu()).numpy()
        n1 = np.linalg.norm(sgm.detach().cpu())
        n2 = np.linalg.norm(tsr.cpu())
        HT += np.sum(temp)/(n1*n2)
    HT /= len(stylised_gram_matrix)
    GE = 0.5*(GC+HT)
    return GE

def tupler(l):
    # 3d nested lists to nested tuples
    tup1 = tuple()
    for i in range(len(l)):
        tup2 = tuple()
        for j in range(len(l[0])):
            tup3 = tuple()
            for k in range(len(l[0][0])):
                tup3 += (l[i][j][k],)
            tup2 += (tup3,)
        tup1 += (tup2,)
    return tup1

def patchify_tensor(t):
    # Expects an image like tensor (C,H,W)
    # Returns list of 3x3 patches
    h,w,c = t.shape
    # patches = []
    # for i in range(h-2):
    #     for j in range(w-2):
    #         patches.append(t[i:i+3,j:j+3,:])
    out_shape = (h-2,w-2,3,3,c)
    patches = np.lib.stride_tricks.as_strided(t, shape = out_shape, strides=(t.strides[0],t.strides[1],t.strides[0],t.strides[1],t.strides[2]))
    return patches

class HashNumpyWrapper():
    def __init__(self, npy):
        self.npy = npy

    def __hash__(self):
        return hash(self.npy.tobytes())

    def __eq__(self, other):
        return np.all(self.npy == other.npy)

def get_normalized_cross_correlation_measure_and_layer_lp2(stylized_feature_map, style_feature_map):
    stylized_feature_map = stylized_feature_map.detach().cpu().numpy().transpose((1,2,0))
    style_feature_map = style_feature_map.cpu().numpy().transpose((1,2,0))
    stylized_patches = patchify_tensor(stylized_feature_map)
    style_patches = patchify_tensor(style_feature_map)

    unique_style_patches = {HashNumpyWrapper(style_patches[i][j]) for j in range(style_patches.shape[1]) for i in range(style_patches.shape[0])}
    unique_max_patches = set()

    stylized_norms = np.sqrt(np.sum(stylized_patches*stylized_patches, axis = (2,3,4)))
    style_norms = np.sqrt(np.sum(style_patches*style_patches, axis = (2,3,4)))

    normalized_cross_correlation_measure = 0
    for i in range(stylized_patches.shape[0]):
        for j in range(stylized_patches.shape[1]):
            stylized_patch = stylized_patches[i][j]
            stylized_norm = stylized_norms[i][j]
            vals_numerator = np.einsum("ijhwc,hwc->ij",style_patches,stylized_patch)
            vals_denominator = stylized_norm*style_norms
            vals = np.divide(vals_numerator, vals_denominator)
            max_ind = np.argmax(vals)
            max_ind_2d = np.unravel_index(max_ind,vals.shape)
            max_val = vals[max_ind_2d[0]][max_ind_2d[1]]
            normalized_cross_correlation_measure += max_val
            unique_max_patches.add(HashNumpyWrapper(style_patches[max_ind_2d[0]][max_ind_2d[1]]))
    
    normalized_cross_correlation_measure /= (stylized_patches.shape[0]*stylized_patches.shape[1])
    layer_lp2 = len(unique_max_patches)/len(unique_style_patches)
    return normalized_cross_correlation_measure, layer_lp2

def local_patterns(stylized_image, style_image, config, device):
    neural_net, _, style_feature_maps_indices_names = prepare_model(config['model'], device)
    stylized_img_set_of_feature_maps = neural_net(stylized_image)
    style_img_set_of_feature_maps = neural_net(style_image)
    stylized_img_layer_of_feature_maps = [x for cnt, x in enumerate(stylized_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    style_img_layer_of_feature_maps = [x for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    temp_lp1, temp_lp2 = zip(*[get_normalized_cross_correlation_measure_and_layer_lp2(stylized_feature_map, style_feature_map) 
                for stylized_feature_map, style_feature_map in zip(stylized_img_layer_of_feature_maps, style_img_layer_of_feature_maps)])
    
    lp1 = sum(temp_lp1)/len(temp_lp1)
    lp2 = sum(temp_lp2)/len(temp_lp2)

    lp = (lp1+lp2)/2

    return lp