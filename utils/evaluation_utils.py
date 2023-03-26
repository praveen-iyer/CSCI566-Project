from utils.utils import prepare_model
from utils.utils import gram_matrix
import numpy as np
import cv2

def content_fidelity():
    pass

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
    #torch.dot?
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
    c,h,w = t.shape
    patches = []
    for i in range(h-2):
        for j in range(w-2):
            patches.append(t[:,i:i+3,j:j+3])
    return patches

def get_normalized_cross_correlation_measure(stylized_feature_map, style_feature_map):
    stylized_patches = patchify_tensor(stylized_feature_map)
    style_patches = patchify_tensor(style_feature_map)

    normalized_cross_correlation_measure = 0
    for stylized_patch in stylized_patches:
        max_val = float('-inf')
        for style_patch in style_patches:
            normalized_cross_correlation_measure += np.sum(np.multiply(stylized_patch.detach().cpu().numpy(), style_patch.cpu().numpy()))/(np.linalg.norm(stylized_patch.detach().cpu().numpy())*np.linalg.norm(style_patch.cpu().numpy()))
            max_val = max(max_val, normalized_cross_correlation_measure)
        normalized_cross_correlation_measure += max_val

    normalized_cross_correlation_measure /= len(stylized_patches)
    return normalized_cross_correlation_measure

def calc_layer_lp2(stylized_feature_map, style_feature_map):
    stylized_patches = patchify_tensor(stylized_feature_map)
    style_patches = patchify_tensor(style_feature_map)

    unique_style_patches = {tupler(style_patch.tolist()) for style_patch in style_patches}
    unique_max_patches = set()

    for stylized_patch in stylized_patches:
        max_val = float('-inf')
        for style_patch in style_patches:
            normalized_cross_correlation_measure += np.sum(np.multiply(stylized_patch.detach().cpu().numpy(), style_patch.cpu().numpy()))/(np.linalg.norm(stylized_patch.cpu().numpy())*np.linalg.norm(style_patch.cpu().numpy()))
            if max_val<normalized_cross_correlation_measure:
                max_val = normalized_cross_correlation_measure
                max_patch = style_patch
        unique_max_patches.add(tupler(max_patch).tolist())
        normalized_cross_correlation_measure += max_val

    return len(unique_max_patches)/len(unique_style_patches)

def local_patterns(stylized_image, style_image, config, device):
    neural_net, _, style_feature_maps_indices_names = prepare_model(config['model'], device)
    stylized_img_set_of_feature_maps = neural_net(stylized_image)
    style_img_set_of_feature_maps = neural_net(style_image)
    stylized_img_layer_of_feature_maps = [x for cnt, x in enumerate(stylized_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    style_img_layer_of_feature_maps = [x for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    temp_lp1 = [get_normalized_cross_correlation_measure(stylized_feature_map, style_feature_map) 
                for stylized_feature_map, style_feature_map in zip(stylized_img_layer_of_feature_maps, style_img_layer_of_feature_maps)]
    lp1 = sum(temp_lp1)/len(temp_lp1)

    temp_lp2 = [calc_layer_lp2(stylized_feature_map, style_feature_map) 
               for stylized_feature_map, style_feature_map in zip(stylized_img_layer_of_feature_maps, style_img_layer_of_feature_maps)]
    lp2 = sum(temp_lp2)/len(temp_lp2)

    lp = (lp1+lp2)/2

    return lp