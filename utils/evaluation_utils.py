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

def patchify(x, patch_dim = 3):
    # Expecting x to be 3d in the format C x H x W
    patch_size = (patch_dim,patch_dim)
    stride = 1
    patches = x.unfold(1, patch_size[0], stride).unfold(2, patch_size[1], stride)
    patches = patches.permute(1,2,0,3,4)
    patches = patches.reshape(patches.shape[0]*patches.shape[1],patches.shape[2]*patches.shape[3]*patches.shape[4])
    return patches

def local_patterns(stylized_image, style_image, config, device):
    neural_net, _, style_feature_maps_indices_names = prepare_model(config['model'], device)
    
    stylized_img_set_of_feature_maps = neural_net(stylized_image)
    style_img_set_of_feature_maps = neural_net(style_image)
    
    stylized_img_relevant_feature_maps = [x.squeeze(0) for cnt, x in enumerate(stylized_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    style_img_relevant_feature_maps = [x.squeeze(0) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    
    stylized_img_patches_all_layers = [patchify(x,3) for x in stylized_img_relevant_feature_maps]
    style_img_patches_all_layers = [patchify(x,3) for x in style_img_relevant_feature_maps]

    n_layers = len(stylized_img_patches_all_layers)
    lp1,lp2 = 0,0

    for i in range(n_layers):
        stylized_img_patches = stylized_img_patches_all_layers[i]
        style_img_patches = style_img_patches_all_layers[i]
        
        stylized_img_patches_norm = torch.sqrt(torch.sum(stylized_img_patches**2,dim=1,keepdim=True))
        style_img_patches_norm = torch.sqrt(torch.sum(style_img_patches**2,dim=1,keepdim=True))

        stylized_img_patches_normalized = stylized_img_patches / stylized_img_patches_norm
        style_img_patches_normalized = style_img_patches / style_img_patches_norm

        similarities = torch.matmul(stylized_img_patches_normalized,style_img_patches_normalized.T)
        max_indices = torch.argmax(similarities, dim=-1)
        max_indices = max_indices.reshape(-1)

        style_patches_of_interest = style_img_patches[max_indices]
        style_patches_of_interest_norm = torch.sqrt(torch.sum(style_patches_of_interest**2,dim=1,keepdim=True))
        style_patches_of_interest_normalized = style_patches_of_interest / style_patches_of_interest_norm

        style_similarities = torch.matmul(style_img_patches_normalized,style_img_patches_normalized.T)
        style_max_indices = torch.argmax(style_similarities, dim=-1)
        style_max_indices = style_max_indices.reshape(-1)

        num,den = len(set(max_indices.tolist())), len(set(style_max_indices.tolist()))

        all_similarities = torch.matmul(stylized_img_patches_normalized, style_patches_of_interest_normalized.T).diagonal()
        lp1 += (torch.sum(all_similarities).item())/len(all_similarities)
        lp2 += num/den

    lp = (lp1+lp2)/(2*n_layers)
    return lp