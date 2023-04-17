from copy import deepcopy
import time
from utils.evaluation_utils import content_fidelity, global_effects, local_patterns
import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config, styles_to_use):
    target_content_representation = target_representations[0]
    target_style1_representation = target_representations[1]
    target_style2_representation = target_representations[2]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style1_loss = 0.0
    style2_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    
    if "style1" in styles_to_use:
        for gram_gt, gram_hat in zip(target_style1_representation, current_style_representation):
            style1_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
        style1_loss /= len(target_style1_representation)

    if "style2" in styles_to_use:
        for gram_gt, gram_hat in zip(target_style2_representation, current_style_representation):
            style2_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
        style2_loss /= len(target_style2_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style1_weight'] * style1_loss + config['style2_weight'] * style2_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style1_loss, style2_loss, tv_loss

def neural_style_transfer(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style1_img_path = os.path.join(config['style_images_dir'], config['style1_img_name'])
    style2_img_path = os.path.join(config['style_images_dir'], config['style2_img_name'])

    out_dir_name = f'combined_{config["architecture"]}_{config["model"]}_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style1_img_path)[1].split('.')[0] + '_' + os.path.split(style2_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device being used is {device}")
    print(f"Architecture being used is: {config['architecture']}")

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style1_img = utils.prepare_img(style1_img_path, config['height'], device)
    style2_img = utils.prepare_img(style2_img_path, config['height'], device)

    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = deepcopy(content_img)
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style1_img_resized = utils.prepare_img(style1_img_path, np.asarray(content_img.shape[2:]), device)
        style2_img_resized = utils.prepare_img(style2_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = (style1_img_resized + style2_img_resized)/2

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style1_img_set_of_feature_maps = neural_net(style1_img)
    style2_img_set_of_feature_maps = neural_net(style2_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style1_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style1_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_style2_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style2_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style1_representation, target_style2_representation]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = 100

    if config['architecture']=="mo-net":
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
        model_(neural_net,optimizer, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config, ["style1","style2"], num_of_iterations, dump_path, None)

    elif config['architecture']=="cascade-net":
        #Cascade Layer 1
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
        model_(neural_net,optimizer, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config, ["style1"], num_of_iterations, dump_path, True)

        #Cascade Layer 2
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
        model_(neural_net,optimizer, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config, ["style2"], num_of_iterations, dump_path, False)
        
    elif config['architecture']=="cascade-net_parallel":
        pass

    start = time.time()
    cf = content_fidelity(optimizing_img, content_img, config, device)
    print(f"Content Fidelity={cf} calculated in {time.time() - start}s")
    
    start = time.time()
    ge_style1 = global_effects(optimizing_img, style1_img, config, device)
    ge_style2 = global_effects(optimizing_img, style2_img, config, device)
    print(f"Global Effects: style1={ge_style1}, style2={ge_style2} calculated in {time.time()-start}s")
    
    start = time.time()
    lp_style1 = local_patterns(optimizing_img, style1_img, config, device)
    lp_style2 = local_patterns(optimizing_img, style2_img, config, device)
    print(f"Local Patterns: style1={lp_style1}, style2={lp_style2} calculated in {time.time() - start}s")

    return dump_path

def model_(neural_net,optimizer, optimizing_img, target_representations, content_feature_maps_index_name_0, style_feature_maps_indices_names_0, config, style_names, num_of_iterations, dump_path, first):
    cnt = 0
    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style1_loss, style2_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name_0, style_feature_maps_indices_names_0, config, style_names)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            if cnt%5==0:
                if style1_loss ==0: 
                    style1_loss = torch.zeros_like(style2_loss)
                if style2_loss ==0: 
                    style2_loss = torch.zeros_like(style1_loss)

                print(f'LBFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style1 loss={config["style1_weight"] * style1_loss.item():12.4f}, style2 loss={config["style2_weight"] * style2_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations, should_display=False, first = first)
        
        cnt += 1
        return total_loss
    
    optimizer.step(closure)

if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img_name", type=str, help="content image name", default='golden_gate.jpg')
    parser.add_argument("--style1_img_name", type=str, help="style1 image name", default='flowers_crop.jpg')
    parser.add_argument("--style2_img_name", type=str, help="style2 image name", default='vg_starry_night_resized.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=155)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style1_weight", type=float, help="weight factor for style1 loss", default=1.5e4)
    parser.add_argument("--style2_weight", type=float, help="weight factor for style2 loss", default=1.5e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)
    parser.add_argument("--architecture", choices=["mo-net", "cascade-net"], type=str, help="architecture used for performing multi style transfer", default="cascade-net")

    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19','transformernet'], default='transformernet')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=3)
    args = parser.parse_args()

    # just wrapping settings into a dictionary
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    results_path = neural_style_transfer(optimization_config)

    # uncomment this if you want to create a video from images dumped during the optimization procedure
    # create_video_from_intermediate_results(results_path, img_format)
