import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse

#for vae
import argparse

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
from PIL import Image, ImageOps
import os
from libs.models import encoder4
from libs.models import decoder4
from libs.Matrix import MulLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval(config):

    matrix.eval()
    vgg.eval()
    dec.eval()
    cont_file = os.path.join(config['content_images_dir'], config['content_img_name'])
    ref_file = os.path.join(config['style_images_dir'], config['style1_img_name'])
    #print(content_path)
    #content_path = os.path.join(opt.image_dataset, 'content')
    #output_path = os.path.join(opt.image_dataset, 'result')
    #ref_path = os.path.join(opt.image_dataset, 'style')

    #for cont_file in os.listdir(content_path):
    #    for ref_file in os.listdir(ref_path):
    t0 = time.time()
    content = Image.open(cont_file).convert('RGB')
    ref = Image.open(ref_file).convert('RGB')

    content = transform(content).unsqueeze(0).to(device)
    ref = transform(ref).unsqueeze(0).to(device)

    with torch.no_grad():
        sF = vgg(ref)
        cF = vgg(content)
        feature, _, _ = matrix(cF['r41'], sF['r41'])
        prediction = dec(feature)

        prediction = prediction.data[0].cpu().permute(1, 2, 0)

    t1 = time.time()
    #print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))

    prediction = prediction * 255.0
    prediction = prediction.clamp(0, 255)

    #file_name = cont_file.split('.')[0] + '_' + ref_file.split('.')[0] + '.jpg'
    #save_name = os.path.join(output_path, file_name)
    #Image.fromarray(np.uint8(prediction)).save(save_name)
    return Image.fromarray(np.uint8(prediction))



transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)






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

    out_dir_name = f'combined_{config["architecture"]}_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style1_img_path)[1].split('.')[0] + '_' + os.path.split(style2_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        init_img = content_img
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
    num_of_iterations = 1000

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

    elif config['architecture']=="cascade-net_vae":
        pass
        #cascade layer 1 has to be vae, output image has to go to gram matrix
        #Cascade Layer 1
        ##Eval Start!!!!
        optimizing_img = eval(config)
        
        #Cascade Layer 2
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
        model_(neural_net,optimizer, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config, ["style2"], num_of_iterations, dump_path, False)

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
    parser.add_argument("--content_img_name", type=str, help="content image name", default='figures.jpg')
    parser.add_argument("--style1_img_name", type=str, help="style1 image name", default='giger_crop.jpg')
    parser.add_argument("--style2_img_name", type=str, help="style2 image name", default='mosaic.jpg')
    parser.add_argument("--height", type=int, help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style1_weight", type=float, help="weight factor for style1 loss", default=1.5e4)
    parser.add_argument("--style2_weight", type=float, help="weight factor for style2 loss", default=1.5e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)
    parser.add_argument("--architecture", choices=["mo-net", "cascade-net","cascade-net_vae"], type=str, help="architecture used for performing multi style transfer", default="cascade-net")

    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=3)
    

    # Training settings
    #parser = argparse.ArgumentParser(description='LT-VAE Style transfer')
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
    parser.add_argument('--image_dataset', type=str, default='Test')
    parser.add_argument("--latent", type=int, default=256, help='length of latent vector')
    parser.add_argument("--vgg_dir", default='models/definitions/vgg_r41.pth', help='pre-trained encoder path')
    parser.add_argument("--decoder_dir", default='models/definitions/dec_r41.pth', help='pre-trained decoder path')
    parser.add_argument("--matrixPath", default='models/definitions/matrix_r41_new.pth', help='pre-trained model path')

    args = parser.parse_args()
    vgg = encoder4()
    dec = decoder4()
    matrix = MulLayer(z_dim=args.latent)
    vgg.load_state_dict(torch.load(args.vgg_dir,map_location=torch.device('cpu')))
    dec.load_state_dict(torch.load(args.decoder_dir,map_location=torch.device('cpu')))
    matrix.load_state_dict(torch.load(args.matrixPath,map_location=torch.device('cpu')))

    vgg.to(device)
    dec.to(device)
    matrix.to(device)
    print('===> Loading datasets')


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

