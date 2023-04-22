import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from os.path import join
from PIL import Image
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def eval_vae(config,matrix,vgg,dec):

    matrix.eval()
    vgg.eval()
    dec.eval()
    cont_file = os.path.join(config['content_images_dir'], config['content_img_name'])
    ref_file = os.path.join(config['style_images_dir'], config['style1_img_name'])

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

    prediction = prediction * 255.0
    prediction = prediction.clamp(0, 255)

    Image.fromarray(np.uint8(prediction)).save("/home/swarnita/CSCI566-Project/data/output-images/vae_output.jpg")
    return Image.fromarray(np.uint8(prediction))

transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)