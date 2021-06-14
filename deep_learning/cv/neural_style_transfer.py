'''
Reference: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
'''

from torchvision import models
from torchvision import transforms
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from PIL import Image
import copy
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()
content_layer_default = ['conv_4']
style_layer_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def gram_matrix(x):
    b, c, h, w = x.size()
    # serialize h and w
    features = x.view(b * c, h * w)
    # G = FF'
    g = torch.mm(features, features.T)
    # normalize by diving the size
    return g.div(b * c * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # detach the target tensor so that 
        # its gradient is no longer being tracked
        self.target = target.detach()
    
    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    
    def forward(self, x):
        g = gram_matrix(x)
        self.loss = F.mse_loss(g, self.target)
        return x


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)
    
    def forward(self, x):
        return (x - self.mean) / self.std


def get_style_model_and_losses(cnn, mean, std, style_img, content_img, content_layers, style_layers):
    cnn = copy.deepcopy(cnn)

    norm = Normalization(mean, std)

    content_losses = []
    style_losses = []

    # sequentially add in modules
    model = nn.Sequential(norm)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            # off inplace so that ContentLoss can work
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'layer {layer} not recognized.')
        
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)
    
    # do a bit of trimming
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]

    return model, style_losses, content_losses


def get_input_optimizer(img):
    # specify optimizer to only optimize for the img parameters
    optimizer = optim.LBFGS([img.requires_grad_()])
    return optimizer


def run(cnn, mean, std, content_img, style_img, input_img,
        num_steps=300, style_weight=1, content_weight=1,
            content_layers=content_layer_default, style_layers=style_layer_default):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, mean, std, style_img, content_img, content_layers, style_layers
    )
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct value of input image after update
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            style_loss *= style_weight
            content_loss *= content_weight

            loss = style_loss + content_loss
            loss.backward()
            
            run[0] += 1
            if run[0] % 100 == 0:
                print(f'run {run}')
                print(f'style loss: {style_score.item()}')
                print(f'content loss: {content_score.item()}')
                print()
            
            return style_score + content_score
        
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', required=True, help='content image file path')
    parser.add_argument('--style_path', required=True, help='style image file path')
    parser.add_argument('--save_path', default='./output.png', help='ouput image file path')
    args = parser.parse_args()


    style_img = load_image(args.style_path)
    content_img = load_image(args.content_path)
    # input is set to a copy of content, to be updated
    input_img = content_img.clone()

    if style_img.size() != content_img.size():
        raise ValueError('style image has different size from content image')

    # use vgg19 as the backbone
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # use vgg image mean and std
    cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    output = run(
        cnn,
        cnn_mean,
        cnn_std,
        content_img,
        style_img,
        input_img
    )

    # save to local
    image = output.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(args.save_path)
