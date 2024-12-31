import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16
from tqdm import tqdm
import cv2
import os

vgg = vgg16(weights='DEFAULT')

class Utils: # replaces import module
    def __init__(self):
        pass

    def gram_matrix(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, w*h)
        xt = x.transpose(1, 2) # B x F x C
        gram = torch.bmm(x, xt) / c*w*h
        return gram

    def total_variation(self, x):
        mean = x.mean()
        mean = torch.ones_like(x)*mean
        var = (mean - x)**2
        return var.sum()

    def read_image(self, path, height=None):
        if not os.path.exists(path):
            raise Exception(f'File not found: {path}')
        img = cv2.imread(path)
        if height is not None:
            w, h = img.shape[:2]
            width = int(w * height // h)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img /= 255.0

        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
        ])
        img = transform(img).unsqueeze(0)
        return img

    def show_image(self, im, path):
        im = im.detach().cpu().numpy()
        im = np.transpose(im, (1, 2, 0))
        plt.imshow(im)
        plt.savefig(path)

utils = Utils()

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.block1 = nn.ModuleList(vgg.features[:4]) + [avg_pool]
        self.block2 = nn.ModuleList(vgg.features[5:9]) + [avg_pool]
        self.block3 = nn.ModuleList(vgg.features[10:16]) + [avg_pool]
        self.block4 = nn.ModuleList(vgg.features[17:23]) + [avg_pool]
        self.block5 = nn.ModuleList(vgg.features[24:30]) + [avg_pool]

        self.block1 = nn.Sequential(*self.block1)
        self.block2 = nn.Sequential(*self.block2)
        self.block3 = nn.Sequential(*self.block3)
        self.block4 = nn.Sequential(*self.block4)
        self.block5 = nn.Sequential(*self.block5)

    def forward(self, x):
        emb1 = self.block1(x)
        emb2 = self.block2(emb1)
        emb3 = self.block3(emb2)
        emb4 = self.block4(emb3)
        emb5 = self.block5(emb4)
        return emb1, emb2, emb3, emb4, emb5
        
def loss_fn(model, input_img, content_feats, style_feats, content_idx, style_idxs, config):
    input_representation = model(input)
    input_content = input_representation[content_idx]
    input_style_representation = [utils.gram_matrix(v) for i, v in enumerate(input_representation) if i in style_idxs]

    content_loss = torch.nn.MSELoss(reduction='mean')(input_content, content_feats[content_idx])

    style_loss = 0.0
    for target, current in zip(style_feats[style_idxs], input_style_representation):
        style_loss += torch.nn.MSELoss(reduction='mean')(current, target)
    style_loss /= len(style_idxs)

    tv_loss = utils.total_variation(input_img)

    tot_loss = config['content_weight']*content_loss + config['style_weight']*style_loss + config['tv_weight']*tv_loss
    return tot_loss, content_loss, style_loss, tv_loss

def tuning_step(model, optimizer, input_img, content_feats, style_feats, config):
    content_idx = config['content_feature_index']
    style_idxs = config['style_features_indices']
    optimizer.zero_grad()
    tot_loss, content_loss, style_loss, tv_loss = loss_fn(model, input_img, content_feats, style_feats, content_idx, style_idxs, config)
    tot_loss.backward()
    optimizer.step()
    return tot_loss, content_loss, style_loss, tv_loss

def neural_style_transfer(config):
    assert config['optimizer'] in ['adam', 'lbfgs']
    assert config['input_type'] in ['random', 'content']
    assert config['content_feature_index'] in range(5)
    assert all(idx in range(5) for idx in config['style_features_indices'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img_path = config['content_img']
    style_img_path = config['style_img']
    content = utils.read_image(content_img_path, config['height']).to(device)
    style = utils.read_image(style_img_path, config['height']).to(device)

    #gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
    #init_img = torch.from_numpy(gaussian_noise_img).float().to(device)

    if config['input_type'] == 'random':
        input = torch.randn_like(content, requires_grad=True, device=device) # optimizer will automatically update the input image
    elif config['input_type'] == 'content':
        input = content.clone().requires_grad_(True)
    
    # safe checking
    utils.show_image(input.squeeze(0), 'initial_input.png')
    utils.show_image(content.squeeze(0), 'nst_content.png')
    utils.show_image(style.squeeze(0), 'nst_style.png')

    model = Model().to(device)
    for p in model.parameters():
        p.requires_grad = False

    content_feature_maps = model(content)
    style_feature_maps = model(style)

    if config['optimizer'] == 'adam':
        input_history = []
        optimizer = torch.optim.Adam((input,), lr=config['lr'])
        for step in tqdm(range(config['num_steps'])):
            tot_loss, content_loss, style_loss, tv_loss = tuning_step(model, optimizer, input, content_feature_maps, style_feature_maps, config)
            if step % 300 == 0:
                input_history.append(input.detach().cpu())
                print(f'step: {step}, tot_loss: {tot_loss.item()}, content_loss: {content_loss.item()}, style_loss: {style_loss.item()}, tv_loss: {tv_loss.item()}')
        
        return input_history

    if config['optimizer'] == 'lbfgs':
        optimizer = torch.optim.LBFGS((input,), max_iter=1000)
        tot_loss, content_loss, style_loss, tv_loss = tuning_step(model, optimizer, input, content_feature_maps, style_feature_maps, config)
        return input.detach().cpu()

if __name__ == '__main__':
    config = {
        'lr': 1e1,
        'num_steps': 3000,
        'height': 400,
        'content_feature_index': 4,
        'style_features_indices': [0, 1, 2, 3],
        'content_weight': 1e5,
        'style_weight': 3e4,
        'tv_weight': 1,
        'optimizer': 'adam',
        'content_img': './lion.jpg',
        'style_img': './vg_starry_night.jpg',
        'input_type': 'content'
    }

    result = neural_style_transfer(config)

    if isinstance(result, list):
        for i, r in enumerate(result):
            torch.save(r, f'generated_input_{i}.pt')
    else:
        torch.save(result, 'generated_input.pt')
