import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from generator import Generator
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


noise_dim = 100
d_filter_depth_in = 3
batch_size = 100

# this converts any pytorch tensor,
# an n-dimensional array, to a 
# variable and puts it on a gpu if
# a one is available
def to_variable(x):
    '''
    convert a tensor to a variable
    with gradient tracking
    '''
    if torch.cuda.is_available():
        x = x .cuda()
    return Variable(x)


# we're going normalize our images
# to make training the generator easier
# this de-normalizes the images coming out
# of the generator so they look intelligble
def denorm_monsters(x):
    renorm = (x*0.5)+0.5
    return renorm.clamp(0,1)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Generate noise on MPS
noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

# Load model to MPS
load_model = os.path.join('generator_ep_%d' % 100)
generator_final = Generator()
generator_final.load_state_dict(torch.load(load_model, map_location=device))
generator_final.to(device)
generator_final.eval()

# Generate and move to CPU for visualization
with torch.no_grad():
    generated_imgs = generator_final(noise).cpu()


# Denormalize
generated_imgs = denorm_monsters(generated_imgs)


os.makedirs("generated_cards", exist_ok=True)

# Convert each image tensor to a high-res PIL image and save
for idx, img_tensor in enumerate(generated_imgs):
    img_pil = TF.to_pil_image(img_tensor)


    # Optional: draw border or card frame (basic example)
    #draw = ImageDraw.Draw(img_pil)
    #draw.rectangle([0, 0, 420, 613], outline="black", width=5)

    # Save card
    img_pil.save(f"generated_cards/yugioh_card_{idx+1}.png")




