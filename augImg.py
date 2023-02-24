from torchvision.transforms import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomResizedCrop(224,(0.8,1.0)),
                                      #transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      #transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11)),
                                      #transforms.RandomRotation(degrees=(60, 90)),
                                      transforms.RandomInvert(p=0.5),
                                      transforms.GaussianBlur(kernel_size=9),
                                      transforms.RandomApply(
                                        [transforms.ColorJitter(
                                          0.8*0.5, 
                                          0.8*0.5, 
                                          0.8*0.5, 
                                          0.2*0.5)], p = 0.8),
                                        transforms.RandomGrayscale(p=0.2)])

path = '/home/atik/Documents/Meta Augmentation/MAML/img/'

img0 = transform(os.path.join(path,'Picture1.png'))
img1 = transform(os.path.join(path,'Picture1.png'))

plt.imshow(img0.permute(1, 2, 0))
plt.axis('off')
plt.savefig(path+"test00.png", bbox_inches='tight', pad_inches = 0)

plt.imshow(img1.permute(1, 2, 0))
plt.axis('off')
plt.savefig(path+"test01.png", bbox_inches='tight', pad_inches = 0)

