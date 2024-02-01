# %%
from model import UNet
from dataset_creator import MyDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from imageio import imwrite
from PIL import Image
import model_hyper_parameters as config
from tqdm.auto import tqdm

# %%
class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.normalization=nn.Softmax(dim=1)

    def forward(self, inputs, targets, smooth=1e-4):
        inputs = self.normalization(inputs)

        targets = targets[:, 1:2, ...]
        inputs = torch.where(inputs[:, 1:2, ...] > 0.5, 1.0, 0.0)

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice

# %%
DEFAULT_KIDNEY_COLOR = [255, 0, 0]
DEFAULT_PRED_COLOR = [0, 0, 255]
ALPHA = 0.3
# DEVICE = torch.device('cuda:0')
DEVICE = torch.device('cpu')
dicescore=DiceScore()
os.makedirs('pred_img',exist_ok=True)

# %%
model = UNet(64,5,use_xavier=True,use_batchNorm=True,dropout=0.5,retain_size=True,nbCls=2)
model.to(DEVICE)
if torch.cuda.is_available():
    print('CUDA Available!')
    model.load_state_dict(torch.load('final_result/best_model.pt'))
else:
    print('CUDA is unavailable, using CPU instead!')
    print('Warning: using CPU might require several hours')
    model.load_state_dict(torch.load('final_result/best_model.pt', map_location=torch.device('cpu')))

# %%
# make dataLoader
base_path = '/scratch/student/sinaziaee/datasets/2d_dataset/'
test_dir = os.path.join(base_path, 'testing')

test_dataset = MyDataset(kind='test')

# BATCH_SIZE = 32
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=config.PIN_MEMORY)

# %%
num_data = len(test_dataset)
total_dice=0
softmax=nn.Softmax(dim=1)

for inx in tqdm(range(num_data)):
    model.eval()
    img_np, seg_np = test_dataset[inx]
    
    # Move input tensor to the GPU using clone().detach()
    img_tensor = torch.tensor(img_np.reshape((1, 1, 512, 512)), dtype=torch.float32).clone().detach().to(DEVICE)  # Replace 'DEVICE' with your actual device
    
    pred = model(img_tensor)
    pred = pred.cpu()
    
    # Use clone().detach() for creating a new tensor
    dice = dicescore(pred.clone().detach(), torch.tensor(seg_np.reshape((1, 2, 512, 512)), dtype=torch.float32).clone().detach())

    total_dice+=dice
    
    pred=softmax(pred)
    pred=np.where(pred[:,1,...].cpu().detach().numpy()>0.5,1,0)
    
    img_np=img_np.reshape((1,512,512))
    seg_np=seg_np[1,...]
    seg_np=seg_np.reshape((1,512,512))
    
    img=255*img_np
    img=np.stack((img,img,img),axis=-1)
    
    shp=seg_np.shape
    
    seg_color_np = pred.astype(np.float32)
    
    
    seg_color=np.zeros((shp[0],shp[1],shp[2],3),dtype=np.float32)
    seg_color[np.equal(seg_np,1)]=DEFAULT_KIDNEY_COLOR
    seg_color[np.equal(pred,1)]=DEFAULT_PRED_COLOR
    
    # img.astype(np.uint8)
    # seg_color.astype(np.uint8)
    # seg_np.astype(np.uint8)
    print(type(seg_np))
    
    segbin1=np.greater(seg_np,0)
    segbin2=np.greater(pred,0)
    
    segbin=segbin1*0.5+segbin2*0.5
    
    r_segbin=np.stack((segbin,segbin,segbin),axis=-1)
    overlayed=np.where(
        r_segbin,
        np.round(ALPHA*seg_color+(1-ALPHA)*img).astype(np.uint8),
        np.round(img).astype(np.uint8)
        
    )
    imwrite('./pred_img/'+'{:04d}_{:.2f}%.png'.format(inx,dice*100),overlayed[0])
print('Image Generated Finished, Average F1 Score: {:.3f}%'.format((total_dice/num_data)*100))

# %%



