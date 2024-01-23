# %%
# from dataset2d import Dataset2D
from dataset_creator import MyDataset
from model import UNet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import model_hyper_parameters as config
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses import DiceLoss, GeneralizedDiceLoss
from torchvision import transforms

# %%
class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.normalization = nn.Softmax(dim=1)

    def forward(self, inputs, targets, smooth=1):
        inputs = self.normalization(inputs)

        targets = targets[:, 1:2, ...]
        inputs = torch.where(inputs[:, 1:2, ...] > 0.5, 1.0, 0.0)

        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice

# %%
# make dataLoader
base_path = '/scratch/student/sinaziaee/datasets/2d_dataset/'
# base_path = 'new_3d_dataset'
train_dir = os.path.join(base_path, 'training')
valid_dir = os.path.join(base_path, 'validation')
test_dir = os.path.join(base_path, 'testing')

# transformer = transforms.Compose([
#     transforms.RandomCrop(512),
#     transforms.ToTensor(),
# ])

# train_dataset = Dataset2D(input_root=f'{train_dir}/images/', target_root=f'{train_dir}/labels/', transform= transformer)
# valid_dataset = Dataset2D(input_root=f'{valid_dir}/images/', target_root=f'{valid_dir}/labels/', transform= transformer)
# test_dataset = Dataset2D(input_root=f'{test_dir}/images/', target_root=f'{test_dir}/labels/', transform= transformer)

train_dataset = MyDataset(kind='train')
valid_dataset = MyDataset(kind='valid')
test_dataset = MyDataset(kind='test')

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=config.PIN_MEMORY)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=config.PIN_MEMORY)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=config.PIN_MEMORY)

# %%
train_data_shape = next(iter(train_loader))[0].shape
print(train_data_shape)

# %%
os.makedirs('final_result', exist_ok=True)
lr_rate = 0.0001
model = UNet(64, 5, use_xavier=True, use_batchNorm=True, dropout=0.5, retain_size=True, nbCls=2)
DEVICE = torch.device('cuda:1')
model.to(DEVICE)

history = {'train_loss': [], 'valid_loss': [], 'dice_valid_score': []}
optimizer = torch.optim.NAdam(model.parameters(), lr=lr_rate)
schedular = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.25, verbose=True)

dicelossfunc = GeneralizedDiceLoss(normalization='softmax')
diceScore = DiceScore()

num_train = int(len(train_loader) // config.BATCH_SIZE)
outer_loop = tqdm(range(30), leave=False, position=0)
inner_loop = tqdm(range(num_train), leave=False, position=0)

# %%
import sys

min_valid_loss = sys.maxsize

for i in outer_loop:
    model.train()
    
    total_loss = 0
    total_valid_loss = 0
    total_valid_dice = 0
    
    train_step = 0
    valid_step = 0

    data_iter = iter(train_loader)
    for j in inner_loop:
        (x, y) = next(data_iter)
        (x, y) = (x.to(DEVICE), y.to(DEVICE))

        pred = model(x)
        loss = dicelossfunc(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        train_step += 1
        # inner_pbar.set_postfix({'Train_loss': "{:.4f}".format(loss)})
        with torch.no_grad():
            model.eval()
            for (x, y) in valid_loader:
                (x, y) = (x.to(DEVICE), y.to(DEVICE))

                pred = model(x)
                valid_loss = dicelossfunc(pred.clone(), y.clone())
                total_valid_loss += valid_loss

                valid_score = diceScore(pred, y)

                total_valid_dice += valid_score
                valid_step += 1
                
    avg_loss = (total_loss / train_step).cpu().detach().numpy()
    avg_valid_loss = (total_valid_loss / valid_step).cpu().detach().numpy()
    avg_valid_dice = (total_valid_dice / valid_step).cpu().detach().numpy()

    schedular.step(avg_valid_loss)

    history['train_loss'].append(avg_loss)
    history['valid_loss'].append(avg_valid_loss)
    history['dice_valid_score'].append(avg_valid_dice)
        
    # torch.save(model.state_dict(), f'./final_result/unet_{i + 1}.pt')

    
    if min_valid_loss > avg_valid_loss:
        min_valid_loss = avg_valid_dice
        torch.save(model.state_dict(), './final_result/best_model.pt')
        
        with open(f'./final_result/history_{i + 1}.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        print(history)
    # print('Saving model...\n\n')
    # torch.save(model.state_dict(), './final_result/UNet.pt')

print('Saving figure...\n\n')
plt.style.use('ggplot')
plt.figure(figsize=(15, 10))
plt.plot(history['train_loss'], label='Train_Dice_Loss')
plt.plot(history['valid_loss'], label='Validation_Dice_Loss')
plt.title('Training Dice Score on Dataset')
plt.xlabel('Number of Epoch')
plt.ylabel('Dice Loss')
plt.legend(loc='lower left')
plt.savefig('./final_result/train_result.png')

print('Saving History...\n\n')
with open('./final_result/history.pkl', 'wb') as f:
    pickle.dump(history, f)

print('***************End of System***************')

# %%



