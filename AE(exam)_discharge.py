import pickle
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch import nn
from torch.nn import functional as F
# from models import EntropyLossEncap
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from AE_1DCNN import AutoEncoderConv1d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
# import lightgbm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pathlib import Path
from sklearn.model_selection import train_test_split
### CUDAAutoEncoderMem
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)

### data load
data_dir = Path('./data')

with open(data_dir/'model1/input/normal_x.pickle', 'rb') as f:
    x_data = pickle.load(f)
# plt.plot(x_data)
data_scaler = RobustScaler()

### 훈련 데이터
train_data_raw, test_data_raw = train_test_split(x_data, shuffle=True)
train_data = data_scaler.fit_transform(train_data_raw)
train_data_tensor = torch.Tensor(list(train_data)).reshape(train_data.shape[0], 1, -1)
print(train_data_tensor.shape)

### 테스트 데이터

test_data = data_scaler.transform(test_data_raw)
test_data_tensor = torch.Tensor(list(test_data)).reshape(test_data.shape[0], 1, -1)
print(test_data_tensor.shape)

### 등급 데이터
with open(data_dir/'model1/input/abnormal_x.pickle', 'rb') as f:
    r_data_raw = pickle.load(f)
r_data = data_scaler.transform(r_data_raw)

grade_data_tensor = torch.Tensor(list(r_data)).reshape(r_data.shape[0], 1, -1)
print(grade_data_tensor.shape)

# plt.plot(test_data_raw.T.values, 'k')
# plt.plot(r_data_raw.T.values, 'r')

### 2. Modeling
### @ Hyperparamter
max_epoch_num = 10000
learning_rate = 0.0025
entropy_loss_weight = 0.0002

# channel = [1, 4, 8, 10]
# kernel = [10, 5, 1]

channel = [1, 4, 8]
kernel = [5, 1]

### 모델 정의
model = AutoEncoderConv1d(channel, kernel).to(DEVICE)

### 그외함수들
tr_recon_loss_func = nn.MSELoss().to(DEVICE)
#tr_entropy_loss_func = EntropyLossEncap().to(DEVICE)
tr_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data_tensor,
    batch_size  = 256,
    shuffle     = False
)

test_loader = torch.utils.data.DataLoader(
    dataset     = test_data_tensor,
    batch_size  = 126,
    shuffle     = False
)

grade_loader = torch.utils.data.DataLoader(
    dataset     = grade_data_tensor,
    batch_size  = 126,
    shuffle     = False
)

### Training
loss_history = []
for epoch_idx in tqdm(range(0,max_epoch_num)):
    for batch_idx,value in enumerate(train_loader):
        recon_res=model(value.to(DEVICE))
        recon_frames=recon_res['output']
        loss = tr_recon_loss_func(recon_frames,value.to(DEVICE))
        recon_loss_val = loss.item()
        loss_val = loss.item()
        tr_optimizer.zero_grad()
        loss.backward()
        tr_optimizer.step()
    loss_history.append(loss.detach().cpu().numpy())

plt.figure(figsize = (10,4))
plt.plot(loss_history)

torch.save(model, data_dir/'model1/autoencoder.pkl')
# model = torch.load('autoencoder_discharge.pkl')

### 3. threshold in traindata
train_tmp=[]
model.eval()
with torch.no_grad():
    for idx, value in enumerate(train_loader):
        output = model(value.to(DEVICE))['output']
        hidden = model(value.to(DEVICE))['hidden vector']

        for i in range(output.shape[0]):
            torch_recon_value = torch.squeeze(output[i]).detach().cpu().numpy()
            real_value = torch.squeeze(value[i]).detach().cpu().numpy()
            recon_err = (torch_recon_value-real_value)**2
            train_tmp.append([idx, i, recon_err,
                              recon_err[10:-10].mean(),
                              recon_err[10:-10].max(),
                              real_value,
                              torch_recon_value])

            hidden_vector = hidden[i].shape

train_err_df = pd.DataFrame(train_tmp, columns=['idx','no','error','mean_err','max_err','real','recon'])

### 4. Test
### Testing

test_tmp=[]
model.eval()
with torch.no_grad():
    for idx, value in enumerate(test_loader):
        output = model(value.to(DEVICE))['output']
        for i in range(output.shape[0]):
            torch_recon_value = torch.squeeze(output[i]).detach().cpu().numpy()
            real_value = torch.squeeze(value[i]).detach().cpu().numpy()
            recon_err = (torch_recon_value-real_value)**2
            test_tmp.append([idx, i, recon_err,
                              recon_err[10:-10].mean(),
                              recon_err[10:-10].max(),
                              real_value,
                              torch_recon_value])

test_err_df = pd.DataFrame(test_tmp, columns=['idx','no','error','mean_err','max_err','real','recon'])

### 5. grading
### Testing
grade_tmp=[]
model.eval()
with torch.no_grad():
    for idx, value in enumerate(grade_loader):
        output = model(value.to(DEVICE))['output']
        for i in range(output.shape[0]):
            torch_recon_value = torch.squeeze(output[i]).detach().cpu().numpy()
            real_value = torch.squeeze(value[i]).detach().cpu().numpy()
            recon_err = (torch_recon_value-real_value)**2
            grade_tmp.append([idx, i, recon_err,
                              recon_err[10:-10].mean(),
                              recon_err[10:-10].max(),
                              real_value,
                              torch_recon_value])

grade_err_df = pd.DataFrame(grade_tmp, columns=['idx','no','error','mean_err','max_err','real','recon'])

plt.figure(figsize = (10,4))
plt.scatter(range(len(train_err_df)), train_err_df['mean_err'])
plt.scatter(range(len(train_err_df), len(train_err_df) + len(test_err_df)), test_err_df['mean_err'])
plt.scatter(range(len(train_err_df) + len(test_err_df), len(train_err_df) +len(test_err_df) + len(grade_err_df)),grade_err_df['mean_err'])

Q1 = train_err_df['mean_err'].quantile(0.25)
Q3 = train_err_df['mean_err'].quantile(0.75)
threshold = Q3 + 3*(Q3 - Q1)
plt.axhline(threshold, c='r')
a = grade_err_df[grade_err_df['mean_err'] <threshold]
r_data_raw.iloc[[58, 69, 122, 123, 130, 135, 147]]

plt.figure(figsize = (10,4))
plt.plot(data_scaler.inverse_transform(train_err_df['real'][0].reshape(1,-1))[0])
plt.plot(data_scaler.inverse_transform(train_err_df['recon'][0].reshape(1,-1))[0])

plt.figure(figsize = (10,4))
plt.plot(data_scaler.inverse_transform(test_err_df['real'][0].reshape(1,-1))[0])
plt.plot(data_scaler.inverse_transform(test_err_df['recon'][0].reshape(1,-1))[0])

plt.figure(figsize = (10,4))
plt.plot(data_scaler.inverse_transform(grade_err_df['real'][i].reshape(1,-1))[0])
plt.plot(data_scaler.inverse_transform(grade_err_df['recon'][i].reshape(1,-1))[0])

with open('data/fcs/fcs_data.pickle', 'rb') as f:
    fcs = pickle.load(f)

a = fcs[['Cell ID','Maximum Temperature_Charge #01']]