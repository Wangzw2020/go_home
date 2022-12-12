# ---------import packages -----------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ------------ hyperparameters --------------

Number = 1000    # length of original time series
epoch = 10
m = 0.7   # training set proportion
time_step = 10  # sequence length
input_size = 1  # dim of input feature
hidden_size = 100    # dim of hidden feature
output_size = 1  # dim of output
num_layers = 2  # how many LSTM will be stacked
learning_rate = 1e-2
L = 1   # how many steps we predict ahead
batch_size = 16
# ------------- generate dataset ---------------
t = np.linspace(0, 20, 1000)
dataset = np.sin(t*5)
dataset = np.reshape(dataset, (-1, 1))
plt.plot(t, dataset)
plt.show()
# ------------ data preprocessing ----------------
# ----------- split dataset into training set and test set ----------------
train_len = int(len(dataset) * m)   # 700
train_data = dataset[:train_len]    # (700, 1)
test_data = dataset[train_len-time_step-L+1:]   # (310, 1)
# since the first element of test data is the first prediction, we need previous time_step values to predict.
# so we need to look back give time_step to construct the test data pairs

# -----------standardization data by using (train) samples -------------
sc = StandardScaler().fit(train_data)
train_data = sc.transform(train_data)
test_data = sc.transform(test_data)

# ------------- package time series into ML-learnable data-------------
def create_dataset(dataset, look_back, look_ahead):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back + L - 1])
    return np.array(dataX), np.array(dataY)

train_X, train_Y = create_dataset(train_data, look_back=time_step, look_ahead=L)    # (690, 10, 1)

print(train_X.shape,train_Y.shape)

test_X, test_Y = create_dataset(test_data, look_back=time_step, look_ahead=L)    # (300, 10, 1)
train_size = len(train_X)
test_size = len(test_X)

# ------------ Put data and labels together via TensorDataset -------------
train_X = torch.Tensor(train_X)
train_Y = torch.Tensor(train_Y)
train_dataset = TensorDataset(train_X, train_Y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# for batch_idx, (inp, label) in enumerate(train_loader):
#   print(f'the shape of train data is: {inp.size()}, label is {label.size()}')
# make the input data shape now is (batch, seq, feature)

# ------------------ test --------------------------------
test_X = torch.Tensor(test_X)
test_Y = torch.Tensor(test_Y)
test_dataset = TensorDataset(test_X, test_Y)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ------------ LSTM model ------------------
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(lstm_reg, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)    # LSTM structure
        self.reg = nn.Linear(hidden_size, output_size)   # linear layer for regression

    def forward(self, x):
        lstm_out, self.hidden = self.rnn(x)     # (seq, batch, hidden)
        # s, b, h = lstm_out.shape
        # shape_lstm_out = (s, b, h)
        dense_out = self.reg(lstm_out[-1]) # only use the last hidden step h_(time_step) to do regression.

        return dense_out

# ----------------  -----------------------------
model = lstm_reg(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ----------------- start learning ----------------------
best_loss = 1000000000000
train_loss_vector = []   # batch loss on training set
val_loss_vector = []    # total loss on test set
best_epoch = 0

for e in range(epoch):
    # ----------- training start --------------
    # ---- forward ---------
    for batch_idx, (data, target) in enumerate(train_loader):
        # data = data.view(time_step, -1, 1)  这种操作会打乱数据的顺序！
        data = data.permute(1, 0, 2)   # switch dim0 and dim1
        # make the input data shape is (seq, batch, feature)
        print(f'the shape of data is: {data.size()}')
        print(f'the shape of data target is: {target.size()}')
        model.train()  # training mode, dropout will be applied
        pred = model(data)      #
        # print(f'the shape of pred is: {pred.size()}')
        loss = criterion(pred, target)
        # ------- backward --------
        optimizer.zero_grad()
        loss.backward()   # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Variables with requires_grad=True.
    # After this call var_x.grad will be Variables holding the gradient
    # of the loss with respect to var_x.
        optimizer.step()
        train_loss_vector.append(loss.item())
        if batch_idx % 2 == 0:
            print(f'batch_idx {batch_idx}, loss is {loss.item()}')

    # ------------- start validation -------------------
    val_loss = 0.0
    for batch_idx1, (data1, target1) in enumerate(test_loader):
        data1 = data1.permute(1, 0, 2)  # convert shape (b, s, i) into (seq, batch, input_size)
        model.eval()
        pred1 = model(data1)
        loss = criterion(pred1, target1)
        val_loss += loss.item() * len(target1)   # since MSE is mean square error of each batch!
    total_val_loss = val_loss / test_size
    val_loss_vector.append(total_val_loss)
    print(f'epoch {e+1}, total val loss is {total_val_loss}')
    # --------- save the loss vector --------------
    np.savetxt(f'training loss_{time_step}_{L} .txt', train_loss_vector, delimiter='\n')
    np.savetxt(f'val loss_{time_step}_{L} .txt', val_loss_vector, delimiter='\n')

    if total_val_loss < best_loss:
        best_loss = total_val_loss
        best_epoch = e + 1
        save_path = f'./best_model_{time_step}_{L}.pth'
        torch.save(model.state_dict(), save_path)  # saving the model
# --------- record results ----------------
print(f'the best epoch is: {best_epoch}')
model.load_state_dict(
        torch.load(f'./best_model_{time_step}_{L}.pth', map_location=lambda storage, loc: storage))  # 加载模型并将参数赋值给刚创建的模型
model = model.eval()
dataset1 = sc.transform(dataset)
data_x, data_y = create_dataset(dataset1, look_back=time_step, look_ahead=L)
data_x = torch.Tensor(data_x)
data_x = data_x.permute(1, 0, 2)
prediction = model(data_x)     # shape  (batch, output_size)
prediction = prediction.detach().numpy()
data_y = np.array(data_y, np.float32)
prediction1 = np.hstack((data_y, prediction))
pred1list = prediction1.tolist()
with open(f'prediction of all.txt', 'w', encoding='utf-8') as f:
    for sample in pred1list:
        sample = [str(x) for x in sample]
        f.write(' '.join(sample) + '\n')
# ---------- Visualization ------------------
plt.clf()   
plt.axvline(x=train_size, c='r', linestyle='--')   # 画一条竖线
plt.plot(prediction, 'r', label='pred')
plt.plot(data_y, 'b', label='real')
plt.legend(loc='best')
plt.show()
