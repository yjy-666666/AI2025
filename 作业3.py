import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from tqdm import tqdm
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
df = pd.read_csv('E:\\aiSummerCamp2025-master\\aiSummerCamp2025-master\\day3\\assignment\\data\\household_power_consumption.zip', sep=";")

# 检查数据
print("数据基本信息：")
df.info()

# 数据预处理
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis=1, inplace=True)

# 处理缺失值
print(f"处理前缺失值情况：\n{df.isnull().sum()}")
df.dropna(inplace=True)
print(f"处理后缺失值情况：\n{df.isnull().sum()}")

# 查看数据时间范围
print("数据时间范围：")
print("开始日期：", df['datetime'].min())
print("结束日期：", df['datetime'].max())

# 设置时间索引并排序
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

# 划分训练集、验证集和测试集
train, val, test = df.loc[df.index <= '2009-09-30'], df.loc[(df.index > '2009-09-30') & (df.index <= '2009-12-31')], df.loc[df.index > '2009-12-31']
print(f"训练集大小：{len(train)}，验证集大小：{len(val)}，测试集大小：{len(test)}")

# 数据标准化
# 只对数值特征进行标准化
numeric_features = [col for col in df.columns if col != 'datetime']
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[numeric_features])
val_scaled = scaler.transform(val[numeric_features])
test_scaled = scaler.transform(test[numeric_features])

# 为LSTM准备数据
def create_sequences(data, seq_length):
    """
    将时间序列数据转换为适合LSTM的序列数据
    参数:
        data: 输入的时间序列数据
        seq_length: 序列长度
    返回:
        X: 特征序列
        y: 目标值
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # 预测全局活动功率（第一列）
    return np.array(X), np.array(y)

# 设置序列长度
seq_length = 48  # 使用前48个时间步预测下一个时间步

# 创建训练、验证和测试序列
X_train, y_train = create_sequences(train_scaled, seq_length)
X_val, y_val = create_sequences(val_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
print(f"验证数据形状: X={X_val.shape}, y={y_val.shape}")
print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")

# 创建PyTorch数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建数据加载器
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取序列的最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型
input_size = X_train.shape[2]  # 特征数量
hidden_size = 128
num_layers = 2
output_size = 1
dropout = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

# 早停机制
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, patience=5):
    # 创建保存模型的目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.train()
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='models/best_model.pt')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss = 0
        model.train()
        
        # 创建训练进度条
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for i, (X_batch, y_batch) in train_progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 更新进度条信息
            train_progress.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        val_loss = 0
        model.eval()
        
        # 创建验证进度条
        val_progress = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Epoch {epoch+1}/{epochs} [Val]')
        
        with torch.no_grad():
            for i, (X_batch, y_batch) in val_progress:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # 预测
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                
                # 更新进度条信息
                val_progress.set_postfix({'loss': loss.item()})
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 打印训练和验证信息
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 早停检查
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('models/best_model.pt'))
    
    # 绘制训练和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model

# 训练模型
print("开始训练模型...")
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=30, patience=5)

# 评估模型
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0
    
    # 创建测试进度条
    test_progress = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing')
    
    with torch.no_grad():
        for i, (X_batch, y_batch) in test_progress:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 预测
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # 保存预测和实际值
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
            
            # 更新进度条信息
            test_progress.set_postfix({'loss': loss.item()})
    
    # 计算评估指标
    mse = mean_squared_error(actuals, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    print(f'测试集损失: {total_loss/len(test_loader):.4f}')
    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'均方根误差 (RMSE): {rmse:.4f}')
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    
    return np.array(predictions), np.array(actuals)

# 评估模型
print("开始评估模型...")
predictions, actuals = evaluate_model(model, test_loader, criterion, device)

# 反标准化预测值和实际值
# 复制一个标准化器来处理预测结果
pred_scaler = StandardScaler()
pred_scaler.mean_ = scaler.mean_[0]
pred_scaler.scale_ = scaler.scale_[0]

predictions_inv = pred_scaler.inverse_transform(predictions.reshape(-1, 1))
actuals_inv = pred_scaler.inverse_transform(actuals.reshape(-1, 1))

# 可视化预测结果
def plot_predictions(actual, predicted, title, start=0, end=200):
    plt.figure(figsize=(14, 6))
    plt.plot(actual[start:end], label='实际值')
    plt.plot(predicted[start:end], label='预测值')
    plt.title(title)
    plt.xlabel('时间步')
    plt.ylabel('全局活动功率 (千瓦)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制短期预测
plot_predictions(actuals_inv, predictions_inv, '电力消耗预测 - 短期')

# 绘制长期预测
plot_predictions(actuals_inv, predictions_inv, '电力消耗预测 - 长期', end=len(actuals_inv))

# 绘制误差分布
errors = actuals_inv - predictions_inv
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=50)
plt.title('预测误差分布')
plt.xlabel('误差值')
plt.ylabel('频率')
plt.grid(True)
plt.show()