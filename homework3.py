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

# 划分训练集和测试集
train, test = df.loc[df.index <= '2009-12-31'], df.loc[df.index > '2009-12-31']
print(f"训练集大小：{len(train)}，测试集大小：{len(test)}")

# 数据标准化
# 只对数值特征进行标准化
numeric_features = [col for col in df.columns if col != 'datetime']
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[numeric_features])
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

# 创建训练和测试序列
X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
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
test_dataset = TimeSeriesDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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

# 训练模型
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印训练信息
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        # 学习率调整
        scheduler.step(avg_loss)
    
    return model

# 训练模型
print("开始训练模型...")
model = train_model(model, train_loader, criterion, optimizer, device, epochs=10)

# 评估模型
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    predictions = []
    actuals = []
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 预测
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # 保存预测和实际值
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
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