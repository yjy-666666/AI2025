 # -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# 导入所需库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# %%
# load data（加载数据，Windows 路径用反斜杠，这里用原始字符串避免转义）
data = pd.read_csv(r'E:\aiSummerCamp2025-master\aiSummerCamp2025-master\day1\assignment\data\train.csv')  # 修改为 Windows 路径格式，根据实际数据位置调整
df = data.copy()
# 随机查看 10 条数据
df.sample(10)

# %%
# delete some features that are not useful for prediction（删除无用特征）
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
# 查看数据信息
df.info()

# %%
# check if there is any NaN in the dataset（检查并处理缺失值）
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))

# %%
# convert categorical data into numerical data using one-hot encoding（独热编码处理分类特征）
df = pd.get_dummies(df)
# 随机查看编码后 10 条数据
df.sample(10)

# %% 
# separate the features and labels（分离特征和标签）
X = df.drop(columns=['Survived'])
y = df['Survived']

# %%
# train-test split（划分训练集和测试集）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 测试集占比 20%，固定随机种子方便复现
)

# %%
# build model（构建三个分类模型：SVM、KNN、随机森林）
svm_model = SVC(random_state=42)
knn_model = KNeighborsClassifier()
rf_model = RandomForestClassifier(random_state=42)

# 训练模型
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# %%
# predict and evaluate（预测并评估模型，同时记录准确率用于绘图）
model_names = ['SVM', 'KNN', 'Random Forest']
accuracies = []

# SVM 模型评估
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
accuracies.append(svm_accuracy)
print("=== SVM 模型评估结果 ===")
print(f"准确率: {svm_accuracy}")
print(classification_report(y_test, svm_pred))

# KNN 模型评估
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
accuracies.append(knn_accuracy)
print("=== KNN 模型评估结果 ===")
print(f"准确率: {knn_accuracy}")
print(classification_report(y_test, knn_pred))

# 随机森林模型评估
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
accuracies.append(rf_accuracy)
print("=== 随机森林模型评估结果 ===")
print(f"准确率: {rf_accuracy}")
print(classification_report(y_test, rf_pred))

# %%
# 绘制准确率对比柱状图
plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.xlabel('Model Name')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Classification Models')
plt.ylim(0, 1)  # 准确率范围 0-1
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center')  # 在柱子上方显示准确率数值
plt.show()
