import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# ============== 配置部分 ==============
last_digit_of_id = 5  

# ============== 数据增强（防止过拟合的方法1）==============
# 训练集使用数据增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 验证集不使用数据增强
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# ============== 加载数据集 ==============
# 加载训练集
train_dataset_full = datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=train_transform)

# 根据QMUL ID划分训练集和验证集
split_ratio = 0.7 if last_digit_of_id <= 4 else 0.8
train_size = int(split_ratio * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size

train_dataset, val_dataset_temp = random_split(train_dataset_full, 
                                                [train_size, val_size])

# 为验证集重新应用transform（不使用数据增强）
val_dataset_full = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=val_transform)
val_dataset = torch.utils.data.Subset(val_dataset_full, val_dataset_temp.indices)

# DataLoaders
batch_size = 32 + last_digit_of_id
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                         shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                       shuffle=False, num_workers=2, pin_memory=True)

print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
print(f"批次大小: {batch_size}")

# ============== 改进的模型架构（添加Dropout和BatchNorm）==============
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        
        # 使用卷积层代替全连接层（更适合图像）
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 批归一化（防止过拟合的方法2）
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Dropout2d(0.2),  # Dropout（防止过拟合的方法3）
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Dropout2d(0.3),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4
            nn.Dropout2d(0.4),
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 创建模型并移到GPU
model = ImprovedCNN(dropout_rate=0.5).to(device)
print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

# ============== 训练配置 ==============
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001 + (last_digit_of_id * 0.0001)
num_epochs = 100 + last_digit_of_id

# 使用Adam优化器 + 学习率衰减（防止过拟合的方法4）
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                  factor=0.5, patience=5, verbose=True)

print(f"\n学习率: {learning_rate}")
print(f"训练轮数: {num_epochs}")
print(f"权重衰减: 1e-4 (L2正则化)")

# ============== 早停机制（防止过拟合的方法5）==============
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0

early_stopping = EarlyStopping(patience=15)

# ============== 训练循环 ==============
train_losses = []
train_accuracies = []
val_accuracies = []
best_val_acc = 0.0

print("\n开始训练...")
print("=" * 60)

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        # 移到GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            # 移到GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    
    # 记录结果
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    # 学习率调度
    scheduler.step(val_accuracy)
    
    # 保存最佳模型
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} | "
              f"训练准确率: {train_accuracy:.2f}% | "
              f"验证准确率: {val_accuracy:.2f}% ⭐ (最佳)")
    else:
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} | "
              f"训练准确率: {train_accuracy:.2f}% | "
              f"验证准确率: {val_accuracy:.2f}%")
    
    # 早停检查
    early_stopping(val_accuracy)
    if early_stopping.early_stop:
        print(f"\n早停触发！在第 {epoch+1} 轮停止训练。")
        break

print("=" * 60)
print(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%")

# ============== 可视化结果 ==============
# 绘制损失曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label="训练损失")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="训练准确率")
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="验证准确率")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n图表已保存为 'training_results.png'")

# ============== 加载最佳模型进行最终评估 ==============
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_val_accuracy = 100 * correct / total
print(f"\n最佳模型的验证准确率: {final_val_accuracy:.2f}%")

# 分析过拟合情况
overfitting_gap = train_accuracies[-1] - val_accuracies[-1]
print(f"\n过拟合分析:")
print(f"训练准确率: {train_accuracies[-1]:.2f}%")
print(f"验证准确率: {val_accuracies[-1]:.2f}%")
print(f"准确率差距: {overfitting_gap:.2f}%")

if overfitting_gap < 5:
    print("✅ 模型表现良好，过拟合程度较低")
elif overfitting_gap < 10:
    print("⚠️ 存在轻微过拟合")
else:
    print("❌ 存在明显过拟合，建议增强正则化")