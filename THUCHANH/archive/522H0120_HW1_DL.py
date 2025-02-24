import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Tạo mô hình Neural Network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# Tạo dữ liệu giả định
train_data = torch.randn(100, 10)
train_labels = torch.randint(0, 2, (100,))
test_data = torch.randn(20, 10)
test_labels = torch.randint(0, 2, (20,))
# Tạo DataLoader cho dữ liệu
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
# Khởi tạo mô hình
model = MyModel()
# Định nghĩa hàm mất mát và bộ tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(model.parameters(), lr=0.01)  # Sử dụng optimizer SGD
optimizer2 = optim.Adam(model.parameters(), lr=0.01)  # Sử dụng optimizer Adam
# Huấn luyện mô hình
epochs = 10
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer1.step()
        optimizer2.step()

# Đánh giá mô hình trên tập kiểm tra
with torch.no_grad():
    test_outputs = model(test_data)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
    print("Accuracy on test data:", accuracy)