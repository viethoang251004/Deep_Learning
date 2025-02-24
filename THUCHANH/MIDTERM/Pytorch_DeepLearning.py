import torch
import torch.nn as nn
import torch.optim as optim

# Autograd trong PyTorch cung cấp cơ chế tự động tính toán đạo hàm, 
#giúp đơn giản hóa quá trình huấn luyện mô hình deep learning
# Định nghĩa hàm mất mát tùy chỉnh
class CustomLoss(nn.Module):
    # Gọi constructor method của class cha là Module dùng cho class con CustomLoss
    def __init__(self):
        super(CustomLoss, self).__init__()

    # Method forward này dùng để tính sai số trung bình tuyệt đối trong process forward propagation(lan truyền thuận)
    def forward(self, y_pred, y_true):
        # Tính toán hàm mất mát tùy chỉnh
        # y_pred: output dự đoán
        # y_true: output thực tế
        loss = torch.mean(torch.abs(y_pred - y_true))  # mean: Sai số trung bình tuyệt đối
        return loss

# Định nghĩa mô hình mạng nơ-ron đơn giản
class SimpleNet(nn.Module):
    
    def __init__(self):
        # Gọi constructor method của class cha là Module dùng cho class con SimpleNet
        super(SimpleNet, self).__init__()
        # Tạo lớp tuyến tính với input có size là 10(có thể chứa được 10 elements) và output có size là 5(chứa 5 elements)
        self.fc1 = nn.Linear(10, 5)
        # Tạo lớp tuyến tính với input có size là 5(chứa tối đa 5 elements) và output có size là 1(chứa được 1 element)
        self.fc2 = nn.Linear(5, 1)
        # Tạo một biến đại diện cho class nn.ReLU(), thực hiện phép kích hoạt phi tuyến trong mạng nơ-ron.
        # Phép kích hoạt phi tuyến được áp dụng lên output hoặc một chuỗi các class linear
        # Hàm ReLU kích hoạt max(0, x) lên mỗi phần tử đầu ra. Nếu input là dương thì output giữ nguyên. Nếu input là âm thì output là 0
        self.relu = nn.ReLU()

    # Method forward nhận input là x
    def forward(self, x):
        # x là input được truyền qua class linear self.fc1 ở dòng 26
        # self.fc1 có thể là fully connected layer(lớp tuyến tính) vì
        # nó thực hiện phép biến đổi tuyến tính(
        # ánh xạ tuyến tính: ánh xạ giữa hai mô đun(hai ko gian vector) 
        # mà vẫn bảo toàn được tổ hợp tuyến tính - nghĩa là các thao tác cộng và nhân vô hướng vector)
        # trên x bằng cách nhân weighted matrix vs thêm bias(vector độ lệch). Kết quả là một tensor mới.
        # Tensor này nạp vào hàm relu để cho nó trả về giá trị dương và trả về 0 thông qua phép max(0, x)
        x = self.relu(self.fc1(x))
        # self.fc2 là một lớp tuyến tính khác tương tự self.fc1
        x = self.fc2(x)
        #Trả về x cho lớp tuyến tính cuối cùng
        return x

# Tạo các tensor đầu vào(x) và đầu ra(y) ngẫu nhiên
# Hàm torch.randn dùng để tạo một tensor với kích thước(shape) là (100,10)
#là các value ngẫu nhiên tuân theo phân phối chuẩn là Gaussian
#có trung bình bằng 0 và độ lệch chuẩn là 1
#-> x được gán là một tensor với shape là (100,10) với các value ngẫu nhiên
x = torch.randn(100, 10)
# y được thực hiện Tương tự như tensor x
y = torch.randn(100, 1)

# Khởi tạo mô hình và hàm mất mát tùy chỉnh
#Gán model là SimpleNet() có các attribute và method từ class cha là nn.Module và bao gồm các lớp tuyến tính vs hàm kích hoạt phi tuyến ở trên.
model = SimpleNet()
# Gán cho biến loss_fn là class CustomLoss() có các attribute và method từ class cha là nn.Module
#giống như class SimpleNet()
#ustomLoss() dùng để đánh giá sự sai khác giữa output mà model predict và output thực.
loss_fn = CustomLoss()

# Chọn một thuật toán tối ưu hóa (ví dụ: SGD hoặc Adam)
#optim.SGD nhận input là parameter(bao gồm ma trận trọng số và vector độ lệch) của model với tỉ lệ học các parameter đó là 0.01 của model
#Tỉ lệ học của model càng nhỏ thì đán đến quá trình học của model đó càng chậm nhưng chúng sẽ đảm bảo đạt đến giá trị tối ưu
#Còn nếu tỉ lệ học nó càng cao thì dẫn đến quá trình học của model càng nhanh nhưng nó sẽ không cho ra output như chúng ta mong đợi.
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent (SGD)

# Vòng lặp huấn luyện
# Mỗi epoch đại diện cho một lần chạy qua toàn bộ dataset
for epoch in range(100):
    # Lan truyền thuận
    # Thực hiện forward pass(lan truyền thuận) với input là x-một tensor để dự đoán y có output là gì.
    y_pred = model(x)
    # Tính value của loss func bằng cách so sánh ouput dự đoán là y của model với ouput thực tế
    loss = loss_fn(y_pred, y)

    #Gradient đại diện cho vector đạo hàm riêng của một hàm số đối với biến độc lập của nó.
    # Lan truyền ngược (tính toán gradient)
    # Đặt gradient của tham số trong model = 0 để tránh việc bị dư thừa gradient quá trình backward pass(lan truyền ngược)
    optimizer.zero_grad()
    #Thực hiện lan truyền ngược (backward pass) để tính gradient cho các paramter dựa trên class CustomLoss
    #Gradient sẽ được giữ lại cho các bước tiếp theo.
    loss.backward()

    # Cập nhật trọng số của model dựa trên gradien được tính ở trên
    optimizer.step()

    #Một epoch đại diện cho một lần chạy qua toàn bộ dữ liệu huấn luyện một lượt.
    # In tiến trình
    # Nếu epoch hiện tại là bội số của 10(tức là sau mỗi 10 epochs) thì in ra tt về epoch
    #và giá trị mất mát để cho dev dễ follow
    if (epoch + 1) % 10 == 0:
        # IN ra tt về epoch hiện tại và giá trị mất mát tương ứng.
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# Đánh giá mô hình đã huấn luyện
#Tạo một ngữ cảnh(context) không có vector đạo hàm riêng với biến độc lập của nó 
with torch.no_grad():
    #Impliment forward pass(lan truyền thuận) của model đã trained để dự đoán output y cho input x
    y_pred = model(x)
    # Tính hàm mất mát bằng cách so sánh output y được dự đoán với output y thực tế(sử dụng hàm loss_fn)
    loss = loss_fn(y_pred, y)
    # In ra value của hàm mất mát cúi cùng để đánh giá performance của model đã được huấn luyện dựa trên dataset đã có.
    print(f"Loss cuối cùng: {loss.item()}")
# -> Loss cúi cùng càng tiến về 0 thì model sẽ cho ra output dự đoán vs output thực tế càng giống nhau và ít có sự sai lệch hơn về kết quả cho ra.