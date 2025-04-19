import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import LinearSVR

# # Tạo mảng từ danh sách
# arr = np.array([1, 2, 3, 4, 5])
# print(arr)
# # Tạo mảng 2 chiều từ danh sách lòng nhau
# matrix = np.array([[1, 2, 3], [4, 5, 6]])
# print(matrix)


# # Tạo màng từ ở đến 9
# arr = np.arange(0,10)
# print(arr)
# # Tạo mảng từ 1 đến 9 với bước nhảy 2
# arr= np.arange(1, 9, 2)
# print(arr)

# # Tạo mảng 1 chiều toàn số 9 với kích thước 5
# zeros_arr = np.zeros(5)
# print(zeros_arr)
# # Tạo máng 2x3 toàn số 1
# ones_arr = np.ones((2,3))
# print(ones_arr)

# # Tạo mảng 1 chiều từ ở đến 8
# arr = np.arange(9)
# # Chuyển đổi thành màng 3x3
# reshaped_arr = arr.reshape((3, 3))
# print(reshaped_arr)

# arr = np.array([1, 2, 3, 4, 5])
# # Tính giá trị trung bình
# mean_value = np.mean(arr)
# print (f'Giá trị trung bình: {mean_value}')
# sum_value = np.sum(arr)
# # Tính tổng
# print(f'Tổng: {sum_value}')
# # Tìm giá trị lớn nhất và nhỏ nhất
# max_value = np.max(arr)
# min_value = np.min(arr)
# print(F'Giá trị lớn nhất: {max_value}, Giá trị nhỏ nhất: {min_value}')

# # Tạo hai ma trận 2x2
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[5, 6], [7, 8]])
# # Tính tích ma trận
# result = np.dot(A, B)
# print(result)

# data = {
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'City': ['New York', 'Paris', 'London']
# }
#
# df = pd.DataFrame(data)
# print(df)


# Đọc dữ liệu từ file CSV
# df = pd.read_csv('gradedata.csv')

# # Lọc dữ liệu với điều kiện age > 18
# filtered_df = df[df['age'] > 18]
# print(filtered_df)
#
# # Sử dụng loc để truy vấn theo nhãn (chỉ mục hàng)
# row = df.loc[1]
# print(row)
#
# # Sử dụng iloc để truy vấn theo vị trí chỉ số
# row = df.iloc[2]
# print(row)

# # Tính giá trị trung bình của cột Age
# mean_age = df['age'].mean()
# print(f'Trung bình tuổi: {mean_age}')
#
# # Tính tổng số tuổi
# total_age = df['age'].sum()
# print(f'Tổng số tuổi: {total_age}')
#
# # Hiển thị thống kê mô tả
# print(df.describe())

# # Tạo dữ liệu ngẫu nhiên từ phân phối chuẩn
# data = np.random.randn(1000)
#
# # Vẽ biểu đồ histogram
# plt.hist(data, bins=30, color='blue', alpha=0.7, edgecolor='black')
#
# # Thêm tiêu đề và nhãn
# plt.title('Biểu đồ phân phối tần số', fontsize=14)
# plt.xlabel('Giá trị', fontsize=12)
# plt.ylabel('Tần suất', fontsize=12)
#
# # Hiển thị biểu đồ
# plt.show()

# # Tạo dữ liệu ngẫu nhiên
# x = np.random.rand(50)
# y = np.random.rand(50)
#
# # Vẽ biểu đồ phân tán (scatter plot)
# plt.scatter(x, y, color='green', alpha=0.5, edgecolors='black')
#
# # Thêm tiêu đề và nhãn
# plt.title('Biểu đồ phân tán', fontsize=14)
# plt.xlabel('X', fontsize=12)
# plt.ylabel('Y', fontsize=12)
#
# # Hiển thị biểu đồ
# plt.show()

# # Danh mục và giá trị tương ứng
# categories = ['A', 'B', 'C', 'D']
# values = [10, 24, 36, 18]
#
# # Vẽ biểu đồ cột
# plt.bar(categories, values, color='orange', edgecolor='black')
#
# # Thêm tiêu đề và nhãn
# plt.title('Biểu đồ thanh', fontsize=14)
# plt.xlabel('Danh mục', fontsize=12)
# plt.ylabel('Giá trị', fontsize=12)
#
# # Hiển thị biểu đồ
# plt.show()

# # Dữ liệu cho biểu đồ
# sizes = [15, 30, 45, 10]
# labels = ['Phần A', 'Phần B', 'Phần C', 'Phần D']
# colors = ['red', 'blue', 'green', 'yellow']
#
# # Vẽ biểu đồ tròn
# plt.figure(figsize=(6, 6))  # Đặt kích thước biểu đồ
# plt.pie(
#     sizes, labels=labels, colors=colors,
#     autopct='%1.1f%%', startangle=90,
#     wedgeprops={'edgecolor': 'black'}
# )
#
# # Tiêu đề biểu đồ
# plt.title('Biểu đồ hình tròn', fontsize=14)
#
# # Hiển thị biểu đồ
# plt.show()

# # Tạo DataFrame mẫu
# data = pd.DataFrame({
#     'Chiều cao (cm)': [150, 160, 170, 180, 190],
#     'Cân nặng (kg)': [50, 60, 65, 80, 90]
# })
#
# # Thiết lập kích thước biểu đồ
# plt.figure(figsize=(6, 4))
#
# # Vẽ biểu đồ phân tán
# sns.scatterplot(x='Chiều cao (cm)', y='Cân nặng (kg)', data=data, color='blue', s=100, edgecolor='black')
#
# # Thêm tiêu đề và nhãn
# plt.title('Biểu đồ phân tán: Chiều cao vs Cân nặng', fontsize=12)
# plt.xlabel('Chiều cao (cm)', fontsize=11)
# plt.ylabel('Cân nặng (kg)', fontsize=11)
#
# # Hiển thị biểu đồ
# plt.show()

# # Tạo DataFrame mẫu
# data = pd.DataFrame({
#     'Chiều cao (cm)': [150, 160, 170, 180, 190],
#     'Cân nặng (kg)': [50, 60, 65, 80, 90]
# })
#
# # Thiết lập kích thước biểu đồ
# plt.figure(figsize=(6, 4))
#
# # Vẽ biểu đồ đường (line plot)
# sns.lineplot(x='Chiều cao (cm)', y='Cân nặng (kg)', data=data, marker='o', color='blue', linewidth=2)
#
# # Thêm tiêu đề và nhãn trục
# plt.title('Mối quan hệ giữa Chiều cao và Cân nặng', fontsize=12)
# plt.xlabel('Chiều cao (cm)', fontsize=11)
# plt.ylabel('Cân nặng (kg)', fontsize=11)
#
# # Hiển thị biểu đồ
# plt.show()
#
# # Tạo DataFrame mẫu
# data = pd.DataFrame({
#     'Chiều cao (cm)': [150, 160, 170, 180, 190],
#     'Cân nặng (kg)': [50, 60, 65, 80, 90]
# })
#
# # Thiết lập kích thước biểu đồ
# plt.figure(figsize=(6, 4))
#
# # Vẽ biểu đồ histogram (biểu đồ tần suất) cho cột Cân nặng
# sns.histplot(data['Cân nặng (kg)'], bins=5, color='skyblue', edgecolor='black')
#
# # Thêm tiêu đề và nhãn trục
# plt.title('Phân bố cân nặng', fontsize=12)
# plt.xlabel('Cân nặng (kg)', fontsize=11)
# plt.ylabel('Tần suất', fontsize=11)
#
# # Hiển thị biểu đồ
# plt.show()
#
# # Tạo DataFrame mẫu
# data = pd.DataFrame({
#     'Chiều cao': [150, 160, 170, 180, 190],
#     'Cân nặng': [50, 60, 65, 80, 90]
# })
#
# # Vẽ biểu đồ boxplot
# plt.figure(figsize=(8, 5))
# sns.boxplot(x='Chiều cao', y='Cân nặng', data=data)
#
# # Hiển thị biểu đồ
# plt.title("Biểu đồ phân tán giữa Chiều cao và Cân nặng")
# plt.xlabel("Chiều cao (cm)")
# plt.ylabel("Cân nặng (kg)")
# plt.show()
#
# # Tạo ma trận dữ liệu ngẫu nhiên (5x5)
# data_matrix = np.random.rand(5, 5)
#
# # Vẽ heatmap
# plt.figure(figsize=(6, 5))
# sns.heatmap(data_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
#
# # Thêm tiêu đề biểu đồ
# plt.title("Heatmap Ma trận Ngẫu nhiên")
#
# # Hiển thị biểu đồ
# plt.show()
#
# print("Hello, World!")
#
# my_list = [1, 2, 3, 4, 5]
# print(len(my_list))
#
# numbers = [1, 2, 3, 4, 5]
# print(sum(numbers))
#
# numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print("Max:", max(numbers))
# print("Min:", min(numbers))
#
# num = -19
# print(abs(num))
#
# pi = 3.14159
# print(round(pi, 2))
#
# numbers = [12, -4, 7, -19, 1, 0]
# print(sorted(numbers))
#
# print(type(5))
# print(type([1, 2, 3]))
# print(type("a"))
#
# for i in range(5):
#     print(i)
#
# fruits = ['apple', 'banana', 'cherry']
# for index, fruit in enumerate(fruits):
#     print(index, fruit)
#
# names = ['John', 'Jane', 'Doe']
# ages = [25, 30, 22]
# for name, age in zip(names, ages):
#     print(f"{name} is {age} years old")
#
# numbers = [1, 2, 3, 4, 5]
# squared = list(map(lambda x: x ** 2, numbers))
# print(squared)
#
# numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
# print(even_numbers)
#
# name = input("Enter your name: ")
# print(f"Hello, {name}!")

# from sklearn.model_selection import train_test_split
# #
# # Dữ liệu mẫu
# x = [[1, 2], [3, 4], [5, 6], [7, 8]]
# y = [0, 1, 0, 1]
#
# # Chia dữ liệu thành 80% huấn luyện và 20% kiểm thử
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#
# print("Tập huấn luyện:", x_train, y_train)
# print("Tập kiểm thử:", x_test, y_test)

# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
# # Tạo dữ liệu số
# X = np.array([[10], [20], [30], [40], [50]])
#
# # Chuẩn hóa dữ liệu
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# print("Dữ liệu đã chuẩn hóa:\n", X_scaled)

# from sklearn.preprocessing import OneHotEncoder
#
# # Dữ liệu phân loại
# data = [['Red'], ['Blue'], ['Green'], ['Red']]
#
# # Mã hóa one-hot
# encoder = OneHotEncoder(sparse_output=False)
# encoded_data = encoder.fit_transform(data)
#
# print("Dữ liệu sau khi one-hot encoding:\n", encoded_data)

# from sklearn.preprocessing import Binarizer
# import numpy as np
#
# # Dữ liệu liên tục
# X = np.array([[0.1], [0.5], [0.8], [1.2]])
#
# # Biến đổi thành giá trị nhị phân với ngưỡng = 0.5
# binarizer = Binarizer(threshold=0.5)
# X_binarized = binarizer.fit_transform(X)
#
# print("Dữ liệu sau khi Binarizer:\n", X_binarized)

# from sklearn.compose import ColumnTransformer
# import pandas as pd
#
# # Tạo dữ liệu giả lập
# data = pd.DataFrame({
#     'Soil_Type': ['Clay', 'Sand', 'Loam'],
#     'Rainfall_mm': [800, 900, 1000]
# })
#
# # Định nghĩa bộ chuyển đổi
# preprocessor = ColumnTransformer([
#     ('num', StandardScaler(), ['Rainfall_mm']),  # Chuẩn hóa dữ liệu số
#     ('cat', OneHotEncoder(), ['Soil_Type'])      # One-hot encoding dữ liệu phân loại
# ])
#
# # Áp dụng tiền xử lý
# transformed_data = preprocessor.fit_transform(data)
# print("Dữ liệu sau tiền xử lý:\n", transformed_data)


# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# # Giá trị thực tế và dự đoán
# y_true = [10, 20, 30, 40, 50]
# y_pred = [12, 19, 31, 38, 49]
#
# # Tính toán các chỉ số đánh giá
# mae = mean_absolute_error(y_true, y_pred)
# mse = mean_squared_error(y_true, y_pred)
# r2 = r2_score(y_true, y_pred)
#
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R^2 Score: {r2}")

