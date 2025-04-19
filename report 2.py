import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import LinearSVR

# Đọc dữ liệu
data = pd.read_csv('crop_yield.csv',
                    usecols=['Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
                            'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition',
                            'Days_to_Harvest', 'Yield_tons_per_hectare'])
#print(data)

# # Hiển thị thông tin tổng quát
# data.info()

# # Hiển thị các thông tin thống kê mô tả
# pd.set_option('display.max_columns', None)  # Hiển thị tất cả các cột
# pd.set_option('display.width', 1000)  # Tăng độ rộng dòng hiển thị
# print(data.describe())

# # Đếm số lượng giá trị thiếu trong từng cột
# print(data.isnull().sum())

# # Đếm số lượng hàng bị trùng lặp
# print(data.duplicated().sum())

# # Biểu đồ phân phối năng suất
# plt.figure(figsize=(10, 5))
# sns.histplot(data['Yield_tons_per_hectare'], bins=30, kde=True, color='blue')
# plt.title('Biểu đồ phân phối năng suất (Yield per hectare)', fontsize=14)
# plt.xlabel('Năng suất (tấn/ha)', fontsize=12)
# plt.ylabel('Tần suất', fontsize=12)
# plt.show()

# # Tạo biểu đồ phân tán giữa luợng mưa và năng suất
# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=data['Rainfall_mm'], y=data['Yield_tons_per_hectare'], color='blue')
# plt.title('Biểu đồ phân tán: Lượng mưa vs Năng suất', fontsize=14)
# plt.xlabel('Lượng mưa (mm)', fontsize=12)
# plt.ylabel('Năng suất (tấn/ha)', fontsize=12)
# plt.show()

# # Tạo biểu đồ hộp giữa Năng suất và Loại cây trồng
# plt.figure(figsize=(10, 5))
# sns.boxplot(x=data['Crop'], y=data['Yield_tons_per_hectare'], color='lightblue')
# plt.title('Biểu đồ hộp: Năng suất theo Loại cây trồng', fontsize=14)
# plt.xlabel('Loại cây trồng', fontsize=12)
# plt.ylabel('Năng suất (tấn/ha)', fontsize=12)
# plt.xticks(rotation=45)
# plt.show()

# # Tạo biểu đồ cột cho năng suất trung bình theo loại đất trồng
# avg_yield_by_soil = data.groupby('Soil_Type')['Yield_tons_per_hectare'].mean().reset_index()
# plt.figure(figsize=(15, 8))
# sns.barplot(x='Soil_Type', y='Yield_tons_per_hectare', data=avg_yield_by_soil, palette='viridis')
# plt.title('Năng suất trung bình theo Loại đất trồng', fontsize=18)
# plt.xlabel('Loại đất trồng', fontsize=15)
# plt.ylabel('Năng suất trung bình (tấn/ha)', fontsize=15)
# for index, value in enumerate(avg_yield_by_soil['Yield_tons_per_hectare']):
#     plt.text(index, value + 0.1, f'{value:.2f}', ha='center', fontsize=15, color='black')
# plt.xticks(rotation=45)
# plt.show()


# # Tính toán ma trận tương quan
# correlation_matrix = data.corr()
# print(correlation_matrix)

# Tách features và target
target = 'Yield_tons_per_hectare'
x = data.drop(target, axis=1)
y = data[target]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training size:", len(y_train))
print("Testing size:", len(y_test))

