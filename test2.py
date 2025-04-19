import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from gplearn.genetic import SymbolicRegressor  # Thêm phần này để sử dụng symbolic regression

# Đọc dữ liệu
data = pd.read_csv('C:/Project/Dự đoán nông nghiệp/crop_yield.csv', 
                    usecols=['Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
                            'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition',
                            'Days_to_Harvest', 'Yield_tons_per_hectare'])

# Làm tròn giá trị năng suất
data['Yield_tons_per_hectare'] = data['Yield_tons_per_hectare'].round(2)

# Tách features và target
target = 'Yield_tons_per_hectare'
x = data.drop(target, axis=1)
y = data[target]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu
preprocessor = ColumnTransformer(transformers=[  
    ("one_hot_feature", OneHotEncoder(), ["Soil_Type", "Crop", "Weather_Condition"]),
    ("bool_feature", Binarizer(), ["Fertilizer_Used", "Irrigation_Used"]),
    ("stander_feature", StandardScaler(), ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"])
])

# Xây dựng pipeline
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", SymbolicRegressor(
        population_size=5000,  # Số lượng quần thể trong mỗi thế hệ
        generations=20,  # Số thế hệ
        tournament_size=20,  # Kích thước giải đấu
        stopping_criteria=0.01,  # Tiêu chí dừng (chỉ số phù hợp)
        const_range=(0, 1),  # Giới hạn giá trị hằng số
        metric='mean absolute error',  # Đánh giá theo MAE
        parsimony_coefficient=0.001,  # Hệ số phạt phức tạp của mô hình
        random_state=42  # Đảm bảo tái tạo được kết quả
    ))
])

# Huấn luyện mô hình
reg.fit(x_train, y_train)

# Dự đoán
y_predict = reg.predict(x_test)

# # In kết quả dự đoán và so sánh với giá trị thực
# for i, j in zip(y_predict, y_test):
#     print("Predicted value: {}. Actual value: {}".format(i, j))

# Tính các chỉ số đánh giá mô hình
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
