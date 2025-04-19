import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyRegressor

# Đọc dữ liệu
data = pd.read_csv('C:\Project\Dự đoán nông nghiệp\crop_yield.csv', 
                    usecols=['Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
                            'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition',
                            'Days_to_Harvest', 'Yield_tons_per_hectare'])

# Chuyển cột Yield_tons_per_hectare thành biến mục tiêu
target = 'Yield_tons_per_hectare'
x = data.drop([target], axis=1)
y = data[target]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu
preprocessor = ColumnTransformer(transformers=[
    ("one_hot_feature", OneHotEncoder(), ["Soil_Type", "Crop", "Weather_Condition"]),
    ("bool_feature", StandardScaler(), ["Fertilizer_Used", "Irrigation_Used"]),
    ("stander_feature", StandardScaler(), ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"])
])

# Sử dụng LazyRegressor để tìm mô hình hồi quy tốt nhất
regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = regressor.fit(x_train, x_test, y_train, y_test)

# In kết quả mô hình
print(models)
