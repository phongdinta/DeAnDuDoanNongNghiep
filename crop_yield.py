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

# Làm tròn giá trị năng suất
data['Yield_tons_per_hectare'] = data['Yield_tons_per_hectare'].round(2)

# Hiển thị một số mẫu dữ liệu
st.subheader("Mẫu dữ liệu ban đầu")
st.write(data.sample(5))

# Hiển thị thông tin đặc điểm của bộ dữ liệu
st.subheader("Thông tin về bộ dữ liệu")
st.write(data.describe())

# Giới hạn dữ liệu biểu đồ chỉ lấy 1000 mẫu đầu tiên
sampled_data = data.head(1000)

# Sidebar lựa chọn
st.sidebar.header("Chọn thông số để hiển thị biểu đồ")
chart_type = st.sidebar.selectbox("Loại biểu đồ", ["Scatter Plot", "Box Plot", "Histogram"])
selected_feature = st.sidebar.selectbox("Chọn thông số để so sánh với năng suất", [col for col in data.columns if col != "Yield_tons_per_hectare"])

# Hiển thị biểu đồ
st.subheader("Biểu đồ phân tích dữ liệu")
fig, ax = plt.subplots()

if chart_type == "Scatter Plot":
    sns.scatterplot(x=sampled_data[selected_feature], y=sampled_data["Yield_tons_per_hectare"], ax=ax)
    ax.set_title(f"Scatter Plot: {selected_feature} vs Yield_tons_per_hectare")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Yield_tons_per_hectare")

elif chart_type == "Box Plot":
    sns.boxplot(x=sampled_data[selected_feature], y=sampled_data["Yield_tons_per_hectare"], ax=ax)
    ax.set_title(f"Box Plot: {selected_feature} vs Yield_tons_per_hectare")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Yield_tons_per_hectare")

elif chart_type == "Histogram":
    sns.histplot(sampled_data[selected_feature], kde=True, ax=ax)
    ax.set_title(f"Histogram: {selected_feature}")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")

st.pyplot(fig)

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
    ("model", LinearSVR())
])

# Huấn luyện mô hình
reg.fit(x_train, y_train)

# Dự đoán
y_predict = reg.predict(x_test)

# Tính các chỉ số đánh giá mô hình
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

# Streamlit UI
st.title("Crop Yield Prediction")

st.subheader("Dự đoán năng suất nông nghiệp dựa trên điều kiện môi trường")

# Hiển thị chỉ số đánh giá mô hình
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R² Score: {r2}")

# Tạo form cho người dùng nhập các giá trị
st.sidebar.header("Nhập thông số để dự đoán")

soil_type = st.sidebar.selectbox("Loại đất", data["Soil_Type"].unique())
crop = st.sidebar.selectbox("Loại cây trồng", data["Crop"].unique())
rainfall = st.sidebar.number_input("Lượng mưa (mm)", min_value=0, max_value=1000, value=100)
temperature = st.sidebar.number_input("Nhiệt độ (°C)", min_value=-10, max_value=50, value=25)
fertilizer_used = st.sidebar.selectbox("Có sử dụng phân bón không?", ['Có', 'Không'])
irrigation_used = st.sidebar.selectbox("Có sử dụng tưới tiêu không?", ['Có', 'Không'])
weather_condition = st.sidebar.selectbox("Điều kiện thời tiết", data["Weather_Condition"].unique())
days_to_harvest = st.sidebar.number_input("Số ngày đến thu hoạch", min_value=1, max_value=365, value=100)

# Tiền xử lý dữ liệu đầu vào
input_data = pd.DataFrame({
    'Soil_Type': [soil_type],
    'Crop': [crop],
    'Rainfall_mm': [rainfall],
    'Temperature_Celsius': [temperature],
    'Fertilizer_Used': [1 if fertilizer_used == 'Có' else 0],
    'Irrigation_Used': [1 if irrigation_used == 'Có' else 0],
    'Weather_Condition': [weather_condition],
    'Days_to_Harvest': [days_to_harvest]
})

# Dự đoán giá trị năng suất
predicted_yield = reg.predict(input_data)

# Hiển thị kết quả dự đoán
st.write(f"Dự đoán năng suất: {predicted_yield[0]:.2f} tấn/hectare")