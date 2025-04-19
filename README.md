# 🌾 ỨNG DỤNG HỒI QUY TUYẾN TÍNH TRONG DỰ ĐOÁN NĂNG SUẤT NÔNG NGHIỆP DỰA TRÊN ĐIỀU KIỆN MÔI TRƯỜNG 

Ứng dụng này được phát triển bằng **Streamlit** nhằm dự đoán năng suất cây trồng (tính bằng tấn/hecta) dựa trên các yếu tố môi trường và đầu vào nông nghiệp như loại đất, loại cây, lượng mưa, nhiệt độ, phân bón, tưới tiêu và điều kiện thời tiết.

---

## 📌 Mục tiêu dự án
- Xây dựng một mô hình dự đoán năng suất cây trồng dựa trên dữ liệu môi trường.
- Cung cấp giao diện trực quan giúp người dùng dễ dàng tương tác và dự đoán trực tiếp.
- Phân tích dữ liệu thông qua các biểu đồ để hiểu rõ mối quan hệ giữa các yếu tố đầu vào và năng suất.

---

## 📁 Cách chạy chương trình
```bash
pip install -r requirements.txt
streamlit run "C:\Users\Admin\Documents\DEAN\Du doan nong nghiep\crop_yield.py" 
```

---

## 🧠 Mô hình sử dụng
- **LinearSVR** (Support Vector Regression dạng tuyến tính) từ thư viện `scikit-learn`
- Dữ liệu được tiền xử lý bằng `Pipeline`, bao gồm:
  - **OneHotEncoder** cho các cột phân loại: `Soil_Type`, `Crop`, `Weather_Condition`
  - **Binarizer** cho các giá trị boolean: `Fertilizer_Used`, `Irrigation_Used`
  - **StandardScaler** cho các biến liên tục: `Rainfall_mm`, `Temperature_Celsius`, `Days_to_Harvest`

---

## 🔄 Các bước chính trong ứng dụng
1. Đọc dữ liệu từ file `crop_yield.csv`
2. Hiển thị mẫu dữ liệu và thống kê mô tả ban đầu
3. Tùy chọn hiển thị biểu đồ: Scatter Plot, Box Plot hoặc Histogram để phân tích mối liên hệ giữa năng suất và các đặc trưng
4. Xây dựng pipeline để huấn luyện mô hình LinearSVR
5. Hiển thị các chỉ số đánh giá mô hình: MAE, MSE, R²
6. Giao diện người dùng nhập liệu trực tiếp để dự đoán năng suất mới

---

## 📊 Bộ dữ liệu sử dụng (`crop_yield.csv`)
| Tên cột               | Ý nghĩa |
|-----------------------|---------|
| `Soil_Type`           | Loại đất trồng:<br> - Clay: Đất sét<br> - Sandy: Cát<br> - Silt: Đất bùn<br> - Loam: Đất thịt<br> - Peaty: Đất than bùn<br> - Chalky: Đất vôi |
| `Crop`                | Loại cây trồng:<br> - Wheat: Lúa mì<br> - Rice: Lúa gạo<br> - Maize: Ngô<br> - Barley: Lúa mạch<br> - Soybean: Đậu nành<br> - Cotton: Bông |
| `Rainfall_mm`         | Lượng mưa nhận được trong giai đoạn phát triển cây trồng (mm) |
| `Temperature_Celsius` | Nhiệt độ trung bình trong giai đoạn phát triển cây trồng (°C) |
| `Fertilizer_Used`     | Có sử dụng phân bón không:<br> - True: Có<br> - False: Không |
| `Irrigation_Used`     | Có sử dụng tưới tiêu trong giai đoạn phát triển:<br> - True: Có<br> - False: Không |
| `Weather_Condition`   | Điều kiện thời tiết chủ đạo:<br> - Sunny: Nắng<br> - Rainy: Mưa<br> - Cloudy: Có mây |
| `Days_to_Harvest`     | Số ngày cần thiết để thu hoạch sau khi gieo trồng |
| `Yield_tons_per_hectare` | Năng suất cây trồng (tấn/hecta) |

---

## 📈 Đánh giá mô hình
Sau khi huấn luyện mô hình LinearSVR, các chỉ số đánh giá bao gồm:
- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình
- **MSE (Mean Squared Error)**: Sai số bình phương trung bình
- **R² Score**: Hệ số xác định (mức độ giải thích của mô hình)

---

## 💻 Giao diện người dùng
- Ứng dụng có giao diện trực quan giúp người dùng:
  - Xem dữ liệu mẫu và biểu đồ phân tích
  - Tùy chọn loại biểu đồ và đặc trưng phân tích
  - Nhập các thông số đầu vào như loại đất, cây trồng, lượng mưa... để dự đoán năng suất

---

## 👨‍🌾 Tác giả
- Sinh viên thực hiện: Tạ Đình Phong
- Giảng viên hướng dẫn: ThS Nguyễn Phương Nam

---

