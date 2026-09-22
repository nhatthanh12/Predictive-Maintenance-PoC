# NHẬT KÝ TIẾN ĐỘ DỰ ÁN: DENSO PREDICTIVE MAINTENANCE POC

## 1. MỤC TIÊU DỰ ÁN & TIÊU CHÍ KỸ THUẬT (CHẶNG NƯỚC RÚT - 15 NGÀY)
- **Hạn chót:** Hoàn thành trong 15 ngày để nộp hồ sơ Vòng Sơ Khảo (Mục tiêu: Top 10).
- **Mục tiêu kỹ thuật cốt lõi:**
  1. Xây dựng Pipeline sinh dữ liệu lỗi ảo (bu lông lỏng/mòn) chuẩn vật lý cơ khí (Physics-informed: sóng hài phân số 0.5x, sóng hài bậc cao 2x/3x, va đập ngắt quãng).
  2. Chứng minh định lượng: Mô hình AI khi có dữ liệu tăng cường (Synthetic Fault Data) cải thiện vượt trội chỉ số **Recall / F1-Score** và giảm **False Positive Rate** so với mô hình baseline.
  3. Xuất tối thiểu 3 biểu đồ trực quan (DPI 300) phục vụ báo cáo thuyết minh.

## 2. KIẾN TRÚC MÃ NGUỒN DỰ KIẾN
```text
denso-predictive-maintenance/
├── data/                  # Chứa file mẫu CSV/dữ liệu thử nghiệm
├── reports/figures/       # Chứa ảnh biểu đồ xuất ra cho báo cáo
├── src/
│   ├── baseline_generator.py   # Tầng 1: Sinh chuỗi thời gian bình thường (chuẩn CWRU)
│   ├── features.py             # Trích xuất đặc trưng công nghiệp (RMS, Kurtosis, FFT)
│   ├── fault_injector.py       # Tầng 2: Động cơ tiêm lỗi bu lông giả lập
│   └── train_eval.py           # Tầng 3: Huấn luyện, đánh giá & xuất biểu đồ so sánh
├── tests/                 # Script kiểm thử từng module
├── .clinerules            # Luật hệ thống bất khả xâm phạm
└── agent.md               # Bảng theo dõi tiến độ


3. KẾ HOẠCH HÀNH ĐỘNG CHI TIẾT THEO NGÀY
Ngày 1 - 3: Nền tảng Dữ liệu & Trích xuất Đặc trưng
[ ] Khởi tạo môi trường ảo và file requirements.txt (numpy, pandas, scipy, scikit-learn, matplotlib).

[ ] Xây dựng src/baseline_generator.py: Sinh dữ liệu rung động/dòng điện chuẩn hóa (Sampling rate 1000-12000 Hz, f_base=30 Hz, kèm nhiễu nền công nghiệp, random_state=42).

[ ] Xây dựng src/features.py: Trích xuất đặc trưng vật lý (Time-domain: RMS, Kurtosis, Crest Factor, Peak-to-Peak; Frequency-domain: Phổ biên độ FFT).

[ ] Tạo tests/test_baseline.py: Kiểm thử luồng dữ liệu sạch và xuất file mẫu ra data/sample_baseline.csv.

Ngày 4 - 7: Động cơ Sinh lỗi Bu lông Giả lập (Physics-informed Engine)
[ ] Xây dựng src/fault_injector.py:

[ ] Hàm mô phỏng bu lông lỏng: Bất đối xứng độ cứng, tiêm sóng hài 0.5x, 2x, 3x và xung va đập ngắt quãng.

[ ] Cơ chế Domain Randomization: Ngẫu nhiên hóa biên độ rung (±10%) và độ lệch tần số để chống Overfitting.

[ ] Tạo tests/test_injector.py: Xuất biểu đồ so sánh phổ FFT giữa tín hiệu bình thường và tín hiệu lỗi ra reports/figures/fft_comparison.png.

Ngày 8 - 11: Mô hình Học máy & Đánh giá Thực nghiệm
[ ] Xây dựng src/train_eval.py:

[ ] Phân chia dữ liệu theo TimeSeriesSplit (tuyệt đối không shuffle ngẫu nhiên).

[ ] Huấn luyện mô hình (Isolation Forest / Random Forest) theo 2 kịch bản:

Kịch bản A: Chỉ học trên dữ liệu bình thường (One-Class/Baseline).

Kịch bản B: Học trên tập dữ liệu tăng cường (Normal + Synthetic Fault).

[ ] Tính toán các chỉ số: Precision, Recall, F1-Score, ROC-AUC, False Positive Rate.

[ ] Tự động lưu biểu đồ so sánh (Confusion Matrix, ROC Curve, Phân bố điểm số bất thường) ra thư mục reports/figures/.

Ngày 12 - 15: Đóng gói Báo cáo & Tổng kết
[ ] Thu thập các bảng số liệu thực nghiệm và hình ảnh từ reports/figures/.

[ ] Hoàn thiện bản báo cáo thuyết minh kỹ thuật nộp ban giám khảo vòng sơ khảo.

[ ] Rà soát source code, chuẩn hóa docstrings và thực hiện commit/push toàn bộ lên nhánh main GitHub.

4. CHECKLIST TIẾN ĐỘ THỰC TẾ
[x] Khởi tạo Git repository và Workspace Trust trên VS Code.

[x] Thiết lập file .gitignore, .clinerules và agent.md.

[x] Milestone 1: Baseline Generator & Feature Extraction.

[x] Milestone 2: Physics-informed Fault Injector.

[ ] Milestone 3: Machine Learning Model & Evaluation Figures.

[ ] Milestone 4: Báo cáo Thuyết minh Kỹ thuật.