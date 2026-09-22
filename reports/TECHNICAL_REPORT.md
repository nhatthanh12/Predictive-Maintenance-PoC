# Báo cáo thuyết minh kỹ thuật

## Denso Predictive Maintenance PoC

**Chủ đề:** Phát hiện lỗi cơ khí khi dữ liệu lỗi thực tế khan hiếm bằng dữ liệu tổng hợp có cơ sở vật lý  
**Phạm vi:** Tín hiệu rung động của cụm máy quay, lỗi bu lông lỏng  
**Phiên bản benchmark:** Milestone 4, TimeSeriesSplit 3 fold, `gap=1` block, random seed `42`

## Tóm tắt điều hành

PoC xây dựng một pipeline ba tầng để mô phỏng tín hiệu khỏe, tiêm lỗi bu lông lỏng theo cơ chế vật lý, trích xuất đặc trưng công nghiệp và so sánh hai chiến lược phát hiện:

- **Baseline:** Isolation Forest chỉ học từ dữ liệu bình thường.
- **Augmented:** Random Forest học từ dữ liệu bình thường và dữ liệu lỗi tổng hợp.

Đánh giá được thực hiện trên hai mức độ lỗi: **Severe** và **Incipient**. Dữ liệu được tạo thành các block tín hiệu độc lập, không dùng cửa sổ trượt chồng lấn; các fold được chia theo thời gian với khoảng đệm một block để giảm nguy cơ rò rỉ. Speed drift được lấy trong khoảng **28.5-31.5 Hz**.

Kết quả cho thấy dữ liệu tăng cường đặc biệt có giá trị ở lỗi chớm lỏng. Ở mức Incipient, Baseline chỉ đạt Recall **0.711 ± 0.329** và F1-Score **0.751 ± 0.241**, trong khi Augmented đạt Recall và F1-Score **1.000 ± 0.000** trên benchmark hiện tại. Kết quả này cần được xem là bằng chứng khả thi của PoC, chưa phải giấy phép triển khai sản xuất: cần xác nhận thêm bằng dữ liệu hiện trường và kiểm thử ngoài miền.

## Phần 1. Bối cảnh công nghiệp và cơ sở vật lý

Trong dây chuyền sản xuất, bu lông lỏng làm thay đổi điều kiện liên kết và độ cứng tương đương của cụm máy. Biến thiên độ cứng tạo ra đáp ứng dao động không còn thuần ổn định như trạng thái khỏe: biên độ bị điều chế, năng lượng xuất hiện ở các thành phần phân số và hài bậc cao, đồng thời va đập ngắt quãng có thể xuất hiện khi các bề mặt tiếp xúc mất tiền tải.

Mô hình fault injector mô phỏng các dấu hiệu chính sau:

- **Sóng hài 0.5x:** thành phần dưới hài quanh một nửa tần số quay, đại diện cho tính phi tuyến và bất đối xứng của liên kết lỏng.
- **Sóng hài 2x và 3x:** các thành phần hài bậc cao do bất đối xứng độ cứng, biến dạng tuần hoàn và tương tác cơ học trong cụm quay.
- **Va đập ngắt quãng:** các xung có xác suất xuất hiện, biên độ và suy giảm theo thời gian, phản ánh tiếp xúc không liên tục.
- **Speed drift:** tần số cơ bản được biến thiên trong khoảng **28.5-31.5 Hz**, thay vì cố định tại 30 Hz. Cơ chế này buộc mô hình học dấu hiệu vật lý tương đối thay vì ghi nhớ một bin FFT cố định.

Hai mức độ được kiểm tra:

- **Severe:** biên độ hài và xung đủ lớn, đại diện lỗi đã phát triển.
- **Incipient / Micro-loose bolt:** biên độ thành phần lỗi được giới hạn ở khoảng **5-15% biên độ tín hiệu khỏe**, tiệm cận nhiễu nền và khó phát hiện hơn.

Các tham số trên là mô phỏng có kiểm soát. Chúng không thay thế việc hiệu chỉnh theo loại máy, tải, vị trí cảm biến và phổ nhiễu của dây chuyền cụ thể.

## Phần 2. Kiến trúc hệ thống ba tầng

### Tầng 1: Data Generation

`src/baseline_generator.py` sinh tín hiệu khỏe gồm thành phần cơ bản, hài yếu, điều chế biên độ chậm và nhiễu Gaussian công nghiệp. Mỗi mẫu benchmark được tạo bởi một generator độc lập với seed dẫn xuất từ seed chính. `src/fault_injector.py` tiêm dấu hiệu bu lông lỏng lên một baseline độc lập và hỗ trợ hai severity profile cùng speed drift.

### Tầng 2: Industrial Feature Extraction

`src/features.py` chuyển từng block tín hiệu thành vector đặc trưng nhỏ, có khả năng diễn giải:

- RMS, Kurtosis, Crest Factor và Peak-to-Peak trong miền thời gian.
- Biên độ FFT tại tần số cơ bản, 2x và 3x trong miền tần số.

Các đặc trưng được tính riêng cho từng block. Pipeline hiện không dùng bước chuẩn hóa; do đó không có scaler nào được fit xuyên train/test. Nếu bổ sung chuẩn hóa trong tương lai, scaler phải được fit riêng trên train của từng fold và chỉ dùng để transform test của fold đó.

### Tầng 3: Validation và mô hình

Mỗi hàng dữ liệu tương ứng một signal block hoàn chỉnh, có `block_id` duy nhất và không tạo sliding windows chồng lấn. `TimeSeriesSplit` dùng 3 fold theo thứ tự thời gian với `gap=1`, tạo một block embargo giữa train và test. Pipeline kiểm tra cả giao nhau của index và giao nhau của `block_id`; nếu phát hiện trùng lặp, quá trình đánh giá sẽ dừng.

Hai mô hình được đánh giá trên cùng các fold:

- **Baseline:** Isolation Forest fit trên các mẫu khỏe của train fold; ngưỡng anomaly được đặt theo phân vị 95% của điểm train khỏe.
- **Augmented:** Random Forest fit trên cả hai lớp của train fold; xác suất lớp Fault được dùng để tính các chỉ số trên test fold.

Kết quả được lưu theo từng fold tại [metric_comparison_by_fold.csv](figures/metric_comparison_by_fold.csv) và tổng hợp mean/std tại [metric_comparison_summary.csv](figures/metric_comparison_summary.csv).

## Phần 3. Kết quả định lượng thực nghiệm

Bảng dưới đây được chép từ `metric_comparison_summary.csv`. Giá trị trình bày theo dạng **mean ± std** trên 3 fold.

| Mức lỗi | Mô hình | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate |
|---|---|---:|---:|---:|---:|---:|
| Incipient | Baseline | 0.870 ± 0.116 | 0.711 ± 0.329 | 0.751 ± 0.241 | 0.913 ± 0.089 | 0.111 ± 0.139 |
| Incipient | Augmented | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| Severe | Baseline | 0.906 ± 0.107 | 0.956 ± 0.077 | 0.926 ± 0.064 | 0.941 ± 0.088 | 0.111 ± 0.139 |
| Severe | Augmented | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |

### Phân tích Incipient

Baseline không được nhìn thấy ví dụ lỗi trong quá trình học. Với lỗi chớm lỏng, các thành phần 0.5x, 2x, 3x và xung va đập chỉ nhỉnh hơn nền một khoảng nhỏ; thêm vào đó, speed drift làm năng lượng dịch chuyển khỏi vị trí tần số cố định. Vì vậy, anomaly score của một phần lỗi chớm lỏng vẫn nằm trong vùng giống dữ liệu khỏe, dẫn tới bỏ sót lỗi và làm Recall trung bình giảm xuống **0.711**. Độ lệch chuẩn Recall **0.329** cũng cho thấy tính ổn định giữa các fold còn yếu.

Augmented được học trực tiếp từ các biến thiên lỗi có severity thấp và speed drift. Mô hình vì vậy có cơ hội học ranh giới giữa nhiễu nền và tổ hợp dấu hiệu vật lý, thay vì chỉ dựa vào độ lớn tổng thể. Trên benchmark hiện tại, Recall đạt **1.000**, F1-Score đạt **1.000** và FPR đạt **0.000**. Tuy nhiên, do kết quả vẫn tuyệt đối, cần xem đây là tín hiệu cần kiểm chứng thêm bằng dữ liệu thực và negative controls, không phải bằng chứng rằng lỗi chớm lỏng luôn dễ phát hiện.

### Phân tích Severe

Lỗi Severe tạo ra dấu hiệu có biên độ lớn hơn nên cả hai mô hình đều hoạt động tốt. Baseline đạt Recall **0.956** và F1-Score **0.926**, nhưng FPR trung bình vẫn là **0.111** với độ lệch **0.139**. Augmented đạt kết quả hoàn hảo trên bộ mô phỏng này, cho thấy dữ liệu lỗi có nhãn giúp giảm báo động giả và giữ được độ nhạy. Khoảng cách giữa hai kịch bản nhỏ hơn so với Incipient vì tín hiệu Severe vốn đã dễ phân tách.

## Phần 4. Phân tích trực quan đồ thị

Các hình được sinh ở DPI 300 trong thư mục `reports/figures/`:

- [Confusion Matrix - Incipient - Baseline](figures/incipient_baseline_confusion_matrix.png)
- [Confusion Matrix - Incipient - Augmented](figures/incipient_augmented_confusion_matrix.png)
- [Confusion Matrix - Severe - Baseline](figures/severe_baseline_confusion_matrix.png)
- [Confusion Matrix - Severe - Augmented](figures/severe_augmented_confusion_matrix.png)
- [FFT comparison của tín hiệu khỏe và lỗi](figures/fft_comparison.png)
- [ROC curve comparison trước đó](figures/roc_curve_comparison.png)

Các confusion matrix cho phép kiểm tra trực tiếp trade-off giữa bỏ sót lỗi và báo động giả. Ở Incipient, cần ưu tiên đọc các ô False Negative của Baseline vì đây là biểu hiện rõ nhất của vấn đề độ nhạy. Ở Severe, các ô chẩn đoán đúng chiếm ưu thế, phản ánh mức tách biệt cao hơn của fault signature.

## Phần 5. Tính khả thi và lộ trình Edge AI

### Tính khả thi trên dây chuyền

Pipeline sử dụng NumPy, pandas, SciPy, scikit-learn và matplotlib; vector đặc trưng chỉ có bảy biến chính nên phù hợp với gateway công nghiệp hoặc máy tính biên có tài nguyên hạn chế. Quy trình suy luận có thể đặt sau tầng thu thập rung động, thực hiện tính năng theo block cố định và phát cảnh báo theo cửa sổ thời gian.

Trước khi triển khai, cần thực hiện:

1. Thu thập dữ liệu khỏe theo nhiều tải, tốc độ, ca vận hành và vị trí cảm biến.
2. Hiệu chỉnh noise floor, ngưỡng cảnh báo và khoảng speed drift theo từng thiết bị.
3. Kiểm thử trên máy chưa từng xuất hiện trong tập phát triển để đánh giá khả năng tổng quát.
4. Đánh giá chi phí FPR theo quy trình sản xuất, vì cảnh báo giả có thể gây dừng máy không cần thiết.
5. Thiết lập cơ chế lưu feature, model version, timestamp và quyết định cảnh báo để truy vết.

### Lộ trình Edge AI

- **Bước 1:** Đóng gói feature extraction và model inference thành service Python tối giản, có health check và logging.
- **Bước 2:** Chuyển mô hình sang runtime phù hợp với gateway, giới hạn bộ nhớ và kiểm tra latency trên dữ liệu trực tuyến.
- **Bước 3:** Bổ sung calibration, drift monitoring và cơ chế cập nhật model có phê duyệt.
- **Bước 4:** Shadow deployment trên một dây chuyền, so sánh cảnh báo với kiểm tra bảo trì và dữ liệu hiện trường.
- **Bước 5:** Mở rộng theo từng thiết bị sau khi xác nhận Recall, F1-Score và FPR ngoài miền mô phỏng.

Kết luận của Milestone 4: PoC đã có cấu trúc mã nguồn, kiểm thử, benchmark leakage-aware, bảng số liệu và hình minh họa đủ để làm cơ sở cho hồ sơ kỹ thuật. Bước tiếp theo có giá trị nhất là xác thực độc lập trên dữ liệu rung động thực tế, đặc biệt ở vùng lỗi chớm lỏng.
