# Denso Predictive Maintenance PoC

PoC phát hiện lỗi cơ khí trên motor/cụm máy quay khi dữ liệu lỗi thực tế khan hiếm. Giải pháp sử dụng **Physics-informed Synthetic Data** để mô phỏng lỗi bu lông lỏng, sau đó so sánh mô hình Baseline chỉ học dữ liệu khỏe với mô hình Augmented được huấn luyện trên dữ liệu khỏe và lỗi tổng hợp.

## Kết quả định lượng

Kết quả được lấy từ benchmark 3-fold `TimeSeriesSplit`, `gap=1`, signal block độc lập và speed drift trong khoảng 28.5-31.5 Hz. Giá trị trình bày theo `mean ± std`.

| Mức lỗi | Mô hình | Recall | F1-Score | FPR |
|---|---|---:|---:|---:|
| Incipient | Baseline | 0.711 ± 0.329 | 0.751 ± 0.241 | 0.111 ± 0.139 |
| Incipient | Augmented | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| Severe | Baseline | 0.956 ± 0.077 | 0.926 ± 0.064 | 0.111 ± 0.139 |
| Severe | Augmented | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |

Đây là kết quả trên bộ dữ liệu mô phỏng. Cần xác thực thêm với dữ liệu hiện trường trước khi triển khai sản xuất.

## Quickstart

Yêu cầu Python 3.10 hoặc 3.11.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce Results

Chạy toàn bộ test từ thư mục gốc:

```bash
pytest -v
```

Chạy benchmark Severe/Incipient và xuất CSV cùng confusion matrix 300 DPI:

```bash
python -m src.train_eval
```

Kết quả được ghi vào:

- `reports/figures/metric_comparison_by_fold.csv`
- `reports/figures/metric_comparison_summary.csv`
- `reports/figures/*_baseline_confusion_matrix.png`
- `reports/figures/*_augmented_confusion_matrix.png`

Sinh lại dữ liệu baseline mẫu và hình FFT thông qua test tương ứng:

```bash
pytest -v tests/test_baseline.py tests/test_injector.py
```

## Project Structure

```text
.
├── data/
│   └── sample_baseline.csv
├── reports/
│   ├── TECHNICAL_REPORT.md
│   └── figures/
│       ├── metric_comparison_by_fold.csv
│       ├── metric_comparison_summary.csv
│       └── *_confusion_matrix.png
├── src/
│   ├── baseline_generator.py
│   ├── fault_injector.py
│   ├── features.py
│   └── train_eval.py
├── tests/
│   ├── test_baseline.py
│   ├── test_injector.py
│   └── test_train_eval.py
├── .github/workflows/ci.yml
├── requirements.txt
└── agent.md
```

## Kiến trúc và kiểm soát leakage

- **Data Generation:** tạo tín hiệu khỏe và fault signature 0.5x, 2x, 3x, va đập ngắt quãng.
- **Feature Extraction:** RMS, Kurtosis, Crest Factor, Peak-to-Peak và các biên độ FFT.
- **Validation:** mỗi hàng là một signal block độc lập có `block_id`; không dùng overlapping windows; `TimeSeriesSplit` có `gap=1`.
- **Models:** Isolation Forest cho Baseline và Random Forest cho Augmented.

## Tài liệu kỹ thuật

Xem báo cáo đầy đủ tại [reports/TECHNICAL_REPORT.md](reports/TECHNICAL_REPORT.md).

## Continuous Integration

GitHub Actions chạy tự động trên mọi `push` và `pull_request` vào `main`, với Python 3.10 và 3.11. Workflow cài `requirements.txt` và thực thi `pytest -v`.
