# Confusion-Aware Semantic Feature Transmission

Mã nguồn mô phỏng cho bài toán truyền đặc trưng ngữ nghĩa thích nghi kênh, có xét cặp lớp dễ nhầm, phục vụ phân loại bệnh lá cà chua.

## Khối chức năng

- M1: MobileNetV3-Small/ResNet-18 trích xuất vector đặc trưng và chia thành nhóm.
- M2: bộ phân loại cục bộ xác định hai lớp cạnh tranh.
- M3: codebook xếp hạng nhóm đặc trưng theo gradient-times-activation của biên logit cặp lớp.
- M4: chính sách cố định ngân sách slot, dùng slot dư để lặp nhóm quan trọng.
- M5: cân bằng kênh, trung bình bản lặp, chèn mask token và phân loại tại phía thu.

## Cơ chế khôi phục

- `latest.pt`, `best.pt`, checkpoint theo epoch và checkpoint theo bước.
- Ghi checkpoint nguyên tử: file tạm được đồng bộ xuống đĩa rồi mới đổi tên.
- Lưu đầy đủ model, optimizer, scheduler, AMP scaler và trạng thái RNG.
- Bắt `SIGINT`, `SIGTERM` và lưu `emergency.pt`.
- `pipeline_state.json` ghi trạng thái từng giai đoạn.
- Mỗi ô mô phỏng `(scheme, budget, channel, SNR, Monte-Carlo seed)` là một JSON độc lập; ô đã hoàn thành được bỏ qua khi chạy lại.

## Chạy nhanh

```bash
cd confusion_aware_semcom_code
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --config configs/default.yaml
```

Sửa `data.root`, `data.manifest_dir` và `project.output_root` trong `configs/default.yaml` trước khi chạy.

Kiểm tra cài đặt bằng dữ liệu giả:

```bash
python scripts/smoke_test.py
```

Hướng dẫn chi tiết nằm trong `guide/huong_dan_mo_phong_confusion_aware_semcom.tex` và bản PDF tương ứng.

## Phạm vi phát hành

Bản này hiện thực đường truyền đặc trưng tương tự và các ablation trực tiếp. JPEG/BPG với mã kênh, DeepJSCC tái tạo ảnh, OFDM/TDL và LDPC chưa nằm trong pipeline thực thi. Xem `TEST_REPORT.md` để biết các kiểm tra đã chạy.
