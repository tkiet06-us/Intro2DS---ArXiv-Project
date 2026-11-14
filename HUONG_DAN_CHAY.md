# 🚀 Hướng dẫn chạy Parser M2 trên Dữ liệu thật (từ Google Drive)

> **Thông tin nhóm:** Toàn bộ dữ liệu crawl của **Milestone 1 (M1)** đã được hoàn tất và nén dạng `.tar.gz` đưa lên Google Drive của nhóm tại thư mục `Data/crawl_result/`.
> Tài liệu này hướng dẫn **từng thành viên lấy dữ liệu thật về máy, giải nén đúng chuẩn và chạy Parser Milestone 2 (M2)** bằng Pipeline Runner `src/main.py`.
>
> 📌 **Lưu ý:** Stage M2 chạy **hoàn toàn offline trên máy cục bộ**, không cần mạng, không cần API key và không cần Kaggle snapshot.
> Luôn chạy lệnh **từ thư mục gốc của repo** và **đã kích hoạt môi trường ảo**.

---

## Mục lục

1. [Tóm tắt nhanh (TL;DR)](#1-tóm-tắt-nhanh-tldr)
2. [Chuẩn bị môi trường & Thư viện (làm 1 lần)](#2-chuẩn-bị-môi-trường--thư-viện-làm-1-lần)
3. [Cách lấy dữ liệu thật từ Google Drive](#3-cách-lấy-dữ-liệu-thật-từ-google-drive)
   - [3.1 Cấu trúc file trên Drive nhóm](#31-cấu-trúc-file-trên-drive-nhóm)
   - [3.2 Tải và giải nén file `papers_<name>.tar.gz`](#32-tải-và-giải-nén-file-papers_nametargz)
   - [3.3 Kiểm tra cấu trúc thư mục sau giải nén](#33-kiểm-tra-cấu-trúc-thư-mục-sau-giải-nén)
4. [Dải ID của từng thành viên](#4-dải-id-của-từng-thành-viên)
5. [Hướng dẫn chạy M2 trên dữ liệu thật](#5-hướng-dẫn-chạy-m2-trên-dữ-liệu-thật)
   - [5.1 Chạy thử 5 bài đầu (Kiểm tra nhanh)](#51-chạy-thử-5-bài-đầu-kiểm-tra-nhanh)
   - [5.2 Chạy toàn bộ dải của từng thành viên](#52-chạy-toàn-bộ-dải-của-từng-thành-viên)
   - [5.3 Gom dữ liệu cả nhóm chạy chung (25.000 bài)](#53-gom-dữ-liệu-cả-nhóm-chạy-chung-25000-bài)
6. [Các tùy chọn dòng lệnh nâng cao & Gỡ lỗi](#6-các-tùy-chọn-dòng-lệnh-nâng-cao--gỡ-lỗi)
7. [Kết quả đầu ra & Các trạng thái bài báo](#7-kết-quả-đầu-ra--các-trạng-thái-bài-báo)
   - [7.1 Cấu trúc file kết quả](#71-cấu-trúc-file-kết-quả)
   - [7.2 Ý nghĩa các trạng thái bài báo (`ok`, `empty_record`...)](#72-ý-nghĩa-các-trạng-thái-bài-báo-ok-empty_record)
8. [Cơ chế Checkpoint & Resume (Dừng / Chạy tiếp)](#8-cơ-chế-checkpoint--resume-dừng--chạy-tiếp)
9. [Lỗi thường gặp & Cách khắc phục](#9-lỗi-thường-gặp--cách-khắc-phục)
10. [Checklist hoàn thành M2](#10-checklist-hoàn-thành-m2)

---

## 1. Tóm tắt nhanh (TL;DR)

```powershell
# BƯỚC 1: Kích hoạt môi trường ảo
.venv\Scripts\Activate.ps1
pip install -r src/requirements.txt

# BƯỚC 2: Tải file papers_<name>.tar.gz từ Drive về đặt vào data/raw/ rồi giải nén từ thư mục gốc
# (File tar.gz đã có sẵn đường dẫn data/raw/ nên giải nén trực tiếp từ thư mục gốc của repo)
tar -xzf data\raw\papers_khang.tar.gz

# BƯỚC 3: Chạy thử 5 bài đầu
python -m src.main --stage m2 --name khang --id_start 2411.03799 --id_end 2411.08798 --limit 5

# BƯỚC 4: Chạy toàn bộ dải của bạn
python -m src.main --stage m2 --name khang --id_start 2411.03799 --id_end 2411.08798
```

---

## 2. Chuẩn bị môi trường & Thư viện (làm 1 lần)

```bash
# Tạo môi trường ảo (khuyến nghị Python 3.11 hoặc >= 3.10)
python -m venv .venv

# Kích hoạt môi trường ảo:
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Cài đặt thư viện:
pip install --upgrade pip
pip install -r src/requirements.txt
```

Kiểm tra bộ test parser M2:
```bash
pytest tests/m2_parsing -q
```
*(Toàn bộ 149 test của `tests/m2_parsing` phải xanh `149 passed`).*

---

## 3. Cách lấy dữ liệu thật từ Google Drive

### 3.1 Cấu trúc file trên Drive nhóm

Trên Google Drive của nhóm, dữ liệu M1 được tổ chức tại thư mục `Data/crawl_result/` với cấu trúc sau:

```text
Data/crawl_result/
├── khang/
│   ├── summary_stats_khang.json
│   ├── s2_failures_khang.json
│   ├── profile_m1_khang.json
│   ├── papers_khang.tar.gz          <-- ⭐️ FILE CHỨA TOÀN BỘ DATA CRAWL
│   └── crawl_failures_khang.json
├── nhi/
│   ├── papers_nhi.tar.gz
│   └── ...
├── kiet/
│   ├── papers_kiet.tar.gz
│   └── ...
├── kien/
│   ├── papers_kien.tar.gz
│   └── ...
└── bao/
    ├── papers_bao.tar.gz
    └── ...
```

### 3.2 Tải và giải nén file `papers_<name>.tar.gz`

1. **Tải file:** 
   - Truy cập vào thư mục trên Drive theo tên của bạn (ví dụ: `Data/crawl_result/khang/`).
   - Tải file nén `papers_<name>.tar.gz` (ví dụ: `papers_khang.tar.gz`) về máy tính.

2. **Đặt file vào repo:**
   - Chép file vừa tải vào thư mục: `data/raw/` của dự án.
   - Đường dẫn file: `data/raw/papers_<name>.tar.gz`.

3. **Giải nén đúng cách:**
   > ⚠️ **ĐẶC BIỆT LƯU Ý:** File `papers_<name>.tar.gz` khi nén ở M1 **đã chứa sẵn đường dẫn `data/raw/papers_<name>/` bên trong archive**.
   > Vì vậy, bạn chạy lệnh giải nén **ngay tại thư mục gốc của repo** mà **KHÔNG thêm cờ `-C data/raw/`** (nếu thêm `-C data/raw/` sẽ bị tạo thành `data/raw/data/raw/papers_<name>` dẫn đến lỗi không tìm thấy thư mục).

   - **Trên Windows (PowerShell hoặc cmd):**
     ```powershell
     tar -xzf data\raw\papers_<name>.tar.gz
     ```
     *(Ví dụ cho Khang: `tar -xzf data\raw\papers_khang.tar.gz`)*

   - **Trên Linux / macOS / Git Bash:**
     ```bash
     tar -xzf data/raw/papers_<name>.tar.gz
     ```

   - **Nếu bạn đã lỡ giải nén bị lồng cấp thành `data/raw/data/raw/papers_<name>`:**
     Sửa nhanh bằng lệnh sau (PowerShell):
     ```powershell
     Move-Item -Path "data\raw\data\raw\papers_<name>" -Destination "data\raw\papers_<name>"
     Remove-Item -Path "data\raw\data" -Recurse -Force
     ```

### 3.3 Kiểm tra cấu trúc thư mục sau giải nén

Sau khi giải nén, thư mục dữ liệu **bắt buộc phải nằm đúng vị trí sau** để runner nhận diện được:

```text
data/raw/papers_<name>/
  ├── 2411-03799/
  │     ├── v1/
  │     │    ├── main.tex (hoặc paper.tex, intro.tex...)
  │     │    └── refs.bib (nếu có)
  │     ├── v2/ (nếu bài có nhiều version)
  │     ├── metadata.json
  │     └── references.json
  ├── 2411-03800/
  └── ...
```

> ⚠️ **Lưu ý kiểm tra:**
> - Tên thư mục phải là `papers_<name>` (ví dụ: `data/raw/papers_khang/`).
> - Tránh trường hợp bị giải nén lồng 2 cấp: `data/raw/papers_khang/papers_khang/...`. Nếu bị lồng cấp, hãy di chuyển thư mục con ra ngoài.
> - Thư mục `data/` đã được thêm vào `.gitignore` nên Git sẽ không track dữ liệu nặng này.

---

## 4. Dải ID của từng thành viên

Định dạng ID: `YYMM.NNNNN` (5 chữ số sau dấu chấm, **có số 0 đệm**).
- Đúng: `2506.00788`, `2410.13025`.
- Sai: `2506.788`, `2506.7880`, `2506-00788`.

| Thành viên | Dải ID | `--name` tương ứng | Số lượng bài | Ghi chú |
|---|---|---|---|---|
| **Tôn Thất Kiên** | `2506.00788` → `2506.05787` | `kien` | 5.000 bài | Tháng 06/2025 |
| **Phạm Gia Bảo** | `2503.16914` → `2503.21913` | `bao` | 5.000 bài | Tháng 03/2025 |
| **Trần Ngô Uyên Nhi** | `2502.18235` → `2503.01913` | `nhi` | 5.000 bài | ⚠️ Dải đi qua 2 tháng (Tháng 2 & 3/2025) |
| **Đinh Công Khang** | `2411.03799` → `2411.08798` | `khang` | 5.000 bài | Tháng 11/2024 |
| **Trần Tuấn Kiệt** | `2410.13025` → `2410.18024` | `kiet` | 5.000 bài | Tháng 10/2024 |

> ⚠️ **Lưu ý riêng cho Nhi:** Dải của Nhi vắt qua 2 tháng (`2502` và `2503`). Runner đã tích hợp sẵn hàm so sánh dải theo khóa `(tháng, số)`, bạn chỉ cần nhập đúng `2502.18235` và `2503.01913`, hệ thống sẽ tự động lọc đúng thứ tự.

---

## 5. Hướng dẫn chạy M2 trên dữ liệu thật

### 5.1 Chạy thử 5 bài đầu (Kiểm tra nhanh)

Sau khi giải nén, hãy chạy thử 5 bài đầu với cờ `--limit 5` để đảm bảo thư mục đọc đúng và parser hoạt động trơn tru:

```bash
# Ví dụ chạy thử cho Khang:
python -m src.main --stage m2 --name khang --id_start 2411.03799 --id_end 2411.08798 --limit 5
```
- Nếu hiển thị log: `[1/5] ... -> ok`, `[2/5] ... -> ok`, kết thúc với exit code `0` thì dữ liệu và môi trường hoàn toàn chính xác.

### 5.2 Chạy toàn bộ dải của từng thành viên

Chạy lệnh parse cho toàn bộ 5.000 bài của bạn (bỏ cờ `--limit`):

```bash
# 1. Tôn Thất Kiên
python -m src.main --stage m2 --name kien  --id_start 2506.00788 --id_end 2506.05787

# 2. Phạm Gia Bảo
python -m src.main --stage m2 --name bao   --id_start 2503.16914 --id_end 2503.21913

# 3. Trần Ngô Uyên Nhi (dải qua 2 tháng)
python -m src.main --stage m2 --name nhi   --id_start 2502.18235 --id_end 2503.01913

# 4. Đinh Công Khang
python -m src.main --stage m2 --name khang --id_start 2411.03799 --id_end 2411.08798

# 5. Trần Tuấn Kiệt
python -m src.main --stage m2 --name kiet  --id_start 2410.13025 --id_end 2410.18024
```

### 5.3 Gom dữ liệu cả nhóm chạy chung (25.000 bài)

Nếu một máy trạm trong nhóm tải toàn bộ file nén của 5 thành viên về, giải nén và gom tất cả các thư mục bài báo vào chung một thư mục `data/raw/papers_all/`:

```bash
python -m src.main --stage m2 --name all --id_start 2410.13025 --id_end 2506.05787
```

---

## 6. Các tùy chọn dòng lệnh nâng cao & Gỡ lỗi

### 6.1 Giới hạn số bài (`--limit N`)
Dùng khi bạn muốn chạy thử nghiệm một số lượng bài nhất định (ví dụ 100 bài) để đo đạc hiệu năng:
```bash
python -m src.main --stage m2 --name khang --id_start 2411.03799 --id_end 2411.08798 --limit 100
```

### 6.2 Gỡ lỗi chi tiết trực tiếp trên main process (`--no_worker`)
Mặc định M2 xử lý mỗi bài trong một tiến trình worker con độc lập để cô lập lỗi và tự ngắt nếu bài chạy quá thời gian (`m2.paper_timeout_s = 120s`). Worker chết giữa chừng (OOM/crash) được dựng lại cho bài kế tiếp, không làm hỏng cả dải.
Khi muốn debug một bài bị lỗi bất thường và xem **toàn bộ stack trace / traceback** chi tiết trên màn hình console, hãy thêm cờ `--no_worker`:
```bash
python -m src.main --stage m2 --name khang --id_start 2411.03799 --id_end 2411.08798 --no_worker
```

### 6.3 Chạy lại từ đầu (`--no_resume`)
Bỏ qua checkpoint cũ và parse lại từ bài đầu tiên trong dải:
```bash
python -m src.main --stage m2 --name khang --id_start 2411.03799 --id_end 2411.08798 --no_resume
```

> Với M1, `--no_resume` chỉ chạy lại crawl/S2; nếu đã có `data/raw/discovery_<name>.jsonl` thì vẫn dùng lại discovery và **không cần** snapshot Kaggle.

### 6.4 Chỉ chạy lại bài lỗi (`--retry_failed`)
Mặc định resume bỏ qua cả bài `failed` để số đo benchmark ổn định. Khi đã sửa nguyên nhân (hoặc muốn thử lại do lỗi mạng tạm thời), chỉ cần thêm `--retry_failed` thay vì xóa cả checkpoint:
```bash
python -m src.main --stage m2 --name khang --id_start 2411.03799 --id_end 2411.08798 --retry_failed
```

---

## 7. Kết quả đầu ra & Các trạng thái bài báo

### 7.1 Cấu trúc file kết quả

Sau khi M2 hoàn tất, toàn bộ sản phẩm được xuất ra theo cấu trúc sau:

| Sản phẩm | Đường dẫn | Ý nghĩa |
|---|---|---|
| **Cây phân cấp** | `data/processed/papers_<name>/<id>/hierarchy.json` | Cây phân cấp các phần tử: Heading, Section, Paragraph, Equation... |
| **Trích dẫn BibTeX** | `data/processed/papers_<name>/<id>/refs.bib` | File BibTeX chứa tài liệu tham khảo đã khử trùng lặp và làm sạch |
| **File Checkpoint** | `reports/checkpoint_m2_<name>.jsonl` | Nhật ký ghi nhận trạng thái của từng bài |
| **Báo cáo tổng kết** | `reports/summary_stats_m2_<name>.json` | Thống kê số bài `ok`, `empty_record`, thời gian parse trung bình... |

### 7.2 Ý nghĩa các trạng thái bài báo (`ok`, `empty_record`...)

Trong file báo cáo `reports/summary_stats_m2_<name>.json`, các bài báo được phân loại:

- **`ok`**: Parse thành công hoàn toàn, đã tạo file `hierarchy.json` và `refs.bib`.
- **`empty_record`**: Bài báo trên arXiv chỉ có file PDF hoặc file ảnh, **hoàn toàn không có file nguồn `.tex`**.
  > 📌 **Lưu ý:** Trạng thái này là **hoàn toàn bình thường theo thiết kế dữ liệu arXiv** (tác giả nộp PDF trực tiếp), parser ghi nhận hợp lệ chứ không phải do lỗi code!
- **`skipped_large`**: File `.tex` vượt quá ngưỡng dung lượng quy định (mặc định > 20MB, tức `m2.max_tex_bytes_skip` trong `configs/config.yaml`), được bỏ qua để bảo vệ RAM.
- **`timeout`**: Bài báo parse quá thời gian cho phép (mặc định 120 giây).
- **`failed`**: Bài báo gặp lỗi cú pháp LaTeX đặc thù không thể phục hồi (được log lại ngắn gọn để rà soát).

Ngoài các trạng thái trên, báo cáo còn có các chỉ số cần ghi lại khi so sánh giữa các máy:

- **`n_timeout`**: số bài timeout — phụ thuộc tốc độ máy nên mỗi thành viên cần nêu con số này trong báo cáo cá nhân.
- **`n_partial`**: số bài `ok` nhưng còn version lỗi/chưa xử lý (vẫn có output, chỉ thiếu một phần).
- **`n_truncated`**: số bài bị cắt cây do vượt `m2.max_nodes`.
- **`n_papers_with_tree`**: số bài dựng được cây; `avg_versions_per_paper` và `avg_tree_depth` chỉ chia cho số này (không bị kéo thấp bởi bài `empty_record`/`failed`).

Bài `failed`/`empty_record`/`skipped_large` **không** tạo `hierarchy.json` rỗng; nếu lần chạy trước đã có output thì output cũ được dọn đi để không nhìn nhầm là hợp lệ.

---

## 8. Cơ chế Checkpoint & Resume (Dừng / Chạy tiếp)

- **Lưu liên tục:** Mỗi khi một bài xử lý xong, runner ghi ngay kết quả vào `reports/checkpoint_m2_<name>.jsonl`.
- **An toàn khi ngắt quãng:** Nếu bạn bấm `Ctrl+C`, tắt máy, hoặc bị ngắt điện giữa chừng:
  - Tất cả các bài đã parse trước đó đều được bảo lưu nguyên vẹn.
  - Khi mở lại, bạn chỉ cần gõ lại **đúng câu lệnh ban đầu**: M2 sẽ tự động đọc checkpoint, **bỏ qua ngay lập tức tất cả các bài đã xong** và tiếp tục ngay từ bài còn lại!
- **Mã thoát (Exit Code) của `--stage m2`:**
  - `0`: Hoàn thành tốt toàn bộ dải bài.
  - `1`: Lỗi cấu hình, tham số hoặc thiếu thư mục dữ liệu đầu vào.
  - `130`: Người dùng nhấn `Ctrl+C` dừng tiến trình.
- **Riêng `--stage m1`:** runner trả **nguyên mã của `run_m1`** để bảng xử lý sự cố P3-02a còn dùng được: `1` còn lỗi tạm thời (chạy lại để thử), `2` sai tham số/dữ liệu, `3` bị arXiv chặn (403), `4` quá nhiều lỗi liên tiếp, `5` đầy đĩa, `6` discovery thất bại. Với `--stage all`, runner gộp về `0`/`1`/`130`.

---

## 9. Lỗi thường gặp & Cách khắc phục

| Thông báo / Triệu chứng | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `LOI M2: Khong thay thu muc bai M1: data/raw/papers_<name>` | Giải nén bị lồng cấp thành `data/raw/data/raw/papers_<name>` hoặc chưa giải nén | Kiểm tra thư mục `data/raw/`. Nếu thấy `data/raw/data/raw/papers_<name>` thì chạy: `Move-Item -Path "data\raw\data\raw\papers_<name>" -Destination "data\raw\papers_<name>"`, sau đó xóa folder `data/raw/data`. |
| Thư mục bị lồng cấp `papers_khang/papers_khang/` | Do công cụ giải nén tự tạo thêm 1 cấp thư mục | Di chuyển các thư mục bài báo con ra ngoài sao cho đúng dạng `data/raw/papers_khang/<id>/`. |
| `LOI M2: Khong co bai nao trong dai ... tai ...` | Thư mục có bài nhưng không khớp với `--id_start` và `--id_end` | Kiểm tra lại dải ID trong lệnh xem có khớp với các thư mục bài đã tải về hay không. |
| `LOI: arXiv ID không hợp lệ: '...'` | Nhập ID thiếu số 0 | Đệm đủ 5 chữ số sau dấu chấm, ví dụ: `2506.00788` (không phải `2506.788`). |
| `LOI: --id_start (...) phai <= --id_end (...)` | Đảo ngược ID bắt đầu và kết thúc | Nhập `id_start <= id_end`. Với dải qua tháng của Nhi (`2502.18235` đến `2503.01913`) runner đã tự động hỗ trợ. |
| Muốn xem toàn bộ traceback của một bài | Worker cô lập lỗi để bảo vệ pipeline | Thêm `--no_worker` vào lệnh để chạy trên luồng chính và in đầy đủ traceback. |

---

## 10. Checklist hoàn thành M2

- [ ] Đã kích hoạt môi trường ảo `.venv` và cài `pip install -r src/requirements.txt`.
- [ ] Chạy `pytest tests/m2_parsing -q` đạt `149 passed`.
- [ ] Tải file `papers_<name>.tar.gz` từ thư mục `Data/crawl_result/<tên_bạn>/` trên Google Drive.
- [ ] Đặt file vào `data/raw/` và giải nén thành công ra `data/raw/papers_<name>/`.
- [ ] Chạy thử 5 bài đầu (`--limit 5`) kiểm tra exit code `0`.
- [ ] Chạy lệnh M2 cho toàn bộ dải ID của mình (Mục 5.2).
- [ ] Kiểm tra kết quả tại `data/processed/papers_<name>/` (có đủ `hierarchy.json` và `refs.bib` cho các bài `ok`).
- [ ] Kiểm tra báo cáo tổng kết tại `reports/summary_stats_m2_<name>.json` và ghi lại `n_timeout` (cùng `n_partial`, `n_truncated`) để so sánh giữa các máy.
- [ ] Nén hoặc đồng bộ thư mục `data/processed/papers_<name>/` và file `summary_stats_m2_<name>.json` lên Drive nhóm để chuẩn bị cho Milestone M3!
