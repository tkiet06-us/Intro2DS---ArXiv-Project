# 📚 arXiv Pipeline: Crawl – Parse – Match

> **Đồ án Khoa học Dữ liệu (HCMUS)**: Xây dựng hệ thống end-to-end tự động tải bài báo khoa học từ arXiv, đọc hiểu cấu trúc của từng bài, rồi dùng Machine Learning để tìm xem mỗi tài liệu tham khảo trong bài thực chất là bài báo nào trên arXiv.
>
> 🔗 Trang môn học: <https://hcmus-cs-ta.github.io/lab-book/intro-ds/>         
> 🔗 Drive đồ án: <https://drive.google.com/drive/u/0/folders/1ePIykL0AaKSXwHjYAjUykROtzT1lmxoP>

---

## 👋 Giới thiệu

### Dự án này làm gì?

Tưởng tượng arXiv là một thư viện chứa hàng triệu bài báo khoa học. Mỗi bài có phần mã nguồn và cuối bài có danh sách **tài liệu tham khảo**.

Dự án của nhóm làm 3 việc nối tiếp nhau:

| Bước | Tên | Ví dụ đời thường | Kết quả |
|---|---|---|---|
| **M1** | **Crawling** (thu thập) | Đi thư viện *photocopy* 25.000 cuốn sách về nhà | Thư mục chứa mã nguồn + metadata từng bài |
| **M2** | **Parsing** (phân tích cấu trúc) | *Đọc* từng cuốn và lập **mục lục dạng cây**: Chương → Mục → Câu → Công thức | File `hierarchy.json` (cây phân cấp) và `refs.bib` |
| **M3** | **Reference Matching** (ghép tài liệu tham khảo) | Thấy dòng "Smith et al., 2019, Deep Learning…" và **tìm đúng cuốn sách đó** trong thư viện | Bảng dự đoán top-5 bài khớp nhất cho mỗi tham khảo |

Cuối cùng, toàn bộ dữ liệu được đóng gói thành **6 bảng Parquet** và đăng lên **Hugging Face** để người khác dùng.

### Bài toán

Mỗi tham khảo trong bài báo thường là một chuỗi chữ lộn xộn, sai khác nhiều kiểu: viết tắt tên tác giả, thiếu năm, sai chính tả. Máy tính không thể chỉ so sánh chuỗi bằng `==`. Vì vậy ta cần **Machine Learning** để chấm điểm mức độ giống nhau và **xếp hạng** các ứng viên.

### Mục tiêu

- Thu thập dữ liệu có đạo đức (tôn trọng giới hạn truy cập của máy chủ)
- Xử lý văn bản bán cấu trúc (LaTeX) bằng regex và xây cây dữ liệu
- Làm sạch và khử trùng lặp dữ liệu (deduplication)
- Feature engineering và mô hình xếp hạng (ranking), đánh giá bằng MRR
- Làm việc nhóm chuẩn: Git, Pull Request, Code Review, CI

> 💡 **Gặp từ lạ?** Xem [Bảng thuật ngữ](#-17-bảng-thuật-ngữ-glossary) ở cuối README.

---

## 📑 Mục lục

1. [Kiến trúc hệ thống & Pipeline](#-1-kiến-trúc-hệ-thống--pipeline)
2. [Tech Stack](#-2-tech-stack)
3. [Cấu trúc thư mục](#-3-cấu-trúc-thư-mục)
4. [Thành viên & Phân công](#-4-thành-viên--phân-công)
5. [Task breakdown & Timeline](#-5-task-breakdown--timeline)
6. [Quy trình làm việc & Git flow](#-6-quy-trình-làm-việc--git-flow)
7. [API Reference](#-7-api-reference)
8. [Biến môi trường](#-8-biến-môi-trường)
9. [Cài đặt (venv / conda)](#-9-cài-đặt-venv--conda)
10. [Chạy chương trình](#-10-chạy-chương-trình)
11. [Testing](#-11-testing)
12. [Quản lý dữ liệu](#-12-quản-lý-dữ-liệu)
13. [Đặc tả dữ liệu đầu ra](#-13-đặc-tả-dữ-liệu-đầu-ra)
14. [Troubleshooting](#-14-troubleshooting)
15. [Quy chế & Liêm chính học thuật](#-15-quy-chế--liêm-chính-học-thuật)
16. [Lưu ý quan trọng](#-16-lưu-ý-quan-trọng)
17. [Bảng thuật ngữ (Glossary)](#-17-bảng-thuật-ngữ-glossary)
18. [Trích dẫn & Tài liệu tham khảo](#-18-trích-dẫn--tài-liệu-tham-khảo)

---

## 🏗 1. Kiến trúc hệ thống & Pipeline

### 1.1 Sơ đồ tổng quan

```mermaid
flowchart TD
    subgraph M1["M1 - Discovery và Crawling"]
        A["Danh sách <br> arXiv ID"] --> B["Tải source<br/>tar.gz hoặc gz"]
        B --> C["Giải nén,<br/>xóa ảnh,<br/>giữ khối figure"]
        C --> D["metadata.json"]
        A --> E["Semantic <br> Scholar API"]
        E --> F["references.json"]
    end
    subgraph M2["M2 - Hierarchical Parsing"]
        C --> G["Multi-file<br/>Resolver"]
        G --> H["LaTeX Cleanup<br/>và Math<br/>Normalization"]
        H --> I["hierarchy.json"]
        G --> J["BibTeX Extractor<br/>refs.bib"]
        H --> K["Dedup<br/>giữa các version"]
        J --> K
    end
    subgraph M3["M3 - Reference Matching"]
        I --> L["Gán nhãn<br/>tay và tự động"]
        K --> L
        F --> L
        L --> FE["Feature <br> Engineering"]
        FE --> RM["Ranking Model"]
        RM --> PR["predictions<br/>.parquet<br/>top-5 và MRR"]
    end
    PR --> HF[("6 bảng Parquet<br/>Hugging Face")]
    K --> HF
    I --> HF
```

### 1.2 Nguyên tắc thiết kế

**① Interface-Driven Development (IDD)**: Nhóm chốt **schema dữ liệu** (`src/schemas.py`) ngay từ Tuần 1 và dùng `tests/make_mock.py` sinh **dữ liệu giả** nhỏ nhẹ. Nhờ đó người làm M2 (parser) và M3 (ML) có thể code song song ngay, không phải ngồi chờ M1 crawl xong.

```
Không có IDD:  M1 xong ──► M2 mới bắt đầu ──► M3 mới bắt đầu     (nối đuôi, chậm)
Có IDD:        M1 ──────────────────►
               M2 (dùng mock) ──────►      cùng lúc
               M3 (dùng mock) ──────►
```

**② Repo không chứa dữ liệu thật.** Mã nguồn nằm trên GitHub, còn 25.000 bài thật nằm trên Google Drive chung hoặc Hugging Face Private (xem [mục 12](#-12-quản-lý-dữ-liệu)).

**③ Báo cáo Jupyter Book ở repo riêng** (`book-repo/`). Pipeline chỉ xuất file thống kê nhẹ `summary_stats.json` để commit sang đó; Notebook trong Book đọc file này và vẽ biểu đồ khi build trên GitHub Actions.

### 1.3 Luồng dữ liệu chi tiết

**Step 1 – M1: Discovery & Crawling**
1. Discovery: so sánh arXiv API (batch ID), OAI-PMH và Kaggle dataset; chọn phương pháp và đo tổng thời gian.
2. Tải **mọi version** (v1, v2, …) qua `export.arxiv.org/src/{id}v{n}`.
3. Nhận diện `.tar.gz` và `.gz` đơn (chứa `.tex`); **xóa ảnh nhưng giữ lệnh `\begin{figure}`**; ghi log `size_before` / `size_after`.
4. Bài chỉ có PDF/Word → giữ **record rỗng** (thư mục version rỗng) và ghi log lỗi.
5. Gọi Semantic Scholar lấy danh sách tham khảo, lưu trữ cục bộ đầy đủ và chỉ **lọc theo arXiv ID khi xuất bảng `references` Parquet**.
6. Hỗ trợ CLI độc lập `--id_start --id_end` và **Checkpoint/Resume** (chạy lại sẽ tiếp tục từ chỗ dừng).

**Step 2 – M2: Hierarchical Parsing**
1. **Multi-file Resolver**: tìm file `.tex` chính, lần theo `\input{}` / `\include{}` để ghép toàn bộ nội dung.
2. **LaTeX Cleanup & Math Normalization** (xóa `\centering`, `[htpb]`…; chuẩn hóa toán): chạy **TRƯỚC** khi dedup.
3. **Hierarchy Tree**: dựng cây Chapter → Section → Subsection → … với thuộc tính `kind` (`document`, `heading`, `itemize`, `item`, `sentence`, `formula`, `figure`).
4. **BibTeX Extractor**: trích `\bibitem` thành `refs.bib`, chuẩn hóa trường.
5. **Dedup 2 cấp**: gộp BibTeX trùng (Field Union) + so khớp nội dung giữa các version.

**Step 3 – M3: Reference Matching**
1. Gán nhãn (thủ công + tự động) → chia Train / Valid / Test.
2. Feature Engineering (Levenshtein, Jaccard n-gram, TF-IDF cosine).
3. Train Ranking Model → xuất top-5 ứng viên → đo **MRR** trên tập Test.
4. Xuất 6 bảng Parquet **tổng** (25.000 bài) → dùng script filter tách thành 5 bộ theo dải ID cá nhân → mỗi thành viên tự upload lên Hugging Face.

---

## 🧰 2. Tech Stack

| Nhóm | Công cụ | Dùng để |
|---|---|---|
| Ngôn ngữ | **Python 3.11** | Toàn bộ pipeline |
| Crawling | `arxiv`, `sickle` (OAI-PMH), `requests` | Discovery & tải source |
| Dữ liệu | `pandas`, `numpy`, `pyarrow` | Xử lý bảng, ghi Parquet |
| Schema | `pydantic` v2 | Khai báo & kiểm tra kiểu dữ liệu |
| Parsing | `re` (regex), `bibtexparser` | Phân tích LaTeX/BibTeX |
| ML | `scikit-learn`, `rapidfuzz` | TF-IDF, Levenshtein, Ranking |
| Giám sát | `psutil` | Đo RAM, đĩa, thời gian |
| Hosting | `huggingface_hub` | Upload dataset |
| Báo cáo | `jupyter-book` + GitHub Pages | Báo cáo M1, M2 |
| Chất lượng | `pytest`, `pytest-cov`, `ruff` | Test & kiểm tra style |
| CI/CD | GitHub Actions | Tự chạy test mỗi PR |
| Cộng tác | GitHub, Discord, Google Drive | Code, họp, lưu data |
| Benchmark | Google Colab (CPU-only) | Đo hiệu năng chuẩn |

---
## 🗂 3. Cấu trúc thư mục

```
Intro2DS---ArXiv-Project/
├── README.md                    # Tài liệu này
├── CONTRIBUTING.md              # Quy tắc đóng góp
├── requirements.txt             # Thư viện Python
├── pytest.ini                   # Cấu hình pytest
├── .env.example                 # Mẫu biến môi trường (copy thành .env)
├── .gitignore                   # Chặn data thật, secrets, cache
│
├── configs/
│   └── config.yaml              # Tham số: delay, retry, đường dẫn, top_k
│
├── src/                         # ★ Mã nguồn chính
│   ├── main.py                  # Pipeline Runner (CLI)                    [P3-01]
│   ├── schemas.py               # Pydantic schema + schema Parquet 6 bảng  [P1-02]
│   ├── common/
│   │   ├── ids.py               # Kiểm tra arXiv ID, đổi ID <-> tên folder [P1-02]
│   │   └── utils_io.py          # Checkpoint, đọc/ghi JSON, logging
│   ├── m1_crawling/
│   │   ├── crawler.py           # Tải source arXiv                         [P2-01]
│   │   ├── s2_api.py            # Semantic Scholar → references.json       [P2-02]
│   │   └── profiler.py          # Đo RAM / đĩa / thời gian                 [P2-02]
│   ├── m2_parsing/
│   │   ├── latex_parser.py      # Multi-file Resolver                      [P2-03]
│   │   ├── tree_builder.py      # Dựng hierarchy.json                      [P2-03]
│   │   ├── latex_cleanup.py     # Dọn LaTeX, chuẩn hóa toán                [P2-05]
│   │   ├── bibtex_extractor.py  # \bibitem → refs.bib                      [P2-04]
│   │   └── dedup.py             # Dedup BibTeX + full-text                 [P2-04]
│   └── m3_matching/
│       ├── features.py          # Tiền xử lý & đặc trưng                   [P2-06]
│       ├── matching_model.py    # Mô hình ranking                          [P2-06]
│       ├── train.py             # Huấn luyện + xuất predictions            [P4-01]
│       ├── evaluate.py          # Tính MRR                                 [P4-01]
│       └── export_parquet.py    # Xuất 6 bảng + upload HF                  [P4-02]
│
├── tests/                       # Script sinh mock, mock data và unit test
│   ├── make_mock.py             # Sinh mock data (dạng làm việc + Parquet) [P1-02]
│   ├── fetch_real_samples.py    # Tải 2-3 bài thật mẫu từ arXiv            [P1-02]
│   ├── mock_data/               # Mock nhỏ sinh bởi make_mock.py (ĐƯỢC commit)
│   │   ├── <yymm-id>/           # Mỗi bài một folder: v1/, v2/, metadata.json,
│   │   │                        #   references.json, hierarchy.json, refs.bib
│   │   ├── parquet/             # 6 bảng Parquet: parquet/<table>/part-000.parquet
│   │   └── real/                # Bài thật do fetch_real_samples.py tải (KHÔNG commit)
│   ├── common/                  # Chứa các file unit test (test_schemas, test_utils_io, test_make_mock...)
│   ├── m1_crawling/             # Unit test cho module M1
│   ├── m2_parsing/              # Unit test cho module M2
│   └── m3_matching/             # Unit test cho module M3
│
├── docs/                        # Tài liệu thiết kế (spec)
│   ├── discovery_report.md      # So sánh phương pháp discovery            [P1-00]
│   ├── latex_spec.md            # Quy tắc dựng cây                         [P1-04]
│   └── ml_design.md             # Thiết kế ML & predictions                [P1-05]
│
├── data/                        # ⚠ Nội dung bị gitignore (chỉ giữ .gitkeep)
│   ├── raw/  processed/  labels/  sources/
│
├── reports/                     # Benchmark JSON, poster A2
├── scripts/setup_dev.sh         # Cài môi trường một lệnh (Linux/macOS)
└── .github/
    ├── pull_request_template.md # Mẫu mô tả PR
    └── workflows/ci.yml         # CI: ruff + pytest
```

> 📌 **Thư mục theo từng bài báo** (đầu ra của M1/M2, dự kiến): `data/processed/<yymm-id>/` chứa `metadata.json`, `references.json`, `hierarchy.json`, `refs.bib`… (tên `yymm-id`, ví dụ `2506-00788`; bài cũ như `hep-th/9901001` được giữ theo quy ước của đề).

> 🔖 `.gitkeep` là file rỗng giúp Git giữ lại thư mục trống (Git không theo dõi thư mục rỗng).

---

## 👥 4. Thành viên & Phân công

Mỗi người **crawl 5.000 bài** trên dải ID cá nhân của mình (tổng **25.000 bài**) rồi gom kết quả lên Google Drive chung. Sau đó nhóm **parse chung toàn bộ 25.000 bài** tập trung (1-2 máy). Ở bước cuối cùng của M3, dữ liệu tổng được trích xuất thành 5 phần tương ứng với dải ID cá nhân để từng thành viên tự đưa lên Hugging Face. Tất cả 5 thành viên đều tham gia gán nhãn thủ công (≥1 bài/bạn) để hiểu trọn pipeline khi bảo vệ.

| Thành viên | Dải ID cá nhân | Vai trò Lead | Nhiệm vụ chính |
|---|---|---|---|
| **Tôn Thất Kiên** | 2025-06 (788 → 5787) | Discovery & Crawling (M1) | Khảo sát Discovery (P1-00), Crawler (P2-01), LaTeX Cleanup (P2-05), Báo cáo M1, Poster |
| **Phạm Gia Bảo** | 2025-03 (16914 → 21913) | External API & Storage (M1, M3) | Schema & Mock (P1-02), S2 API & Profiler (P2-02), Benchmark M1, Feature Eng, HF Exporter (P4-02) |
| **Trần Ngô Uyên Nhi** | 2025-02/03 (18235 → 1913) | Hierarchical Parser (M2) | Khung Jupyter Book (P1-03), Resolver & Tree Builder (P2-03), Báo cáo M2, Benchmark M2, Reviewer parse chung (P3-03a) |
| **Đinh Công Khang** | 2024-11 (3799 → 8798) | Dedup & BibTeX (M2) | Spec LaTeX (P1-04), BibTeX & Dedup (P2-04), Runner `main.py` (P3-01), Chạy parse chung (P3-03a) |
| **Trần Tuấn Kiệt** | 2024-10 (13025 → 18024) | Machine Learning (M3) | Repo GitHub (P1-01), Spec ML (P1-05), Ranking Model, MRR, README spec, Poster |

> ⚠️ **Nhi**: dải ID `18235 → 1913` đi qua hai tháng (2025-02 và 2025-03). Cả nhóm cần thống nhất rõ cách xử lý đoạn "quay vòng" này trước khi chạy crawl để không bị trùng/sót ID.

### Cơ chế Review chéo

Mỗi task có **1 Owner + 2 Reviewers** (xem bảng task ở mục 5). Owner viết code, 2 Reviewer đọc và Approve PR trước khi merge.

---

## 📋 5. Task breakdown & Timeline

### 5.1 Các mốc nộp

| Milestone | Nội dung | Hạn Moodle | **Hạn nội bộ (mục tiêu)** | Dư an toàn |
|---|---|---|---|---|
| **M1** | Discovery & Crawling | 06/11 | **02/11** | 4 ngày |
| **M2** | Hierarchical Parsing | 20/11 | **12/11** | 8 ngày |
| **M3** | Reference Matching & Poster | 27/11 | **22/11** | 5 ngày |

> ⏰ Hạn nội bộ sớm hơn hạn Moodle để có đệm xử lý rủi ro (lỗi mạng, merge conflict, in poster).

### 5.2 Timeline tổng quan

```
Tuần  12/10 ─ 15/10 : Giai đoạn 1 – Khảo sát, Repo, Schema, Spec
      14/10 ─ 26/10 : Giai đoạn 2 – Viết module (crawler, parser, ML)
      26/10 ─ 28/10 : ĐỢT 1 – Mỗi người crawl dải cá nhân (5.000 bài), gom lên Drive
      29/10 ─ 02/11 : Benchmark Colab M1 + Báo cáo M1  ──► NỘP M1
      30/10 ─ 06/11 : ĐỢT 2 – Parse chung 25.000 bài
      07/11 ─ 12/11 : Benchmark Colab M2 + Báo cáo M2  ──► NỘP M2
      31/10 ─ 08/11 : Gán nhãn dữ liệu
      08/11 ─ 14/11 : Train model, xuất predictions
      15/11 ─ 18/11 : Xuất 6 bảng tổng, tách 5 bộ & upload Hugging Face
      15/11 ─ 20/11 : Thiết kế Poster A2
      20/11 ─ 22/11 : Kiểm thử cuối + NỘP M3
```

### 5.3 Danh sách 24 task

**Trạng thái:** 🔴 Chưa sẵn sàng · 🟡 Đang thực hiện · 🟢 Hoàn thành *(cập nhật trong bảng quản lý tiến độ của nhóm)*

#### Giai đoạn 1 – Nền móng (12/10 → 15/10)

| ID | Task | Owner | Reviewer | Thời gian | Phụ thuộc |
|---|---|---|---|---|---|
| P1-00 | Khảo sát Discovery & phương pháp crawl | Kiên | Bảo | 12/10 – 14/10 | – |
| P1-01 | Tạo repo GitHub & khung dự án | Kiệt | Khang | 12/10 – 13/10 | – |
| P1-02 | Data Schema & Mock Generator | Bảo | Kiệt | 12/10 – 15/10 | – |
| P1-03 | Khung Jupyter Book (repo riêng) | Nhi | Khang | 12/10 – 15/10 | – |
| P1-04 | Khảo sát LaTeX & Spec Tree | Khang | Nhi | 12/10 – 15/10 | – |
| P1-05 | Thiết kế kiến trúc ML & Predictions Spec | Kiệt | Bảo | 12/10 – 15/10 | – |

#### Giai đoạn 2 – Viết module (14/10 → 26/10)

| ID | Task | Owner | Reviewer | Thời gian | Phụ thuộc |
|---|---|---|---|---|---|
| P2-01 | Module Crawling arXiv | Kiên | Kiệt | 14/10 – 24/10 | P1-00, P1-02 |
| P2-02 | Module S2 API & Profiler | Bảo | Nhi | 16/10 – 25/10 | P1-02 |
| P2-03 | Multi-file Resolver & Tree Builder | Nhi | Khang | 16/10 – 24/10 | P1-02, P1-04 |
| P2-05 | LaTeX Cleanup & Normalization | Kiên | Khang | 16/10 – 25/10 | P1-02, P1-04 |
| P2-04 | BibTeX Extractor & 2-level Dedup | Khang | Nhi | 16/10 – 26/10 | P1-02, P1-04, P2-03, P2-05 |
| P2-06 | Preprocessing & Feature Engineering | Kiệt & Bảo | Kiên | 16/10 – 26/10 | P1-02, P1-05 |

#### Giai đoạn 3 – Chạy dữ liệu thật & Benchmark

| ID | Task | Owner | Reviewer | Thời gian | Phụ thuộc |
|---|---|---|---|---|---|
| P3-01 | Đóng gói Pipeline Runner (`main.py`) | Khang | Kiên | 26/10 – 28/10 | P2-01..05 |
| P3-02a | **Crawl thật M1** (mỗi người 5.000 bài trên dải cá nhân, gom lên Drive; tổng 25.000 bài) | Cả 5 | Kiên, Bảo | 26/10 – 28/10 | P2-01, P2-02 |
| P3-02b | Colab Benchmark M1 & Summary Export | Bảo | Kiên | 29/10 – 30/10 | P3-02a |
| P3-03a | **Parse chung M2** (25.000 bài, chạy tập trung) | Khang | Nhi | 30/10 – 06/11 | P3-01, P3-02a |
| P3-03b | Colab Benchmark M2 & Parse Summary | Nhi | Khang | 07/11 – 08/11 | P3-03a |
| P3-04 | Gán nhãn Reference Matching (cả 5 cùng gán) | Cả 5 | Kiệt | 31/10 – 08/11 | P1-05, P1-02, P3-03a |

#### Giai đoạn 4 – Báo cáo, ML, Hugging Face, Poster

| ID | Task | Owner | Reviewer | Thời gian | Phụ thuộc |
|---|---|---|---|---|---|
| P4-03a | Báo cáo Jupyter Book M1 & README update | Kiên & Bảo | Nhi | 30/10 – 02/11 | P3-02b |
| P4-03b | Báo cáo Jupyter Book M2 | Nhi & Khang | Bảo | 08/11 – 12/11 | P3-03b |
| P4-01 | Huấn luyện model & đo MRR Top-5 | Kiệt | Bảo | 08/11 – 14/11 | P2-06, P3-04, P3-03a |
| P4-02 | Xuất 6 bảng Parquet tổng, tách 5 bộ theo dải ID & upload Hugging Face cá nhân | Bảo | Kiên | 15/11 – 18/11 | P3-03a, P4-01 |
| P4-04 | Thiết kế Poster A2 & Documentation | Kiên & Kiệt | Bảo | 15/11 – 20/11 | P4-01, P4-03b |
| P4-05 | Benchmark cuối & nộp Moodle | Cả 5 | Kiên | 20/11 – 22/11 | P4-01..04 |

> 💡 **Cột "Phụ thuộc"** nghĩa là: task này chỉ bắt đầu đầy đủ khi các task trong cột đó đã xong.

### 5.4 Sản phẩm cần nộp trên Moodle

| Milestone | Nộp gì |
|---|---|
| M1 | Link Code Repo (không có data) + Link Book Repo riêng + báo cáo M1 |
| M2 | Link báo cáo Jupyter Book M2 |
| M3 | Poster A2 `.pptx` (420 × 594 mm, dọc) + Link Hugging Face (5 dataset cá nhân, mỗi thành viên 1 link ~5.000 bài) + Link Code Repo |

---

## 🔀 6. Quy trình làm việc & Git flow

### 6.1 Giao tiếp

- **Họp sync:** 1 buổi/tuần, **Chủ Nhật 20:00** qua Discord.
- **Cập nhật hằng ngày:** nhắn **1 câu ngắn** vào channel Discord nhóm (làm gì / vướng gì).

### 6.2 Mô hình nhánh

```
main  ●───────●──────────●─────────●        ← luôn chạy được, chỉ merge qua PR
       \     /  \       /  \       /
        ●───●    ●─────●    ●─────●        ← feature/<TASK-ID>-<mô-tả>
```

- `main` là nhánh **ổn định**: luôn cài được, `pytest` luôn xanh. **Không ai push trực tiếp.**
- Mọi thay đổi đi theo chu trình: **tạo nhánh → commit → push → Pull Request → 2 Approve + CI xanh → merge → xóa nhánh**.
- Mỗi task trong bảng ở mục 5 tương ứng **một nhánh** và **một PR**.

### 6.3 Quy tắc đặt tên nhánh

Cú pháp: `<loại>/<TASK-ID>-<mô-tả-ngắn>`

| Loại | Dùng khi | Ví dụ |
|---|---|---|
| `feature/` | Làm task mới (phổ biến nhất) | `feature/P2-01-crawler` |
| `fix/` | Sửa lỗi của code đã merge | `fix/P2-01-handle-403` |
| `docs/` | Chỉ sửa tài liệu, README, spec | `docs/P1-04-latex-spec` |
| `test/` | Chỉ bổ sung test | `test/P2-04-dedup-cases` |
| `refactor/` | Dọn code, không đổi hành vi | `refactor/P2-03-tree-builder` |

Quy tắc chi tiết:

- **TASK-ID viết hoa đúng như bảng task** (`P2-01`, `P3-02a`), phần mô tả viết **chữ thường, ngăn cách bằng dấu gạch ngang**, không dấu tiếng Việt, không khoảng trắng, tối đa khoảng 4 từ.
- Một nhánh chỉ làm **một task**. Không gộp nhiều task vào một nhánh.
- Task có 2 Owner (ví dụ P2-06 của Kiệt & Bảo): **một người tạo nhánh**, người kia push vào cùng nhánh sau khi báo trên Discord.
- ✅ `feature/P2-05-latex-cleanup` · ❌ `kien-branch`, `test123`, `Feature/Crawler_New`

### 6.4 Từng bước làm một task (ví dụ P2-01)

```bash
# 1. Cập nhật main
git checkout main
git pull origin main

# 2. Tạo nhánh riêng cho task
git checkout -b feature/P2-01-crawler

# 3. Code, rồi commit nhỏ & thường xuyên
git add src/m1_crawling/crawler.py
git commit -m "P2-01 feat: thêm hàm crawl_paper với retry"

# 4. Trước khi mở PR: lấy main mới nhất về nhánh của bạn
git fetch origin
git merge origin/main

# 5. Đẩy nhánh lên GitHub
git push -u origin feature/P2-01-crawler

# 6. Mở Pull Request trên GitHub → gán đúng 2 Reviewers theo bảng task
# 7. Sửa theo góp ý (commit thêm vào cùng nhánh) → đủ 2 Approve + CI xanh thì Merge
# 8. Sau khi merge: xóa nhánh và cập nhật main
git checkout main
git pull origin main
git branch -d feature/P2-01-crawler
```

### 6.5 Quy tắc commit message

Cú pháp: `<TASK-ID> <loại>: <mô tả ngắn, thể mệnh lệnh>`

| Loại | Ý nghĩa |
|---|---|
| `feat` | Thêm chức năng |
| `fix` | Sửa lỗi |
| `docs` | Tài liệu |
| `test` | Thêm/sửa test |
| `refactor` | Dọn code, không đổi hành vi |
| `chore` | Việc vặt: config, requirements, .gitignore |

```
✅ P2-01 feat: thêm retry khi gặp HTTP 403
✅ P2-04 fix: giữ trường doi khi gộp BibTeX trùng
✅ P1-04 docs: bổ sung quy tắc xử lý itemize
❌ update
❌ fix bug
❌ done
```

- Dòng đầu **không quá ~72 ký tự**; cần giải thích thêm thì viết ở phần thân (cách một dòng trống).
- **Mỗi commit làm một việc.** Commit nhỏ, thường xuyên thay vì một commit khổng lồ cuối ngày.
- Không commit code đang lỗi cú pháp hoặc `pytest` đang đỏ.

### 6.6 Quy tắc Pull Request

- **Tiêu đề PR:** `<TASK-ID>: <tên task>` (ví dụ `P2-01: Module crawling arXiv`).
- **Mô tả PR** theo mẫu `.github/pull_request_template.md`: làm gì, cách kiểm tra, tài liệu tham khảo (citation).
- **Kích thước:** nên dưới ~400 dòng thay đổi; PR quá lớn thì tách nhỏ để dễ review.
- **Gán đúng 2 Reviewers** theo cột "Reviewers" trong bảng task.
- Mở **Draft PR** nếu muốn nhận góp ý sớm khi code chưa xong.
- **Điều kiện merge** (bắt buộc cả ba):
  1. Đủ **2 Approve** từ đúng 2 Reviewers.
  2. **CI xanh** (`ruff` + `pytest`).
  3. Đã **giải quyết hết comment** và nhánh không còn conflict với `main`.
- Dùng **Squash and merge** để mỗi task chỉ còn **một commit gọn** trên `main`; commit message khi squash giữ dạng `P2-01 feat: ...`.
- **Xóa nhánh** ngay sau khi merge.
- **Owner không tự merge khi chưa đủ duyệt.** Sau khi đủ điều kiện, Owner là người bấm merge.

### 6.7 Quy tắc Review

- Reviewer **phản hồi trong vòng 24 giờ**; nếu bận thì báo trên Discord để đổi người.
- Góp ý cụ thể, lịch sự, nêu rõ **vì sao** và **gợi ý cách sửa**; phân biệt lỗi bắt buộc sửa với gợi ý tùy chọn (đánh dấu `nit:`).
- Chỉ **Approve** khi đã đọc kỹ và chạy được code. Không Approve cho có.
- Owner trả lời từng comment (sửa xong thì ghi "Done", không đồng ý thì giải thích) rồi mới yêu cầu review lại.

### 6.8 Xử lý Merge Conflict

```bash
git fetch origin
git merge origin/main          # Git báo conflict ở file nào đó
# Mở file, tìm các dấu <<<<<<<  =======  >>>>>>>, chọn nội dung đúng, xóa dấu
git add <file-đã-sửa>
git commit                     # hoàn tất merge
pytest -q                      # chạy lại test trước khi push
git push
```

- Giảm conflict bằng cách **chia file theo module** (mỗi owner sửa file của mình) và **merge `main` vào nhánh thường xuyên**.
- Cần sửa file của người khác (ví dụ `schemas.py`): **báo trên Discord trước** và tạo PR riêng, nhỏ, review nhanh.
- **Không** dùng `git push --force` lên `main`; trên nhánh cá nhân chỉ dùng `--force-with-lease` khi thật sự cần và đã báo Reviewer.

### 6.9 Quy định trình bày code

#### 6.9.1 Ngôn ngữ và định dạng cơ bản

| Hạng mục | Quy định |
|---|---|
| Phiên bản | Python **3.11** |
| Thụt lề | **4 dấu cách**, không dùng Tab |
| Độ dài dòng | Tối đa **100 ký tự** |
| Mã hóa file | **UTF-8**; khi mở file luôn ghi rõ `encoding="utf-8"` |
| Dấu nháy | Chuỗi dùng `"..."` (nháy kép), do `ruff format` tự xử lý |
| Cuối file | Có đúng **1 dòng trống**, không có khoảng trắng thừa cuối dòng |
| Dòng trống | 2 dòng trống giữa các hàm/class cấp cao, 1 dòng trống giữa các method |
| Ngôn ngữ tên (biến, hàm, class, file) | **Tiếng Anh** |
| Ngôn ngữ docstring/comment | **Tiếng Việt**|

#### 6.9.2 Quy tắc đặt tên

| Đối tượng | Kiểu | Ví dụ |
|---|---|---|
| File, package, module | `snake_case` | `latex_parser.py`, `m1_crawling/` |
| Hàm, method, biến | `snake_case`, là động từ cho hàm | `crawl_paper`, `size_before` |
| Class (kể cả Pydantic model) | `PascalCase`, danh từ | `PaperMetadata`, `BibtexEntry` |
| Hằng số | `UPPER_SNAKE_CASE`, đặt đầu file | `ARXIV_DELAY_SECONDS = 3` |
| Biến/hàm nội bộ module | Bắt đầu bằng `_` | `_strip_comments` |
| Biến boolean | Bắt đầu bằng `is_`, `has_`, `should_` | `is_pdf_only`, `has_bib` |
| Test | `test_<hành vi>` | `test_dedup_keeps_all_fields` |

Quy tắc kèm theo:

- Tên **mô tả đúng ý nghĩa**, không viết tắt khó hiểu: ✅ `reference_count`, ❌ `rc`, `tmp2`, `data1`.
- Không đặt tên một chữ cái, **trừ** biến vòng lặp ngắn (`i`, `k`, `v`) và công thức toán.
- Từ khóa miền bài toán dùng nhất quán trong toàn repo: `arxiv_id` (không `paper_id`/`aid`), `citation_key`, `version`, `element_id`, `parent_id`. Tên cột bảng Parquet phải **trùng đúng** tên trường trong `src/schemas.py`.
- Một kiểu `arxiv_id` duy nhất trong code: dạng **chuẩn của arXiv** (`2506.00788`, `hep-th/9901001`). Chỉ đổi sang dạng thư mục `yymm-id` (`2506-00788`) tại **một hàm tiện ích duy nhất** trong `utils_io.py`, không tự viết chuyển đổi rải rác.

#### 6.9.3 Cấu trúc file và thứ tự import

Thứ tự trong một file `.py`:

1. Docstring module (1-3 dòng: file này làm gì)
2. `import` (3 nhóm, cách nhau 1 dòng trống: thư viện chuẩn → thư viện ngoài → module của dự án)
3. Hằng số
4. Class / hàm
5. `if __name__ == "__main__":` (nếu có)

```python
"""Tải mã nguồn LaTeX của bài báo từ export.arxiv.org (M1)."""

import logging
import time
from pathlib import Path

import requests

from src.common.utils_io import write_json
from src.schemas import PaperMetadata

logger = logging.getLogger(__name__)

ARXIV_SRC_URL = "https://export.arxiv.org/src/{arxiv_id}v{version}"
ARXIV_DELAY_SECONDS = 3
```

- **Import tuyệt đối** từ gốc repo (`from src.common...`), không dùng import tương đối kiểu `from ..common`.
- **Không** `from x import *`.
- Không import thừa.

#### 6.9.4 Type hint (chú thích kiểu)

- **Mọi hàm public** (không bắt đầu bằng `_`) phải có type hint cho tham số và giá trị trả về.
- Dùng cú pháp Python 3.11: `list[str]`, `dict[str, int]`, `str | None` (không dùng `List`, `Dict`, `Optional`).
- Dữ liệu có cấu trúc cố định dùng **Pydantic model** trong `src/schemas.py`, không truyền `dict` tự do giữa các module.

```python
def fetch_references(arxiv_id: str, api_key: str) -> dict[str, dict]:
    ...
```

#### 6.9.5 Docstring

Dùng **kiểu Google**. Hàm/class public bắt buộc có docstring: dòng tóm tắt, sau đó `Args`, `Returns`, `Raises` khi cần.

```python
def crawl_paper(arxiv_id: str, out_dir: Path) -> bool:
    """Tải mọi phiên bản (v1, v2, ...) của một bài báo và xóa file ảnh.

    Args:
        arxiv_id: Mã arXiv chuẩn, ví dụ "2506.00788".
        out_dir: Thư mục đầu ra; sẽ tạo thư mục con theo dạng "yymm-id".

    Returns:
        True nếu tải được ít nhất một phiên bản có mã nguồn TeX,
        False nếu bài chỉ có PDF/Word (khi đó vẫn tạo record rỗng).

    Raises:
        requests.HTTPError: Khi arXiv trả mã lỗi sau khi đã thử lại hết số lần.
    """
```

- Dòng tóm tắt viết **một câu**, thể mệnh lệnh/mô tả, kết thúc bằng dấu chấm.
- Docstring nói **làm gì và vì sao**; còn **cách làm** thì viết trong comment cạnh code.

#### 6.9.6 Comment

- Comment giải thích **lý do (why)**, không lặp lại điều code đã nói rõ (what).
- Comment đặt **trên** dòng code, cùng mức thụt lề; tránh comment cuối dòng dài.
- Đoạn regex hoặc thuật toán khó **bắt buộc** có comment mô tả mẫu nó khớp, kèm một ví dụ.
- **Không** để code bị comment-out trong PR (đã có Git lưu lịch sử).
- Đánh dấu việc dang dở theo mẫu: `# TODO(Khang, P2-04): xử lý \bibitem có nhiều dòng`. Không dùng `TODO` không tên người và không mã task.

```python
# Khớp "\bibitem{key} nội dung..." đến trước \bibitem kế tiếp hoặc hết khối.
# Ví dụ: "\bibitem{smith19} J. Smith. Deep Learning. 2019."
BIBITEM_PATTERN = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*?)(?=\\bibitem|\Z)", re.S)
```

#### 6.9.7 Hàm và cấu trúc code

- Mỗi hàm **làm một việc**; độ dài khuyến nghị **dưới ~40 dòng**, dài hơn thì tách hàm.
- Tối đa **5 tham số**; nhiều hơn thì gom vào một Pydantic model hoặc dataclass.
- Hạn chế lồng nhau quá **3 cấp** (`if`/`for`): dùng *early return* để code phẳng hơn.
- **Không dùng biến toàn cục thay đổi được.** Hằng số thì được.
- **Không hard-code**: delay, số lần retry, đường dẫn, `top_k`… nằm trong `configs/config.yaml`; khóa bí mật lấy từ biến môi trường (xem mục 8). Không viết cứng đường dẫn tuyệt đối kiểu `C:\Users\...`.
- Đường dẫn dùng `pathlib.Path`, **không** nối chuỗi bằng `+` hay `os.path.join`.
- Regex dùng nhiều lần thì `re.compile` một lần ở đầu file, đặt tên hằng số.

```python
# ✅ Early return, phẳng
def parse_year(raw: str | None) -> int | None:
    if not raw:
        return None
    match = YEAR_PATTERN.search(raw)
    return int(match.group()) if match else None
```

#### 6.9.8 Xử lý lỗi và logging

Dự án chạy hàng nghìn bài, nên **một bài lỗi không được làm sập cả dải** (xem mục 14).

- **Không** dùng `except:` trống hoặc `except Exception: pass` (nuốt lỗi âm thầm). Bắt **loại lỗi cụ thể**; nếu bắt `Exception` ở vòng lặp ngoài cùng của từng bài thì **bắt buộc** `logger.exception(...)` và ghi ID lỗi.
- Dữ liệu có thể thiếu (PDF-only, LaTeX lạ): kiểm tra `None`/rỗng trước khi dùng (defensive programming), trả về giá trị rỗng hợp lệ thay vì để `KeyError`.
- **Không dùng `print()`** trong `src/` (`ruff` rule `T20` sẽ báo). Dùng `logging`:

| Mức | Dùng khi |
|---|---|
| `DEBUG` | Chi tiết khi phát triển |
| `INFO` | Tiến độ bình thường ("Đã crawl 2506.00788 v1, v2") |
| `WARNING` | Bất thường nhưng vẫn chạy tiếp (bài PDF-only, ref thiếu arXiv ID) |
| `ERROR` | Một bài thất bại, kèm `arxiv_id` và lý do |

```python
logger.info("Crawl xong %s (%d phiên bản)", arxiv_id, n_versions)   # dùng %s, không dùng f-string
```

- Log lỗi luôn kèm `arxiv_id` để truy vết và để thống kê cho báo cáo M1/M2.

#### 6.9.9 Làm việc với file và dữ liệu

- Đọc/ghi JSON và checkpoint **chỉ qua các hàm trong `src/common/utils_io.py`**, không mỗi người tự viết một kiểu.
- Ghi file bằng `encoding="utf-8"`; JSON ghi với `ensure_ascii=False, indent=2` cho dễ đọc.
- Ghi file **an toàn khi bị ngắt** (checkpoint): ghi ra file tạm rồi đổi tên, để Ctrl+C giữa chừng không tạo file hỏng.
- Xử lý **từng bài một** rồi ghi ra đĩa và giải phóng bộ nhớ, không giữ cả dải ID trong RAM.
- Cột/bảng Parquet phải **phẳng**, đúng kiểu như `src/schemas.py`; không nhét JSON dạng chuỗi hoặc object lồng nhau.

#### 6.9.10 Quy tắc riêng cho crawler (bắt buộc)

- Mọi request tới arXiv đi qua **một hàm duy nhất** có sẵn delay **≥ 3 giây**, `User-Agent` đọc từ `ARXIV_USER_AGENT`, **một kết nối/lần**. Cấm gọi `requests.get` trực tiếp tới arXiv ở chỗ khác.
- Chỉ dùng domain `export.arxiv.org`; **cấm** `arxiv.org`.
- Mọi request đều có `timeout=` (không để treo vô hạn) và có retry với thời gian chờ tăng dần.
- Semantic Scholar: delay 1s, header `x-api-key`, xử lý HTTP 429.

#### 6.9.11 Notebook (`.ipynb`)

Đề cho phép nộp `.ipynb`, nhưng notebook rất khó review. Quy định:

- Code **tái sử dụng** (crawler, parser, model) viết trong `src/*.py`. Notebook chỉ để **thử nghiệm, phân tích dữ liệu, vẽ biểu đồ**.
- **Xóa toàn bộ output** trước khi commit (`Kernel → Restart & Clear Output`, hoặc `jupyter nbconvert --clear-output --inplace <file>`), vì output chứa dữ liệu thật và làm diff khó đọc.
- Notebook phải **chạy được từ trên xuống dưới** (`Restart & Run All`) mà không lỗi.
- Notebook báo cáo Jupyter Book nằm ở `book-repo/`, không để trong repo code.

#### 6.9.12 Viết test

- Mỗi test kiểm tra **một hành vi**, tên mô tả hành vi (`test_resume_skips_finished_papers`).
- Theo mẫu **Arrange – Act – Assert** (chuẩn bị, chạy, kiểm tra), cách nhau bằng dòng trống.
- Chỉ dùng **mock data** trong `tests/mock_data/`; test **không gọi mạng thật**, hãy giả lập response.
- Bug được sửa thì thêm một test tái hiện bug đó.

#### 6.9.13 Những điều không làm

| ❌ Không | ✅ Thay bằng |
|---|---|
| `print(...)` để debug rồi quên xóa | `logger.debug(...)` |
| `except: pass` | Bắt lỗi cụ thể + `logger.exception` |
| Hard-code delay, đường dẫn, API key | `configs/config.yaml`, `.env` |
| `from module import *` | Import tên cụ thể |
| Code bị comment-out | Xóa đi, Git đã lưu lịch sử |
| Hàm 200 dòng làm nhiều việc | Tách thành các hàm nhỏ |
| Tự viết lại hàm đọc/ghi JSON | Dùng `utils_io.py` |
| `dict` tự do giữa các module | Pydantic model trong `schemas.py` |
| Biến tên `a`, `tmp`, `data2` | Tên có nghĩa |
| Copy code không ghi nguồn | Ghi citation tại chỗ |

### 6.10 Bảng tóm tắt

| ✅ Nên | ❌ Không |
|---|---|
| Mỗi task một nhánh `feature/<TASK-ID>-...` | Push thẳng lên `main` |
| Tên nhánh chữ thường, gạch ngang, có TASK-ID | Đặt tên theo người (`kien-branch`) hoặc mơ hồ (`test`) |
| Commit dạng `P2-01 feat: ...` | Commit "update", "fix", "done" |
| PR đủ **2 Approve + CI xanh** mới merge | Tự merge khi chưa đủ duyệt |
| Merge `origin/main` vào nhánh trước khi mở PR | Để nhánh lệch `main` hàng tuần |
| Chạy `ruff` và `pytest` trước khi push | Commit khi test đang đỏ |
| Ghi citation khi tham khảo code/tài liệu | Chép code không dẫn nguồn |
| Dùng `.env` cho API key | **Commit API key / data thật** |
| Xóa nhánh sau khi merge | Để nhánh cũ tồn đọng |


### 6.11 Checklist khi review PR

- [ ] Code đúng yêu cầu của task (xem cột "Detail" trong bảng tiến độ)
- [ ] Tên nhánh và commit message đúng quy ước
- [ ] Có test, test pass trên mock data; CI xanh
- [ ] Không có data thật / API key / file lớn
- [ ] Có docstring và citation nếu cần
- [ ] Tuân thủ quy định crawler (delay ≥ 3s, User-Agent, chỉ dùng `export.arxiv.org`)
- [ ] Không phá `src/schemas.py` (schema phải ổn định theo IDD)
- [ ] Hàm public có type hint + docstring Google; không còn `print()`, code comment-out, `except: pass`

---

## 🔌 7. API Reference

### 7.1 API bên ngoài

#### arXiv Source – tải mã nguồn bài báo

| | |
|---|---|
| **URL** | `https://export.arxiv.org/src/{id}v{n}` (ví dụ `.../src/2506.00788v1`) |
| **Trả về** | File `.tar.gz` (nhiều file) hoặc `.gz` (một file `.tex`) |
| **Giới hạn** | **≥ 3 giây giữa 2 request**, **1 kết nối tại một thời điểm** |
| **Bắt buộc** | `User-Agent` mô tả rõ (tên dự án + email liên hệ) |
| **Cấm** | ❌ Không dùng domain `arxiv.org` để tải; chỉ dùng `export.arxiv.org` |

```python
import requests, time

headers = {"User-Agent": "DS-Project-HCMUS/1.0 (mailto:ban@student.hcmus.edu.vn)"}
resp = requests.get("https://export.arxiv.org/src/2506.00788v1", headers=headers, timeout=60)
time.sleep(3)  # BẮT BUỘC ≥ 3 giây
```

#### Semantic Scholar (S2) – lấy danh sách tham khảo

| | |
|---|---|
| **URL** | `https://api.semanticscholar.org/graph/v1/paper/arXiv:{id}/references` |
| **Xác thực** | Header `x-api-key: <S2_API_KEY>` |
| **Giới hạn** | Delay **1 giây** giữa các request; gặp lỗi **HTTP 429** thì chờ rồi thử lại |
| **Lọc** | Lưu trữ toàn bộ tham khảo cục bộ; lọc theo arXiv ID khi xuất bảng Parquet `references` |

```python
url = "https://api.semanticscholar.org/graph/v1/paper/arXiv:2506.00788/references"
params = {"fields": "title,externalIds", "limit": 1000}
resp = requests.get(url, params=params, headers={"x-api-key": S2_API_KEY})
```

> 📎 Kiểm tra lại tên field và giới hạn tại tài liệu chính thức của S2 trước khi code, vì API có thể thay đổi.

#### Discovery – 3 phương pháp cần so sánh (P1-00)

| Phương pháp | Mô tả ngắn |
|---|---|
| **arXiv API (batch ID)** | Truy vấn trực tiếp theo danh sách ID |
| **OAI-PMH** (qua `sickle`) | Giao thức chuẩn thu hoạch metadata hàng loạt |
| **Kaggle dataset** | Tải dataset arXiv có sẵn |

Kết quả so sánh, lý do chọn và thời gian đo được ghi trong `docs/discovery_report.md`.

#### Hugging Face Hub

Dùng `huggingface_hub` để upload 6 bảng Parquet và Dataset Card. Bộ tổng 25.000 bài được tách thành 5 bộ theo dải ID cá nhân; **mỗi thành viên dùng `HF_TOKEN` của mình** để upload bộ của mình. Cần `HF_TOKEN`. Hướng dẫn hosting chính thức xin từ Lab Instructors (task P4-02).

### 7.2 Giao diện nội bộ (module)

| Module | Hàm chính | Chức năng |
|---|---|---|
| `m1_crawling.crawler` | `crawl_paper(arxiv_id, out_dir)` | Tải & xử lý 1 bài (mọi version) |
| | `crawl_range(id_start, id_end, out_dir, resume)` | Crawl cả dải ID, có checkpoint |
| `m1_crawling.s2_api` | `fetch_references(arxiv_id)` | Lấy `references.json` |
| `m1_crawling.profiler` | `Profiler` (context manager) | Đo RAM đỉnh, time/bài, đĩa |
| `m2_parsing.latex_parser` | `find_main_tex`, `resolve_includes` | Tìm file chính, ghép `\input` |
| `m2_parsing.tree_builder` | `build_tree(latex, arxiv_id)` | Dựng `hierarchy.json` |
| `m2_parsing.latex_cleanup` | `cleanup_latex`, `normalize_math` | Dọn LaTeX, chuẩn hóa toán |
| `m2_parsing.bibtex_extractor` | `extract_bibitems(latex)` | `\bibitem` → danh sách entry |
| `m2_parsing.dedup` | `dedup_bibtex`, `dedup_fulltext` | Khử trùng lặp |
| `m3_matching.features` | `preprocess`, `build_features` | Làm sạch & tạo đặc trưng |
| `m3_matching.matching_model` | `MatchingModel.fit / .rank` | Huấn luyện & xếp hạng |
| `m3_matching.evaluate` | `mrr(ranked, truth)` | Tính MRR |
| `m3_matching.export_parquet` | `export_all(processed_dir, out_dir)` | Xuất 6 bảng tổng (kèm script filter tách 5 bộ theo dải ID) |

> 🚧 Phần lớn hàm đang là **khung (stub)**, mỗi owner hiện thực theo task của mình. Chữ ký hàm có thể điều chỉnh khi code, nhưng **schema trong `src/schemas.py` phải ổn định** (xem IDD).

---

## 🔐 8. Biến môi trường

Biến môi trường là **thông tin bí mật hoặc riêng của từng người** (như API key) không được ghi cứng trong code.

| Biến | Bắt buộc | Mô tả | Lấy ở đâu |
|---|---|---|---|
| `S2_API_KEY` | ✅ | Khóa Semantic Scholar | Đăng ký trên trang Semantic Scholar API |
| `ARXIV_USER_AGENT` | ✅ | Chuỗi định danh khi gọi arXiv, ví dụ `DS-Project-HCMUS/1.0 (mailto:email@student.hcmus.edu.vn)` | Tự đặt, có email thật |
| `HF_TOKEN` | Khi upload | Token Hugging Face (quyền write) | Hugging Face → Settings → Access Tokens |

**Cách thiết lập:**

```bash
cp .env.example .env      # Windows (cmd): copy .env.example .env
# Mở .env bằng editor và điền giá trị thật
```

> 🚨 **Không bao giờ commit file `.env`.** File này đã nằm trong `.gitignore`. Nếu lỡ commit key, hãy **thu hồi (revoke) và tạo key mới ngay**.

Tham số không bí mật (delay, retry, đường dẫn, `top_k`…) nằm trong `configs/config.yaml`.

---

## 🔧 9. Cài đặt (venv / conda)

### 9.1 Yêu cầu

- **Python 3.11** (khuyến nghị)
- **Git**
- Kết nối internet; ~vài GB trống để thử nghiệm local (data thật lưu trên Drive)

Kiểm tra:
```bash
python --version     # hoặc python3 --version
git --version
```

### 9.2 Clone repo

```bash
git clone https://github.com/tkiet06-us/Intro2DS---ArXiv-Project.git
cd Intro2DS---ArXiv-Project
```

### 9.3 Cách A – `venv` (có sẵn trong Python)

**Môi trường ảo (virtual environment)** là "căn phòng riêng" chứa thư viện của dự án, tránh xung đột với các dự án khác trên máy.

```bash
# Tạo môi trường
python -m venv .venv

# Kích hoạt
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows (cmd)
.venv\Scripts\Activate.ps1         # Windows (PowerShell)

# Cài thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

Khi kích hoạt thành công, đầu dòng lệnh hiện `(.venv)`. Thoát bằng `deactivate`.

### 9.4 Cách B – `conda` (Anaconda / Miniconda)

```bash
conda create -n Intro2DS---ArXiv-Project python=3.11 -y
conda activate Intro2DS---ArXiv-Project
pip install -r requirements.txt
```

Thoát: `conda deactivate`. Xóa môi trường: `conda remove -n Intro2DS---ArXiv-Project --all`.

### 9.5 Cài nhanh bằng script (Linux/macOS)

```bash
bash scripts/setup_dev.sh
```
Script tự tạo venv, cài thư viện, tạo `.env`, sinh mock data và chạy test.

### 9.6 Hoàn tất cài đặt

```bash
cp .env.example .env               # rồi điền khóa của bạn
python tests/make_mock.py          # sinh dữ liệu giả
pytest -q                          # kiểm tra mọi thứ ổn
```

Thấy `passed` là bạn đã sẵn sàng. ✅

---

## 🚀 10. Chạy chương trình

> Luôn **kích hoạt môi trường ảo** và chạy lệnh **từ thư mục gốc** của repo.

### 10.1 Chạy toàn bộ pipeline (M1 + M2)

```bash
python -m src.main --id_start 2506.00788 --id_end 2506.05787
```

| Tham số | Ý nghĩa |
|---|---|
| `--id_start` | arXiv ID bắt đầu của dải |
| `--id_end` | arXiv ID kết thúc của dải |
| `--config` | Đường dẫn file cấu hình (mặc định `configs/config.yaml`) |
| `--no_resume` | Bỏ qua checkpoint, chạy lại từ đầu |

> 📌 Ví dụ trên là một dải ID cá nhân. Khi chạy thật: **crawl (M1)** mỗi người chạy dải của mình rồi gom lên Drive; **parse (M2)** chạy **tập trung** trên bộ 25.000 bài đã gom.

**Checkpoint/Resume:** nếu chương trình bị ngắt (mất mạng, tắt máy), chỉ cần **chạy lại đúng lệnh cũ**, nó sẽ tiếp tục từ bài đang dở, không tải lại từ đầu.

### 10.2 Chạy từng module (khi phát triển)

```bash
python -m src.m3_matching.train        # huấn luyện + xuất predictions (P4-01)
```

### 10.3 Thử nhanh với dữ liệu giả (không cần mạng)

```bash
python tests/make_mock.py
pytest -q
```

### 10.4 Kế hoạch chạy dữ liệu thật

1. **Đợt 1 (Crawl riêng M1, 26/10 → 28/10):** mỗi người crawl 5.000 bài trên dải ID cá nhân, rồi gom toàn bộ kết quả (mã nguồn + `references.json`) lên Drive chung. *Crawl source trước, gọi S2 API bổ sung sau.*
2. **Đợt 2 (Parse chung M2, 30/10 → 06/11):** **đóng băng parser**, rồi đại diện nhóm (Nhi/Khang) chạy parser đồng loạt cho 25.000 bài đã gom trên Drive.
3. **Đợt 3 (Matching & Upload, 31/10 → 22/11):** gán nhãn thủ công → train mô hình → xuất 6 bảng Parquet tổng. Dùng script cắt (filter) 6 bảng này theo dải ID của từng người để tự nộp lên Hugging Face.

> ⚠️ Chạy thật tốn thời gian (delay ≥ 3s/request × nhiều version × 5.000 bài/người). **Hãy chạy thử 5-10 bài trước**, rồi mới chạy cả dải.

---

## 🧪 11. Testing

**Test** là code kiểm tra code khác có chạy đúng không. Dự án dùng `pytest`.

```bash
pytest -q                              # chạy tất cả
pytest tests/m3_matching -q            # chỉ một nhóm
pytest tests/m3_matching/test_evaluate.py::test_mrr   # chỉ một test
pytest --cov=src --cov-report=term-missing            # xem độ phủ code
ruff check src tests                   # kiểm tra style/lỗi cú pháp
```

**Kết quả đọc thế nào?**
- `.` = pass · `F` = fail · `s` = skip (test chưa viết, đang `TODO`)

**Quy ước:**
- Test đặt trong `tests/`, cấu trúc **phản chiếu** `src/` (`src/m2_parsing/dedup.py` ↔ `tests/m2_parsing/test_dedup.py`).
- Test chạy trên **mock data** (`tests/mock_data/`), **không phụ thuộc mạng hay data thật**.
- Owner của module chịu trách nhiệm viết test; Reviewer kiểm tra test đủ chưa.
- **CI** (GitHub Actions) tự chạy `ruff` + `pytest` trên mọi PR. PR đỏ thì không merge.

**Ví dụ viết test:**
```python
# tests/m3_matching/test_evaluate.py
from src.m3_matching.evaluate import mrr

def test_mrr():
    # bài 1: đáp án "b" ở hạng 2 → 1/2; bài 2: không có → 0; trung bình = 0.25
    assert mrr([["a", "b"], ["x", "y"]], ["b", "z"]) == 0.25
```

---

## 💾 12. Quản lý dữ liệu

| Loại dữ liệu | Lưu ở đâu | Có commit vào GitHub không? |
|---|---|---|
| Code, docs, config | GitHub repo | ✅ Có |
| Mock data nhỏ (`tests/mock_data/`) | GitHub repo | ✅ Có |
| 25.000 bài thật (`data/processed/`) | Google Drive chung / HF Private | ❌ **Không** |
| File `.tex` / `.bib` nguồn (`data/sources/`) | Google Drive chung | ❌ **Không** |
| Nhãn `ground_truth.json` (`data/labels/`) | Google Drive chung / HF Private | ❌ **Không** |
| `summary_stats.json` (nhẹ) | `book-repo/` (repo riêng) | ✅ Có (ở repo Book) |

**Gom dữ liệu crawl:** mỗi người upload kết quả crawl dải cá nhân (mã nguồn + `references.json`) lên Drive chung trước khi chạy parse chung.

**Giữ lại file `.tex`/`.bib` gốc** trong thư mục làm việc/Drive để lưu trữ và truy vấn local khi cần.

**Báo cáo Jupyter Book** build trên GitHub Actions: pipeline xuất `summary_stats.json` (M1) và `summary_stats_m2.json` (M2) → commit vào `book-repo/` → Notebook đọc & vẽ biểu đồ.

---

## 📐 13. Đặc tả dữ liệu đầu ra

### 13.1 Cây phân cấp `hierarchy.json` (M2)

- Cấp bậc tiêu đề được **tự nhận diện** (Chapter → Section → Subsection → …).
- **Lá của cây** gồm: câu (sentence), công thức khối (`$$...$$` / `equation`), hình/bảng (`figure`).
- Danh sách `itemize` là **nhánh con**.
- **Bỏ** section *References*; **giữ** *Acknowledgments* và *Appendices*.
- Mỗi element có thuộc tính `kind` (`document`, `heading`, `itemize`, `item`, `sentence`, `formula`, `figure`) theo chuẩn HF elements và **ID dạng publication key + anchor** (ví dụ: `2310-12345#root1`, `2310-12345#s1`).

*(Chi tiết đầy đủ ở `docs/latex_spec.md`.)*

### 13.2 Chuẩn hóa BibTeX (M2)

| Trường | Kiểu |
|---|---|
| `authors` | `list[str]` |
| `year` | `int` (có thể null) |
| `doi`, `journal`, `url` | `str` (có thể null) |

- **Field Union:** khi gộp các entry trùng, hợp nhất mọi trường thông tin có được.
- Đổi tên các `\cite{}` bị trùng key.

### 13.3 Gán nhãn & chia tập (M3)

- **Gán tay:** ≥ 5 bài (mỗi bài ≥ 20 BibTeX entry, tổng ≥ 100 entry gán nhãn), dựa trên parse đợt đầu M2. Mỗi thành viên gán ≥ 1 bài.
- **Gán tự động:** ≥ 10% phần còn lại.
- **Chia tập:** Test = 1 bài tay + 1 bài tự động; Valid = 1 tay + 1 tự động; Train = phần còn lại. *(Làm nhãn Train/Valid trước để có thể train sớm.)*
- Spec nhãn chi tiết được tài liệu hóa trong mục dưới (cập nhật bởi Kiệt, P1-05/P4-04).

> 📝 **Feature & Label spec:** *(TODO – Kiệt điền tại đây khi chốt: định nghĩa từng feature, lý do chọn dựa trên phân tích dữ liệu, định dạng nhãn.)*

### 13.4 Mô hình & đánh giá

- Bài toán: **xếp hạng** (ranking) các cặp (tham khảo × bài arXiv ứng viên), nhãn nhị phân Yes/No.
- Đặc trưng dự kiến: **Levenshtein**, **Jaccard n-gram**, **TF-IDF cosine**.
- Chỉ số: **MRR** (Mean Reciprocal Rank) trên tập Test.

> **MRR là gì?** Với mỗi tham khảo, nếu đáp án đúng nằm ở vị trí *k* trong danh sách dự đoán thì điểm là `1/k` (hạng 1 → 1.0, hạng 2 → 0.5, hạng 5 → 0.2, không có → 0). MRR là điểm trung bình trên mọi tham khảo; càng gần 1 càng tốt.

> ⚠️ Dùng mô hình **pretrained / LLM** sẽ được chấm theo **thang riêng** theo đề bài.

### 13.5 `predictions.parquet` (9 cột)

| Cột | Ý nghĩa |
|---|---|
| `arxiv_id` | Bài báo chứa tham khảo |
| `partition` | `train` / `valid` / `test` |
| `citation_key` | Khóa trích dẫn (`\cite{...}`) |
| `groundtruth_arxiv_id` | arXiv ID đúng (nhãn thật) |
| `cand_1` … `cand_5` | 5 ứng viên xếp hạng cao nhất |

### 13.6 6 bảng Parquet trên Hugging Face

`papers` · `references` · `elements` · `hierarchy` · `bibtex` · `predictions`

Mỗi bảng ở dạng **phẳng**, chia **shard**, kèm **Dataset Card** (`README.md` với YAML configs).

---

## 🛠 14. Troubleshooting

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Chạy sai thư mục / sai cách | Chạy từ **thư mục gốc** repo; dùng `python -m src.main ...` và `pytest` (đã có `pytest.ini`) |
| `ModuleNotFoundError: pydantic` (hay thư viện khác) | Chưa kích hoạt venv/conda hoặc chưa cài | Kích hoạt môi trường rồi `pip install -r requirements.txt` |
| `'python' is not recognized` (Windows) | Python chưa vào PATH | Cài lại Python và tick "Add to PATH", hoặc dùng `py -3.11` |
| PowerShell chặn `Activate.ps1` | Execution Policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` hoặc dùng cmd |
| **HTTP 403** khi tải arXiv | User-Agent thiếu/sai, request quá nhanh, hoặc dùng `arxiv.org` | Đặt User-Agent rõ ràng, delay ≥ 3s, **chỉ dùng `export.arxiv.org`**, 1 kết nối/lần |
| **HTTP 429** từ Semantic Scholar | Gọi quá nhanh / thiếu API key | Thêm `x-api-key`, delay 1s, retry có chờ |
| Lỗi `KeyError` / `None` khi parse | Bài thiếu trường (PDF-only, LaTeX lạ) | Xử lý **defensive** (kiểm tra null), ghi log, giữ record rỗng, không để chương trình sập |
| Bài tải về không có `.tex` | Chỉ có PDF/Word | Đúng thiết kế: tạo record rỗng + ghi log ID lỗi |
| Chạy dở bị ngắt | Mất mạng, tắt máy | Chạy lại cùng lệnh (Resume); chỉ thêm `--no_resume` khi muốn làm lại từ đầu |
| `git push` bị từ chối do file lớn | Lỡ `git add` data thật | `git rm --cached <file>`, kiểm tra `.gitignore`, commit lại. Nếu đã push, báo cả nhóm để xử lý lịch sử |
| Lỡ commit `.env` / API key | Thiếu cẩn thận | **Revoke key ngay**, tạo key mới, xóa file khỏi lịch sử Git |
| PR không merge được | Chưa đủ 2 Approve / CI đỏ | Nhắc Reviewer; xem log CI, sửa lỗi `ruff`/`pytest` |
| Merge conflict | Hai người sửa cùng file | `git pull origin main` vào nhánh của bạn, sửa chỗ xung đột, commit lại. Chia file theo module để giảm xung đột |
| Hết RAM/đĩa khi chạy dải lớn | Giữ cả dải trong bộ nhớ, không dọn file tạm | Xử lý từng bài rồi ghi ra đĩa; dọn file tạm; theo dõi bằng Profiler |
| Colab chạy chậm/khác máy local | Colab CPU-only | Benchmark chuẩn của đề là **Colab CPU-only**; đo trên 50–100 bài mẫu |
| Test `skip` rất nhiều | Chưa hiện thực module | Bình thường ở giai đoạn đầu; owner thay `@pytest.mark.skip` bằng test thật |

**Vẫn kẹt?** Nhắn vào channel Discord kèm: (1) lệnh bạn chạy, (2) thông báo lỗi **đầy đủ**, (3) hệ điều hành & phiên bản Python.

---

## ⚖ 15. Quy chế & Liêm chính học thuật

> 🚨 **Mọi hành vi gian lận, chép code không dẫn nguồn hoặc gian dối dữ liệu sẽ bị nhận 0 điểm cho TOÀN BỘ milestone.**

- 📌 **Được phép** tham khảo tài liệu online, nhưng **bắt buộc** ghi citation đầy đủ trong **code** (docstring/comment) **và báo cáo**.
- 🧠 Dữ liệu và mô hình phải do **chính nhóm tự triển khai và giải thích**; mỗi thành viên phải hiểu phần của mình để bảo vệ.
- 👥 Mỗi thành viên tự crawl dải ID cá nhân; phần parse chạy **tập trung** cho cả 25.000 bài; cả 5 thành viên đều tham gia gán nhãn thủ công và tự upload bộ Hugging Face theo dải ID cá nhân.

**Mẫu citation trong code:**
```python
# Tham khảo: Levenshtein distance, "Binary codes capable of correcting deletions,
# insertions and reversals", 1966. Dùng thư viện rapidfuzz (https://github.com/rapidfuzz/RapidFuzz)
```

---

## ❗ 16. Lưu ý quan trọng

1. **Tôn trọng arXiv:** vi phạm giới hạn truy cập có thể bị **chặn IP** (ảnh hưởng cả nhóm nếu chung mạng). Luôn có delay và User-Agent.
2. **Cleanup trước, Dedup sau:** nếu đảo thứ tự, so khớp nội dung giữa các version sẽ sai vì định dạng LaTeX khác nhau.
3. **Đóng băng parser** trước khi chạy parse thật 25.000 bài; không sửa logic giữa chừng để kết quả nhất quán.
4. **Xóa ảnh nhưng giữ `\begin{figure}`:** chỉ bỏ file ảnh (giảm dung lượng), **không** xóa lệnh figure trong `.tex`.
5. **Bài PDF/Word-only:** không bỏ qua, mà **giữ record rỗng** và ghi log.
6. **Ref không có arXiv ID** được lưu cục bộ trong `references.json`, chỉ lọc lấy các ref có arXiv ID khi xuất bảng `references` Parquet.
7. **Làm nhãn Train/Valid trước** để có thể huấn luyện sớm, nhãn Test làm sau.
8. **Xin hướng dẫn hosting HF** từ Lab Instructors **sớm**, đừng để đến ngày 15/11.
9. **Poster A2 dọc: 420 × 594 mm** (`.pptx`), chừa thời gian in trước khi nộp.
10. **Hạn nội bộ** sớm hơn hạn Moodle; hãy bám hạn nội bộ để có đệm rủi ro.
11. **Backup:** data thật chỉ nằm trên Drive/HF, không nằm trong Git; hãy chắc chắn có ít nhất một bản sao lưu.
12. **Xác nhận lại** cách chia dải ID của Nhi (xem mục 4) và quy ước đặt tên thư mục bài cũ (`hep-th/9901001`).

---

## 📖 17. Bảng thuật ngữ (Glossary)

| Thuật ngữ | Giải thích dễ hiểu |
|---|---|
| **arXiv** | Kho lưu trữ bài báo khoa học miễn phí (Vật lý, Toán, Khoa học máy tính…) |
| **arXiv ID** | Mã định danh bài báo, ví dụ `2506.00788` (năm-tháng + số thứ tự) |
| **Version (v1, v2…)** | Các phiên bản chỉnh sửa của cùng một bài |
| **LaTeX / `.tex`** | Ngôn ngữ soạn thảo khoa học, dùng lệnh như `\section{...}` để định dạng |
| **BibTeX / `.bib`** | Định dạng lưu danh sách tài liệu tham khảo |
| **`\bibitem`** | Một mục tham khảo viết trực tiếp trong LaTeX |
| **Crawling** | Tự động tải dữ liệu từ web bằng chương trình |
| **Rate limit** | Giới hạn số request/giây mà máy chủ cho phép |
| **User-Agent** | Dòng "tự giới thiệu" của chương trình khi gọi máy chủ |
| **API / API key** | Cổng giao tiếp lập trình / "chìa khóa" xác thực |
| **Parsing** | Phân tích văn bản thô thành cấu trúc có ý nghĩa |
| **Hierarchy tree** | Cây phân cấp: mục lớn chứa mục nhỏ chứa câu/công thức |
| **Leaf (lá)** | Phần tử cuối cùng của cây, không chứa phần tử con |
| **Regex** | Biểu thức chính quy: "mẫu" để tìm chuỗi văn bản |
| **Deduplication** | Loại bỏ dữ liệu trùng lặp |
| **Field Union** | Gộp nhiều bản ghi trùng, giữ mọi trường thông tin có được |
| **Schema** | "Bản thiết kế" mô tả dữ liệu có những trường nào, kiểu gì |
| **Pydantic** | Thư viện Python để khai báo & kiểm tra schema |
| **Parquet** | Định dạng file bảng nén theo cột, gọn và nhanh, phổ biến trong data science |
| **Shard** | Chia một bảng lớn thành nhiều file nhỏ |
| **Mock data** | Dữ liệu giả nhỏ nhẹ dùng để kiểm thử |
| **Feature** | Đặc trưng: con số mô tả mức độ giống nhau giữa 2 đối tượng |
| **Levenshtein** | Số thao tác sửa tối thiểu để biến chuỗi A thành chuỗi B |
| **Jaccard (n-gram)** | Tỉ lệ phần chung giữa hai tập các cụm n ký tự/từ |
| **TF-IDF cosine** | Đo độ giống văn bản dựa trên tần suất từ (từ hiếm quan trọng hơn) |
| **Ranking** | Bài toán xếp hạng ứng viên từ tốt nhất đến kém nhất |
| **Ground truth** | Nhãn đúng do con người xác nhận |
| **Train / Valid / Test** | Tập huấn luyện / tập chỉnh tham số / tập chấm điểm cuối |
| **MRR** | Mean Reciprocal Rank: điểm trung bình của `1/hạng` đáp án đúng |
| **Checkpoint / Resume** | Lưu tiến độ / tiếp tục từ chỗ dừng |
| **Git branch / PR** | Nhánh code riêng / yêu cầu gộp nhánh vào `main` để review |
| **CI** | Tự động chạy test mỗi khi có thay đổi code |
| **Hugging Face** | Nền tảng lưu trữ & chia sẻ dataset/mô hình ML |
| **Jupyter Book** | Công cụ biến notebook thành sách/báo cáo web |

---

## 📚 18. Trích dẫn & Tài liệu tham khảo

*(Cập nhật liên tục. Mọi tài liệu/code tham khảo phải được liệt kê ở đây **và** trong docstring.)*

- Trang môn học: <https://hcmus-cs-ta.github.io/lab-book/intro-ds/>
- arXiv bulk data access & API terms of use: <https://info.arxiv.org/help/bulk_data.html>
- Semantic Scholar Academic Graph API: <https://api.semanticscholar.org/api-docs/>
- Hugging Face Datasets documentation: <https://huggingface.co/docs/datasets>
- Jupyter Book documentation: <https://jupyterbook.org>
- *(Thêm các nguồn khác tại đây)*

---

<p align="center">
  <i>Nhóm Nhập môn Khoa học Dữ liệu - Bảo · Nhi · Kiên · Khang · Kiệt</i><br>
  <i>Không 10 điểm thì học lại môn 🚀</i>
</p>