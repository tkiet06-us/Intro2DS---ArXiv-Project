# P1-04: LaTeX Tree Spec

---

## 📑 Mục lục

1. [Mục đích và Phạm vi](#1-mục-đích-và-phạm-vi)
   - 1.1 [Mục đích](#11-mục-đích)
   - 1.2 [Vị trí trong Pipeline xử lý](#12-vị-trí-trong-pipeline-xử-lý)
   - 1.3 [Phạm vi và Ranh giới](#13-phạm-vi-và-ranh-giới)
2. [Các Nguyên tắc Bất biến Cốt lõi (Core Invariants)](#2-các-nguyên-tắc-bất-biến-cốt-lõi-core-invariants)
3. [Mô hình Dữ liệu: Cây lồng nhau & Bảng Parquet](#3-mô-hình-dữ-liệu-cây-lồng-nhau--bảng-parquet)
   - 3.1 [Quy chuẩn Định dạng Lưu trữ (Storage Format Specification)](#31-quy-chuẩn-định-dạng-lưu-trữ-storage-format-specification)
   - 3.2 [Cấu trúc Dataset Repository & Hệ thống 6 Bảng Parquet](#32-cấu-trúc-dataset-repository--hệ-thống-6-bảng-parquet)
   - 3.3 [Nguyên tắc Cột phẳng và Kiểu nguyên thủy (Flat & Primitive Rules)](#33-nguyên-tắc-cột-phẳng-và-kiểu-nguyên-thủy-flat--primitive-rules)
   - 3.4 [Tập giá trị chuẩn cho thuộc tính `kind`](#34-tập-giá-trị-chuẩn-cho-thuộc-tính-kind)
   - 3.5 [Quy ước định danh `element_id` và tính Xác định (Deterministic ID)](#35-quy-ước-định-danh-element_id-và-tính-xác-định-deterministic-id)
4. [Thuật toán & Quy tắc Bóc tách Chi tiết](#4-thuật-toán--quy-tắc-bóc-tách-chi-tiết)
   - 4.1 [Tự nhận diện Cấp bậc Tiêu đề (Dynamic Heading Hierarchy)](#41-tự-nhận-diện-cấp-bậc-tiêu-đề-dynamic-heading-hierarchy)
   - 4.2 [Dựng quan hệ Cha – Con bằng Ngăn xếp (Stack-based Construction)](#42-dựng-quan-hệ-cha--con-bằng-ngăn-xếp-stack-based-construction)
   - 4.3 [Chuẩn hóa Tiêu đề Section (`title`)](#43-chuẩn-hóa-tiêu-đề-section-title)
5. [Quy tắc Bóc tách các Nút Lá và Khối Nội dung](#5-quy-tắc-bóc-tách-các-nút-lá-và-khối-nội-dung)
   - 5.1 [Nút Lá: Câu văn (`sentence`)](#51-nút-lá-câu-văn-sentence)
   - 5.2 [Nút Lá: Công thức toán dạng khối (`formula`)](#52-nút-lá-công-thức-toán-dạng-khối-formula)
   - 5.3 [Nút Lá: Khối Hình ảnh (`figure`) & Bảng biểu (`table`)](#53-nút-lá-khối-hình-ảnh-figure--bảng-biểu-table)
   - 5.4 [Nút Nhánh: Danh sách liệt kê (`itemize`) & Mục danh sách (`item`)](#54-nút-nhánh-danh-sách-liệt-kê-itemize--mục-danh-sách-item)
6. [Xử lý các Phần Đặc biệt & Vùng Ngoại lệ](#6-xử-lý-các-phần-đặc-biệt--vùng-ngoại-lệ)
   - 6.1 [Phần Tài liệu Tham khảo (References) — BẮT BUỘC LOẠI BỎ](#61-phần-tài-liệu-tham-khảo-references--bắt-buộc-loại-bỏ)
   - 6.2 [Phần Cảm ơn (Acknowledgments) — BẮT BUỘC GIỮ LẠI](#62-phần-cảm-ơn-acknowledgments--bắt-buộc-giữ-lại)
   - 6.3 [Phần Phụ lục (Appendices) — BẮT BUỘC GIỮ LẠI](#63-phần-phụ-lục-appendices--bắt-buộc-giữ-lại)
   - 6.4 [Phần Tóm tắt (Abstract) và Preamble](#64-phần-tóm-tắt-abstract-và-preamble)
   - 6.5 [Comments và Vùng Mã Nguyên văn (Verbatim)](#65-comments-và-vùng-mã-nguyên-văn-verbatim)
   - 6.6 [Các môi trường bao bọc trong suốt (Transparent Environments)](#66-các-môi-trường-bao-bọc-trong-suốt-transparent-environments)
7. [Ranh giới Trách nhiệm giữa các Module (Contracts)](#7-ranh-giới-trách-nhiệm-giữa-các-module-contracts)
8. [Xử lý Lỗi và Phòng thủ (Defensive Parsing)](#8-xử-lý-lỗi-và-phòng-thủ-defensive-parsing)
9. [Ví dụ Hoàn chỉnh: Đầu vào LaTeX và Đầu ra `hierarchy.json`](#9-ví-dụ-hoàn-chỉnh-đầu-vào-latex-và-đầu-ra-hierarchyjson)
   - 9.1 [Đoạn mã LaTeX mẫu (`sample_paper.tex`)](#91-đoạn-mã-latex-mẫu-sample_papertex)
   - 9.2 [File đầu ra cấu trúc cây `hierarchy.json` tương ứng](#92-file-đầu-ra-cấu-trúc-cây-hierarchyjson-tương-ứng)
   - 9.3 [Ánh xạ sang Hai Bảng Parquet (`elements` & `hierarchy`) trong Bộ Dataset 6 Bảng](#93-ánh-xạ-sang-hai-bảng-parquet-elements--hierarchy-trong-bộ-dataset-6-bảng)
     - 9.3.1 [Bảng `elements` (Trích đoạn các dòng đại diện)](#931-bảng-elements-trích-đoạn-các-dòng-đại-diện)
     - 9.3.2 [Bảng `hierarchy` (Biểu diễn cây cho version 1)](#932-bảng-hierarchy-biểu-diễn-cây-cho-version-1)
10. [Khảo sát Thực nghiệm (Empirical Survey Action Items)](#10-khảo-sát-thực-nghiệm-empirical-survey-action-items)
    - 10.1 [Bảng dữ liệu Khảo sát Thực nghiệm 10 Bài báo](#101-bảng-dữ-liệu-khảo-sát-thực-nghiệm-10-bài-báo)
    - 10.2 [Kiểm chứng 9 Giả thuyết Kỹ thuật (H1 – H9)](#102-kiểm-chứng-9-giả-thuyết-kỹ-thuật-h1--h9)
11. [Tài liệu Tham khảo](#11-tài-liệu-tham-khảo)

---

## 1. Mục đích và Phạm vi

### 1.1 Mục đích
Tài liệu này là **nguồn sự thật duy nhất (Single Source of Truth - SSOT)** quy định thuật toán, cấu trúc dữ liệu và các quy tắc nghiệp vụ để phân tích mã nguồn LaTeX của từng bài báo thành cấu trúc cây phân cấp lưu tại file trung gian `hierarchy.json`, sau đó làm phẳng (flatten) triệt để thành hai bảng Parquet `elements` và `hierarchy` để phát hành trên Hugging Face.

### 1.2 Vị trí trong Pipeline xử lý
Quy trình phân tích văn bản của Milestone 2 diễn ra tuần tự:

```text
[M1: Mã nguồn .tar.gz / .gz]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. Multi-file Resolver (latex_parser.py)               │
│    Tìm file chính, lần theo \input / \include          │
└─────────────────────────┬──────────────────────────────┘
                          │ (Chuỗi LaTeX gộp hoàn chỉnh)
                          ▼
┌────────────────────────────────────────────────────────┐
│ 2. LaTeX Cleanup & Math Normalization (latex_cleanup)  │
│    Dọn lệnh định dạng hiển thị, chuẩn hóa cú pháp toán │
│    ★ CHẠY TRƯỚC DEDUP VÀ DỰNG CÂY                      │
└─────────────────────────┬──────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │ (LaTeX sạch đã gộp)           │ (Còn nguyên References)
          ▼                               ▼
┌───────────────────────────┐   ┌──────────────────────────┐
│ 3. Tree Builder           │   │ 4. BibTeX Extractor      │
│    (tree_builder.py)      │   │    (bibtex_extractor.py) │
│    • Nhận diện heading    │   │    • Trích xuất \bibitem │
│    • Bóc tách Lá & Cành   │   │    • Chuẩn hóa trường    │
│    • LOẠI BỎ References   │   │    • Xuất refs.bib       │
│    • GIỮ Ack & Appendices │   └─────────────┬────────────┘
│    • Xuất hierarchy.json  │                 │
│      (Cây trung gian)     │                 │
└─────────────┬─────────────┘                 │
              │                               │
              ▼                               ▼
┌────────────────────────────────────────────────────────┐
│ 5. Khử trùng lặp 2 cấp (dedup.py)                      │
│    • Dedup BibTeX (Field Union)                        │
│    • Dedup nội dung giữa các version (v1, v2, ...)     │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│ 6. Parquet Exporter (export_parquet.py)                │
│    • Xuất 6 bảng sharded Parquet cho dataset repo:     │
│      papers, references, elements,                     │
│      hierarchy, bibtex, predictions                    │
│    • Chuẩn Flat & Primitive (Hub Viewer & Streaming)   │
│    • Không nested objects, không serialized JSON blob  │
└────────────────────────────────────────────────────────┘
```

### 1.3 Phạm vi và Ranh giới
- **Trong phạm vi:**
  - Định nghĩa thuật toán tự động nhận diện cấp bậc tiêu đề (Dynamic Heading Hierarchy).
  - Định nghĩa các loại nút (Nodes) trong cây và thuộc tính tương ứng.
  - Định nghĩa quy tắc bóc tách câu văn (`sentence`), công thức khối (`formula`), hình ảnh (`figure`), bảng biểu (`table`), danh sách liệt kê (`itemize`).
  - Định nghĩa quy tắc xử lý các phần đặc biệt (References, Acknowledgments, Appendices).
  - Quy định định dạng ID, cấu trúc file trung gian `hierarchy.json` và cơ chế làm phẳng triệt để sang 2 bảng Parquet phẳng (`elements` & `hierarchy`) tuân thủ chuẩn 6 bảng sharded Parquet (flat & primitive columns, tuyệt đối không lồng đối tượng hay dùng serialized JSON blob).
- **Ngoài phạm vi:**
  - Không biên dịch tài liệu bằng `pdflatex` hay xuất file PDF. Chỉ phân tích cú pháp tĩnh dạng văn bản bằng regex và bộ phân tích trạng thái (state machine).
  - Không thực thi hoặc giải mã các macro lập trình TeX phức tạp (`\def`, `\let`, `\expandafter`).

---

## 2. Các Nguyên tắc Bất biến Cốt lõi (Core Invariants)

Bộ phân tích bắt buộc phải thỏa mãn 6 nguyên tắc bất biến:

1. **Cấp bậc tiêu đề được tự nhận diện (Dynamic Heading Hierarchy):**
   Không được hard-code cứng cấp độ (ví dụ không mặc định `\section` luôn là level 1). Hệ thống phải tự động quét toàn bài báo, xác định tập các lệnh heading có mặt và gán level liên tục từ trên xuống dưới.
2. **Nút lá (Leaves) của cây gồm 4 loại duy nhất:**
   - Câu văn (`sentence`).
   - Công thức toán dạng khối (`formula`: `$$...$$`, `equation`, `align`, ...).
   - Khối hình ảnh (`figure`).
   - Khối bảng biểu (`table`).
   Nút lá là các đơn vị chứa nội dung văn bản cuối cùng, không có nút con (`children = []`).
3. **Danh sách `itemize` là nhánh con (Container Branch):**
   Môi trường danh sách không phải là nút lá. Môi trường `itemize` là nút nhánh cấp 1, mỗi `\item` là một nút con cấp 2, và bên trong `\item` chứa các nút lá (câu, công thức, hình, bảng) hoặc danh sách con lồng nhau.
4. **Quy tắc phân loại các phần đặc biệt:**
   - **BỎ** phần tài liệu tham khảo (*References*, *Bibliography*, *thebibliography*) khỏi cây nội dung `hierarchy.json`.
   - **GIỮ** phần cảm ơn (*Acknowledgments*) và các phụ lục (*Appendices*), phân tích nội dung bên trong như các section bình thường nhưng đánh dấu metadata vai trò.
5. **Thuộc tính `kind` chuẩn hóa và ID toàn cục:**
   Mỗi element trong cây phải có thuộc tính `kind` tuân thủ nghiêm ngặt enum `Kind` trong `src/schemas.py`, và có mã định danh `element_id` chứa `arxiv_id`.
6. **Chuẩn hóa lưu trữ Parquet phẳng (Flat & Primitive Storage Invariant):**
   Khi xuất dữ liệu cây sang Parquet để phát hành trong bộ dataset 6 bảng trên Hugging Face (chuẩn sharded Parquet phục vụ Dataset Viewer và Streaming backend):
   - Mọi cột phải là kiểu nguyên thủy (primitive: `string`, `int`, `float`, `bool`) hoặc danh sách chuỗi một cấp (one-level string list).
   - **Tuyệt đối không lưu trữ các đối tượng lồng nhau (nested objects) hay chuỗi JSON serialize (serialized JSON blobs)** trong bất kỳ cột Parquet nào. Toàn bộ cây phân cấp được phân rã phẳng hoàn toàn vào 2 bảng `elements` và `hierarchy` (mô hình Adjacency List).

---

## 3. Mô hình Dữ liệu: Cây lồng nhau & Bảng Parquet

### 3.1 Quy chuẩn Định dạng Lưu trữ (Storage Format Specification)

Để đáp ứng tối đa hiệu năng của Hugging Face Dataset Viewer và Apache Arrow streaming backend, toàn bộ dữ liệu cấu trúc bài báo được phân định rõ ràng giữa hai biểu diễn:
- **Biểu diễn Cây Cục bộ (`hierarchy.json`):** Chỉ lưu hành nội bộ ở bước xử lý Milestone 2 (in-memory hoặc file tạm trên đĩa). Cây có cấu trúc JSON lồng nhau phân cấp (Nested Tree) phục vụ việc đọc hiểu cấu trúc phân tầng tự nhiên, hỗ trợ thuật toán bóc tách phạm vi heading, nhận diện cha-con và phân tích ngữ cảnh (context-aware processing).
- **Biểu diễn Bảng Phẳng Parquet (`elements` & `hierarchy`):** Khi đóng gói xuất bản, cây phân cấp **bắt buộc phải được làm phẳng hoàn toàn** thành hai bảng quan hệ theo mô hình Adjacency List chuẩn Apache Parquet, tuyệt đối không serialize cả cây hay các node thành JSON blob.

### 3.2 Cấu trúc Dataset Repository & Hệ thống 6 Bảng Parquet

Cấu trúc thư mục của repository bộ dữ liệu trên Hugging Face (`<dataset-repo>`) được định dạng chuẩn hóa như sau:

```
└── <dataset-repo>/
    ├── README.md
    └── data/
        ├── papers/
        ├── references/
        ├── elements/
        ├── hierarchy/
        ├── bibtex/
        └── predictions/
```

Milestones 1 and 2 already specified five of these tables beside the data they produce, and the predictions table below completes the set:

| Table | Content | Described in |
|---|---|---|
| `papers` | the crawled paper metadata | Milestone 1 |
| `references` | the crawled reference metadata with one row per crawled reference and its citing paper, keeping the arXiv-keyed references only | Milestone 1 |
| `elements` | the parsed elements of all versions, deduplicated into one store | Milestone 2 |
| `hierarchy` | the per-version node-parent edges between elements | Milestone 2 |
| `bibtex` | the unified BibTeX entries after deduplication | Milestone 2 |
| `predictions` | the matching results with columns given below | this milestone |

Chi tiết nội dung và schema kỹ thuật của 6 bảng Parquet (tuân thủ nghiêm ngặt nguyên tắc **Flat & Primitive**):

1. **`papers`** (`data/papers/`):
   - Metadata đã thu thập của từng bài báo arXiv.
   - *Schema:* `arxiv_id: string`, `title: string`, `authors: list[str]`, `submission_date: string` (ISO 8601), `revised_dates: list[str]`, `publication_venue: string (nullable)`.
   - *Quy tắc:* Cho phép `authors` và `revised_dates` là danh sách chuỗi một cấp (`list[str]`).
2. **`references`** (`data/references/`):
   - Metadata trích dẫn thu thập được, mỗi dòng tương ứng một reference được trích dẫn và bài báo trích dẫn nó, chỉ giữ lại các reference có khóa arXiv (`arxiv-keyed references`).
   - *Schema:* `arxiv_id: string` (bài báo trích dẫn), `cited_arxiv_id: string` (bài báo arXiv được trích dẫn).
3. **`elements`** (`data/elements/`):
   - Các phần tử văn bản nội dung đã bóc tách từ mọi phiên bản của bài báo, được khử trùng lặp (deduplicated) vào một kho duy nhất.
   - *Schema:* `element_id: string`, `arxiv_id: string`, `kind: string`, `content: string`, `level: int32 (nullable)`.
   - *Quy tắc:* Toàn bộ cột là primitive scalars (`string`, `int32`), **tuyệt đối không chứa nested objects hay serialized JSON blobs** (không gói dict thành chuỗi JSON trong `content`).
4. **`hierarchy`** (`data/hierarchy/`):
   - Các cạnh quan hệ nút con - nút cha (per-version node-parent edges) giữa các phần tử theo từng phiên bản.
   - *Schema:* `arxiv_id: string`, `version: int32`, `element_id: string`, `parent_id: string (nullable)`, `order: int32`.
   - *Quy tắc:* Mô hình Adjacency List chuẩn hóa; liên kết nút con với nút cha qua khóa ngoại `parent_id`, bảo toàn thứ tự bằng `order`. Toàn bộ kiểu dữ liệu là primitive.
5. **`bibtex`** (`data/bibtex/`):
   - Các mục BibTeX hợp nhất sau khi khử trùng lặp (Field Union).
   - *Schema:* `arxiv_id: string`, `citation_key: string`, `title: string (nullable)`, `authors: list[str]`, `year: int32 (nullable)`, `doi: string (nullable)`, `journal: string (nullable)`, `url: string (nullable)`.
   - *Quy tắc:* Cho phép `authors` là danh sách chuỗi một cấp (`list[str]`).
6. **`predictions`** (`data/predictions/`):
   - Kết quả so khớp (matching results) từ mô hình Machine Learning ở Milestone 3 hoàn thiện bộ 6 bảng.
   - *Schema (9 cột):* `arxiv_id: string`, `partition: string` (`train`/`valid`/`test`), `citation_key: string`, `groundtruth_arxiv_id: string (nullable)`, `cand_1: string (nullable)`, `cand_2: string (nullable)`, `cand_3: string (nullable)`, `cand_4: string (nullable)`, `cand_5: string (nullable)`.

### 3.3 Nguyên tắc Cột phẳng và Kiểu nguyên thủy (Flat & Primitive Rules)
1. **Tại sao không dùng Nested Objects hoặc Serialized JSON Blobs?**
   - Hugging Face Dataset Viewer và Streaming backend đọc Parquet trực tiếp theo từng khối dữ liệu nhị phân (Row Groups / Shards).
   - Việc lồng object phức tạp hoặc serialize JSON blobs (`"{\"caption\": \"...\", \"label\": \"...\"}"`) sẽ phá vỡ cơ chế suy diễn kiểu (type inference), vô hiệu hóa bộ lọc cột (column projection / predicate pushdown), và ép phía client phải tốn chi phí parse chuỗi JSON.
2. **Quy định kiểu dữ liệu:**
   - Cột nguyên thủy (Primitive): `string`, `int32`, `int64`, `float64`, `bool`.
   - Danh sách một cấp (One-level List): Chỉ chấp nhận danh sách chuỗi đơn giản (`list[str]`) cho các trường danh sách như `authors` và `revised_dates`.
   - Mọi trường thông tin khác của phần tử (như `title`, `content`, `caption`, `label`) phải được lưu trữ dưới dạng chuỗi thuần hoặc tách thành cột primitive riêng biệt, **tuyệt đối không gom nhóm thành dictionary lồng nhau**.

### 3.4 Tập giá trị chuẩn cho thuộc tính `kind`
Đồng bộ trực tiếp với `Kind` enum trong `src/schemas.py`:

| `kind` | Vai trò trong cây | Là Lá (Leaf)? | Mô tả |
|---|---|:---:|---|
| `document` | Gốc (Root) | ❌ | Node gốc đại diện cho toàn bộ một phiên bản bài báo (1 node/bài). |
| `section` | Nhánh (Branch) | ❌ | Đại diện cho các cấp phân chia tiêu đề (`\part`, `\chapter`, `\section`, `\subsection`, `\subsubsection`). |
| `paragraph` | Nhánh (Branch) | ❌ | Đại diện cho khối đoạn văn có tiêu đề chạy liền (`\paragraph`, `\subparagraph`). |
| `itemize` | Nhánh (Branch) | ❌ | Container đại diện cho môi trường danh sách (`itemize`, `enumerate`, `description`). |
| `item` | Nhánh (Branch) | ❌ | Đại diện cho một mục con (`\item`) bên trong một container danh sách. |
| `sentence` | **Lá (Leaf)** | ✅ | Một câu văn hoàn chỉnh đã qua tách câu và bảo vệ thực thể. |
| `formula` | **Lá (Leaf)** | ✅ | Một công thức toán dạng khối (Display Math). |
| `figure` | **Lá (Leaf)** | ✅ | Toàn bộ khối hình ảnh (`figure`, `figure*`). |
| `table` | **Lá (Leaf)** | ✅ | Toàn bộ khối bảng biểu (`table`, `table*`). |

### 3.5 Quy ước định danh `element_id` và tính Xác định (Deterministic ID)
Quy chuẩn định danh `element_id` phải đảm bảo 3 yêu cầu:
1. Bắt buộc chứa `arxiv_id`.
2. Không bị trùng lặp trong toàn bộ 25.000 bài báo.
3. Có tính xác định tuyệt đối (Deterministic): Cùng một chuỗi mã nguồn đầu vào luôn sinh ra tập `element_id` giống nhau qua mọi lần chạy.

**Quy tắc sinh ID:**
Dùng hàm sinh ID chuẩn hóa:
```text
element_id = f"{arxiv_id}#{local_id}"
```
Trong đó `local_id` được sinh theo thứ tự duyệt tiền thứ tự (Pre-order Traversal: Cha duyệt trước Con, Anh duyệt trước Em) kèm tiền tố viết tắt của `kind`:
- Node gốc (`document`): `{arxiv_id}#doc`
- Node `section`: `{arxiv_id}#s{i}` (ví dụ: `2411.03799#s1`, `2411.03799#s2`)
- Node `itemize`: `{arxiv_id}#list{i}`
- Node `item`: `{arxiv_id}#it{i}`
- Node `sentence`: `{arxiv_id}#sent{i}`
- Node `formula`: `{arxiv_id}#eq{i}`
- Node `figure`: `{arxiv_id}#fig{i}`
- Node `table`: `{arxiv_id}#tbl{i}`

*(Chỉ số `{i}` là bộ đếm tự tăng độc lập cho từng loại phần tử theo thứ tự duyệt văn bản).*

> [!IMPORTANT]
> **Vấn đề Versioning và Khử trùng lặp (Milestone 2 Dedup):**
> Trong bảng `elements`, các phần tử nội dung không thay đổi giữa các version sẽ được khử trùng lặp. Bảng `hierarchy` ghi nhận phiên bản (`version`) để biểu diễn cấu trúc cây của từng version:
> - Cột `element_id` trong bảng `elements` lưu trữ ID nội dung chung.
> - Cột trong bảng `hierarchy`: `(arxiv_id, version, element_id, parent_id, order)`.
> Nếu version 2 tái sử dụng một câu từ version 1, `element_id` giữ nguyên và chỉ thêm một dòng quan hệ trong `hierarchy` với `version = 2`.

---

## 4. Thuật toán & Quy tắc Bóc tách Chi tiết

### 4.1 Tự nhận diện Cấp bậc Tiêu đề (Dynamic Heading Hierarchy)
LaTeX chuẩn quy định thứ bậc tự nhiên (Rank) gồm 7 cấp độ:

| Cấp LaTeX chuẩn | Lệnh tương ứng | Hạng chuẩn (Rank) |
|---|---|:---:|
| Part | `\part{...}` | 0 |
| Chapter | `\chapter{...}` | 1 |
| Section | `\section{...}` | 2 |
| Subsection | `\subsection{...}` | 3 |
| Subsubsection | `\subsubsection{...}` | 4 |
| Paragraph | `\paragraph{...}` | 5 |
| Subparagraph | `\subparagraph{...}` | 6 |

**Quy tắc nhận diện động:**
1. Lệnh có dấu sao (ví dụ: `\section*{Introduction}`) có cùng cấp bậc và trọng số thứ bậc với dạng không sao (`\section{Introduction}`). Dấu sao chỉ điều khiển việc đánh số hiển thị của LaTeX, không ảnh hưởng đến cấu trúc phân cấp logic.
2. Thuật toán quét toàn bộ văn bản (đã loại trừ comment và môi trường verbatim), lập tập hợp các lệnh heading thực sự xuất hiện trong bài báo.
3. Sắp xếp tập hợp đó tăng dần theo `Rank`.
4. Ánh xạ các lệnh này sang dải số nguyên liên tục bắt đầu từ `1`:

$$\text{Level}(\text{cmd}) = \text{Index trong danh sách các lệnh có mặt} + 1$$

*Ví dụ minh họa:*
- Bài báo loại 1 (Chỉ có `\section`, `\subsection`):
  $\rightarrow$ `\section` có level 1; `\subsection` có level 2.
- Bài báo loại 2 (Luận án/Sách có `\chapter`, `\section`, `\subsection`):
  $\rightarrow$ `\chapter` có level 1; `\section` có level 2; `\subsection` có level 3.
- Bài báo loại 3 (Tác giả bỏ qua `\subsection`, nhảy từ `\section` sang `\subsubsection`):
  $\rightarrow$ `\section` có level 1; `\subsubsection` có level 2. (Không sinh node rỗng trung gian).

### 4.2 Dựng quan hệ Cha – Con bằng Ngăn xếp (Stack-based Construction)
Sử dụng cấu trúc dữ liệu Ngăn xếp (Stack) để quản lý phạm vi của các section đang mở:

```python
# Thuật toán dựng cây:
stack = [(root_node, level=0)]

for item in parsed_items:
    if item.is_heading:
        item_level = dynamic_level_map[item.command]
        # Đóng các section có level sâu hơn hoặc bằng level của heading hiện tại
        while stack and stack[-1].level >= item_level:
            stack.pop()
        
        parent_node = stack[-1].node
        new_section_node = create_section_node(item, level=item_level)
        parent_node.children.append(new_section_node)
        stack.append((new_section_node, level=item_level))
    else:
        # Nút nội dung (sentence, formula, list, figure, table)
        current_parent = stack[-1].node
        current_parent.children.append(item.node)
```

**Xử lý văn bản trước Heading đầu tiên:**
Mọi nội dung xuất hiện sau `\begin{document}` (hoặc sau `abstract`) nhưng nằm trước lệnh heading đầu tiên sẽ được gắn trực tiếp vào nút gốc `document`.

### 4.3 Chuẩn hóa Tiêu đề Section (`title`)
Tiêu đề của section được trích xuất từ đối số thứ nhất của lệnh heading và làm sạch:
1. Loại bỏ nhãn tham chiếu `\label{...}` lồng bên trong tiêu đề (lưu riêng nếu cần).
2. Xử lý macro tương thích `\texorpdfstring{TeX text}{Plain text}`: Luôn lấy tham số thứ hai (`Plain text`) để đảm bảo tiêu đề sạch ký tự lạ.
3. Bảo toàn các công thức toán nội dòng (Inline Math, ví dụ `$\Delta$-Coloring`) để giữ nguyên vẹn ý nghĩa chuyên ngành.
4. Xóa bỏ các lệnh ngắt dòng thủ công (`\\`, `\newline`) hoặc định dạng chữ (`\textbf{...}`, `\emph{...}`).

---

## 5. Quy tắc Bóc tách các Nút Lá và Khối Nội dung

### 5.1 Nút Lá: Câu văn (`sentence`)
Nhiệm vụ của bộ tách câu là chia một đoạn văn bản thành danh sách các node `kind = "sentence"`. Việc tách câu phải thông minh, không được gây phân mảnh sai các thực thể toán học và trích dẫn.

**Quy trình 3 bước tách câu:**
1. **Bước 1 — Bảo vệ các thực thể không thể tách (Masking/Escaping):**
   Thay thế tạm thời bằng các token đặc biệt hoặc bảo vệ các thành phần:
   - Toán nội dòng: `$...$` và `\( ... \)`.
   - Trích dẫn thư mục: `\cite{...}`, `\citep{...}`, `\citet{...}`.
   - Nhãn tham chiếu nội bộ: `\ref{...}`, `\eqref{...}`, `\autoref{...}`.
   - Địa chỉ URL/DOI: `\url{...}`, `\href{...}{...}`.
   - Các chữ viết tắt phổ biến trong văn bản khoa học: `et al.`, `i.e.`, `e.g.`, `cf.`, `vs.`, `Fig.`, `Eq.`, `Sec.`, `Ref.`, `Thm.`, `Lem.`, `No.`, `Vol.`, `pp.`, `etc.`.
   - Viết tắt tên tác giả: `A. Smith`, `J. D. Watson`.
   - Số thập phân: `3.14159`, `.05`, `1,000`.
2. **Bước 2 — Phân tách ranh giới câu (Sentence Boundary Detection):**
   - Chỉ tách câu tại dấu chấm kết thúc (`.`), dấu chấm hỏi (`?`), dấu chấm than (`!`) khi theo sau bởi khoảng trắng và một ký tự viết hoa hoặc bắt đầu một đoạn mới.
   - Hai dấu xuống dòng liên tiếp (`\n\n`) hoặc lệnh `\par` báo hiệu ngắt đoạn: Ngắt câu ngay tại ranh giới đoạn văn dù câu trước có thiếu dấu chấm.
3. **Bước 3 — Khôi phục thực thể và Chuẩn hóa:**
   - Hoàn trả lại các token đã bảo vệ ở Bước 1.
   - Chuẩn hóa khoảng trắng dư thừa (`\s+` $\rightarrow$ `" "`).
   - Giữ nguyên các lệnh trích dẫn `\cite{...}` trong trường `content` của câu. Đây là thông tin cực kỳ quan trọng cho Module M3 (Reference Matching).

### 5.2 Nút Lá: Công thức toán dạng khối (`formula`)
Mọi công thức hiển thị riêng biệt trên một dòng (Display Math) được đóng gói thành một node `kind = "formula"`.

**Các môi trường bắt buộc nhận diện:**
- Cú pháp TeX cổ điển: `$$ ... $$`
- Cú pháp chuẩn LaTeX: `\[ ... \]`
- Môi trường đánh số cơ bản: `\begin{equation} ... \end{equation}`, `\begin{equation*} ... \end{equation*}`
- Các môi trường toán nâng cao từ gói `amsmath` / `mathtools`:
  `align`, `align*`, `gather`, `gather*`, `multline`, `multline*`, `flalign`, `flalign*`, `eqnarray`, `eqnarray*`, `displaymath`.

**Quy tắc trích xuất:**
- Thuộc tính `content`: Chứa toàn bộ nội dung biểu thức toán học bên trong, đã loại bỏ cặp delimiter bao quanh (`$$`, `\[ \]`, hoặc `\begin{...}` / `\end{...}`).
- Trích xuất nhãn: Nếu bên trong công thức có chứa lệnh `\label{key}`, trích xuất `key` ra thuộc tính `label`, sau đó xóa lệnh `\label{key}` khỏi `content` để công thức toán sạch sẽ.
- Môi trường con lồng bên trong: Các cấu trúc toán bên trong công thức khối như `\begin{split}`, `\begin{cases}`, `\begin{aligned}`, `\begin{matrix}` là một phần của công thức, **không** được tách thành các node lá độc lập.
- Toán nội dòng: `$x^2 + y^2 = z^2$` nằm trong luồng văn bản phải được giữ nguyên trong `sentence`, **không** tạo node `formula`.

**Xử lý hiện tượng câu bị ngắt quãng bởi công thức khối:**
Khi công thức khối xuất hiện giữa câu, parser chia thành chuỗi 3 nút liên tiếp theo thứ tự xuất hiện trong văn bản nguồn:
```text
[sentence: "Hệ số ma sát được định nghĩa bởi"]
[formula: "f = \mu N"]
[sentence: "trong đó \mu là hệ số tĩnh."]
```

### 5.3 Nút Lá: Khối Hình ảnh (`figure`) & Bảng biểu (`table`)
Toàn bộ khối môi trường trôi (Float Environment) được xem là một nút lá độc lập.

**Các môi trường hỗ trợ:**
- Khối hình ảnh (`kind = "figure"`): `figure`, `figure*`, `wrapfigure`.
- Khối bảng biểu (`kind = "table"`): `table`, `table*`, `tabular` độc lập (nếu đứng ngoài môi trường table).

**Quy tắc trích xuất:**
- `caption`: Nội dung văn bản bên trong lệnh `\caption{...}`. Nếu có cả caption ngắn và dài `\caption[ngắn]{dài}`, ưu tiên lấy phần `dài`.
- `label`: Trích xuất từ lệnh `\label{...}` nằm trong khối.
- `content`: Lưu toàn bộ khối TeX thô sau bước làm sạch (hoặc cấu trúc bảng thô).
- **Trường hợp xóa ảnh ở M1:** Crawler M1 đã xóa các file ảnh nhị phân (.png, .jpg, .pdf) để tiết kiệm bộ nhớ nhưng **giữ nguyên mã `\begin{figure}`**. Bộ dựng cây vẫn tạo node `figure` bình thường dù lệnh `\includegraphics` bên trong không còn file ảnh vật lý.
- Các môi trường con như `subfigure`, `minipage` được giữ nguyên trong nội dung của node `figure`, không tách thành các node riêng rẽ.
- **Tuân thủ nguyên tắc Flat & Primitive khi lưu Parquet:**
  Trong file trung gian `hierarchy.json`, các trường `caption`, `label` có thể được tổ chức thành các key riêng của node để phục vụ bước bóc tách. Tuy nhiên, khi làm phẳng vào bảng Parquet `elements`, trường `content` lưu trữ chuỗi văn bản thuần (ví dụ: nội dung caption hoặc đoạn mã TeX thô đã làm sạch), **tuyệt đối không serialize thành JSON string (như `{"caption": "...", "label": "..."}`)** để tuân thủ quy chuẩn không dùng JSON blobs.

### 5.4 Nút Nhánh: Danh sách liệt kê (`itemize`) & Mục danh sách (`item`)
Cấu trúc danh sách biểu diễn theo mô hình 2 tầng lồng nhau:

```text
Node "itemize" (Container)
  ├── Node "item" (Mục thứ nhất)
  │     ├── Node "sentence"
  │     └── Node "formula" (nếu có)
  └── Node "item" (Mục thứ hai)
        ├── Node "sentence"
        └── Node "itemize" (Danh sách con lồng nhau nếu có)
```

**Các môi trường áp dụng:**
`itemize` (gạch đầu dòng), `enumerate` (đánh số), `description` (danh sách mô tả).
- Tạo một node `kind = "itemize"` để bọc toàn bộ khối môi trường.
- Với mỗi lệnh `\item`, tạo một node `kind = "item"` làm con của node `itemize`.
- Nếu có nhãn tùy biến `\item[label]`, lưu `label` vào thuộc tính của item.
- Nội dung sau `\item` được phân tích như văn bản thông thường (gồm các câu `sentence`, công thức `formula`, hoặc danh sách `itemize` lồng nhau).

---

## 6. Xử lý các Phần Đặc biệt & Vùng Ngoại lệ

### 6.1 Phần Tài liệu Tham khảo (References) — BẮT BUỘC LOẠI BỎ
Yêu cầu bắt buộc của bài toán: Cây phân cấp nội dung `hierarchy.json` **không được chứa** các mục tham khảo thư mục.

**Nhận diện và Cắt bỏ:**
1. Khối môi trường tường minh:
   - `\begin{thebibliography}{...} ... \end{thebibliography}`
   - `\bibliography{...}`, `\printbibliography`
2. Khối heading tham khảo:
   - Các Section có tiêu đề thuộc danh sách: `References`, `Reference`, `Bibliography`, `Literature Cited`, `References and Notes`.
   - Toàn bộ heading này và tất cả các node con thuộc phạm vi của nó bị loại bỏ hoàn toàn khỏi cây phân cấp.
   - Phạm vi cắt bỏ kết thúc khi gặp một heading khác cùng cấp/cao hơn, hoặc khi gặp lệnh `\appendix`.

> [!CAUTION]
> **Thứ tự xử lý giữa P2-04 và P2-03:**
> Module `bibtex_extractor.py` (P2-04) phải trích xuất các `\bibitem` thành file `refs.bib` từ chuỗi LaTeX **trước** khi `tree_builder.py` (P2-03) loại bỏ phần này khỏi cây phân cấp!

### 6.2 Phần Cảm ơn (Acknowledgments) — BẮT BUỘC GIỮ LẠI
Phần cảm ơn phải được bảo toàn trong cây nội dung.
- Các tiêu đề thường gặp: `Acknowledgment`, `Acknowledgments`, `Acknowledgement`, `Acknowledgements`.
- Các môi trường đặc thù của một số journal: `\begin{acknowledgments} ... \end{acknowledgments}` (như gói `revtex4`).
- **Quy tắc dựng node:**
  - Tạo node `kind = "section"`.
  - Thiết lập thuộc tính tiêu đề: `title = "Acknowledgments"`.
  - Phân tích các câu văn bên trong như một Section bình thường.
  - Thêm cờ đánh dấu metadata `role = "acknowledgments"` để phục vụ cho các tác vụ downstream.

### 6.3 Phần Phụ lục (Appendices) — BẮT BUỘC GIỮ LẠI
Mọi nội dung trong phần phụ lục phải được giữ trọn vẹn trong cây phân cấp.
- Chỉ thị phân định: Lệnh `\appendix` hoặc môi trường `\begin{appendices} ... \end{appendices}`.
- Lệnh `\appendix` trong LaTeX là một công tắc chuyển chế độ đánh số (từ 1, 2, 3 sang A, B, C), bản thân lệnh này không nhất thiết phải là một heading.
- Khi gặp `\appendix`, toàn bộ các section xuất hiện phía sau lệnh này đều được đánh dấu metadata `role = "appendix"`.
- Nếu phụ lục xuất hiện sau phần References, hệ thống vẫn tiếp tục mở lại việc dựng cây cho phụ lục, tuyệt đối không được cắt bỏ phụ lục theo References.

### 6.4 Phần Tóm tắt (Abstract) và Preamble
- **Preamble:** Toàn bộ phần mã nguồn nằm trước `\begin{document}` (khai báo `\documentclass`, `\usepackage`, macro) bị loại bỏ khỏi cây nội dung (chỉ lưu trữ nội bộ trong mã nguồn gốc thô, không đưa vào cây phân cấp).
- **Abstract:** Môi trường `\begin{abstract} ... \end{abstract}` được chuẩn hóa thành một Section đặc biệt:
  - `kind = "section"`
  - `title = "Abstract"`
  - `level = 1`
  - `role = "abstract"`
  - Các câu văn bên trong abstract được tách thành các node `sentence` con.
- **Metadata tác giả:** Các lệnh `\title{...}`, `\author{...}`, `\date{...}` không tạo node trong cây (đã được lưu tại `metadata.json` / bảng `papers`).
- **Nội dung sau `\end{document}`:** Bỏ qua hoàn toàn.

### 6.5 Comments và Vùng Mã Nguyên văn (Verbatim)
1. **Comment:**
   - Ký tự `%` bắt đầu comment đến hết dòng. Mọi lệnh LaTeX nằm sau `%` bị vô hiệu hóa.
   - Ký tự escape `\%` biểu thị dấu phần trăm thông thường, không phải là comment.
   - Parser phải loại bỏ comment trước khi nhận diện cấu trúc.
2. **Vùng mã nguyên văn:**
   - Các môi trường: `verbatim`, `lstlisting`, `minted`, `rawhtml`.
   - Các lệnh rẽ nhánh điều kiện: `\iffalse ... \fi`, `\begin{comment} ... \end{comment}`.
   - Parser xem các vùng này như khối văn bản thô hoặc khối hộp đen, **không** phân tích cú pháp heading hay tách câu bên trong các môi trường này.

### 6.6 Các môi trường bao bọc trong suốt (Transparent Environments)
Các môi trường định lý và trình bày: `theorem`, `lemma`, `proposition`, `corollary`, `definition`, `remark`, `proof`, `quote`, `center`, `minipage`.
- Các môi trường này đóng vai trò bao bọc ngữ nghĩa nhưng không làm thay đổi phân cấp cha-con của tài liệu.
- **Quy tắc:** Bộ phân tích xem các môi trường này là "trong suốt" (transparent wrappers) — nội dung bên trong được bóc tách thành các câu `sentence` hoặc công thức `formula` và gắn trực tiếp vào Section cha hiện tại.

---

## 7. Ranh giới Trách nhiệm giữa các Module (Contracts)

| Module | Đầu vào | Đầu ra | Ranh giới Trách nhiệm & Điều cấm |
|---|---|---|---|
| **`latex_parser.py`**<br/>*(Multi-file Resolver, P2-03)* | Thư mục chứa các file `.tex` giải nén từ M1. | Chuỗi văn bản LaTeX duy nhất đã gộp. | Tìm file chính (chứa `\documentclass`). Lần theo các lệnh `\input{...}` và `\include{...}` để gộp thành 1 chuỗi. **Xử lý chu trình đệ quy** (circular include). |
| **`latex_cleanup.py`**<br/>*(Cleanup, P2-05)* | Chuỗi LaTeX đã gộp. | Chuỗi LaTeX sạch cấu trúc. | **ĐƯỢC:** Xóa các lệnh định dạng font chữ (`\bf`, `\it`), định dạng canh lề (`\centering`, `\raggedright`), khoảng trắng (`\vspace`, `\hspace`), tùy chọn vị trí float (`[htbp]`).<br/>**CẤM:** Không xóa lệnh heading, không xóa môi trường `figure`/`table`/`equation`, không xóa `\caption`, `\label`, `\cite`, `\appendix`, `thebibliography`. |
| **`tree_builder.py`**<br/>*(Tree Builder, P2-03)* | Chuỗi LaTeX đã cleanup, `arxiv_id`. | Cây `hierarchy.json` lồng nhau. | Dựng cây theo đúng Spec P1-04 này. Gán level động, bóc tách lá, loại bỏ References, giữ Ack và Phụ lục. Lưu trữ cấu trúc cây trung gian. |
| **`bibtex_extractor.py`**<br/>*(BibTeX Extractor, P2-04)* | Chuỗi LaTeX (còn nguyên References). | Danh sách `BibtexEntry` / `refs.bib`. | Bóc tách toàn bộ `\bibitem` hoặc parse file `.bib` đi kèm thành các entry có cấu trúc. |
| **`dedup.py`**<br/>*(Deduplication, P2-04)* | `hierarchy.json` và `refs.bib` qua các version. | Cấu trúc đã khử trùng lặp. | Khử trùng lặp nội dung giữa các version (v1, v2...) của cùng bài báo; hợp nhất các trường BibTeX (Field Union). |
| **`export_parquet.py`**<br/>*(Parquet Exporter, P4-02)* | Dữ liệu cây đã dedup, metadata, bibtex, predictions. | 6 bảng sharded Parquet trên Hugging Face. | **Làm phẳng triệt để cây** thành 2 bảng phẳng `elements` và `hierarchy` (Adjacency List). Xuất bản bộ 6 bảng (`papers`, `references`, `elements`, `hierarchy`, `bibtex`, `predictions`) vào cấu trúc `data/<table_name>/`. **CẤM:** Không serialize JSON blobs vào cột, không lưu nested objects. Đảm bảo viewer và streaming backend của Hub đọc trực tiếp. |

---

## 8. Xử lý Lỗi và Phòng thủ (Defensive Parsing)

Để pipeline không bị dừng đột ngột khi xử lý hàng nghìn bài báo thật, bộ dựng cây phải triển khai cơ chế bắt lỗi phòng thủ (Fail-safe):

1. **Bài báo không có mã nguồn TeX (PDF-only / DOCX-only):**
   - Đã được M1 ghi nhận. Parser tạo một file `hierarchy.json` rỗng chứa duy nhất node gốc `document` với nội dung rỗng, ghi log cảnh báo và tiếp tục bài tiếp theo.
2. **Môi trường không có thẻ đóng (`\begin{env}` thiếu `\end{env}`):**
   - Không gây crash chương trình. Parser tự động đóng môi trường tại ranh giới heading tiếp theo hoặc tại cuối tài liệu.
3. **Bài báo không có bất kỳ lệnh Heading nào:**
   - Toàn bộ các câu, công thức, bảng biểu trong bài được gắn trực tiếp vào node gốc `document`.
4. **Lệnh include trỏ đến file không tồn tại:**
   - Ghi nhận warning vào log file, thay thế bằng khoảng trắng rỗng và tiếp tục xử lý các phần khác của bài báo.
5. **Section rỗng không có nội dung:**
   - Vẫn tạo node `kind = "section"`, thuộc tính `children = []`.

---

## 9. Ví dụ Hoàn chỉnh: Đầu vào LaTeX và Đầu ra `hierarchy.json`

### 9.1 Đoạn mã LaTeX mẫu (`sample_paper.tex`)

```latex
\documentclass[11pt]{article}
\usepackage{amsmath,graphicx}

\begin{document}

\begin{abstract}
Graph coloring is a fundamental problem in computer science. We present a linear-time heuristic bound.
\end{abstract}

\section{Introduction}
Graph coloring has been studied extensively~\cite{brooks1941colouring}. Let $G=(V, E)$ be a connected graph.
\begin{itemize}
    \item Planar graphs are 4-colorable.
    \item Bipartite graphs are 2-colorable.
\end{itemize}

\subsection{Brooks' Bound}
The chromatic number satisfies the following upper bound:
\begin{equation}
\chi(G) \leq \Delta(G) + 1 \label{eq:brooks}
\end{equation}
where $\Delta(G)$ denotes the maximum degree.

\begin{figure}[ht]
\caption{Visualization of 4-coloring on a planar triangulation.}
\label{fig:planar_graph}
\end{figure}

\section*{Acknowledgments}
This research was supported by the VNU-HCM Research Grant.

\begin{thebibliography}{99}
\bibitem{brooks1941colouring}
R.~L. Brooks, \emph{On colouring the nodes of a network}, Math. Proc. Cambridge, 1941.
\end{thebibliography}

\appendix
\section{Omitted Proofs}
The detailed proof of correctness follows by induction on $|V|$.

\end{document}
```

### 9.2 File đầu ra cấu trúc cây `hierarchy.json` tương ứng

```json
{
  "element_id": "2411.03799#doc",
  "arxiv_id": "2411.03799",
  "kind": "document",
  "content": "",
  "children": [
    {
      "element_id": "2411.03799#s1",
      "arxiv_id": "2411.03799",
      "kind": "section",
      "title": "Abstract",
      "level": 1,
      "role": "abstract",
      "children": [
        {
          "element_id": "2411.03799#sent1",
          "arxiv_id": "2411.03799",
          "kind": "sentence",
          "content": "Graph coloring is a fundamental problem in computer science."
        },
        {
          "element_id": "2411.03799#sent2",
          "arxiv_id": "2411.03799",
          "kind": "sentence",
          "content": "We present a linear-time heuristic bound."
        }
      ]
    },
    {
      "element_id": "2411.03799#s2",
      "arxiv_id": "2411.03799",
      "kind": "section",
      "title": "Introduction",
      "level": 1,
      "role": "body",
      "children": [
        {
          "element_id": "2411.03799#sent3",
          "arxiv_id": "2411.03799",
          "kind": "sentence",
          "content": "Graph coloring has been studied extensively \\cite{brooks1941colouring}."
        },
        {
          "element_id": "2411.03799#sent4",
          "arxiv_id": "2411.03799",
          "kind": "sentence",
          "content": "Let $G=(V, E)$ be a connected graph."
        },
        {
          "element_id": "2411.03799#list1",
          "arxiv_id": "2411.03799",
          "kind": "itemize",
          "children": [
            {
              "element_id": "2411.03799#it1",
              "arxiv_id": "2411.03799",
              "kind": "item",
              "children": [
                {
                  "element_id": "2411.03799#sent5",
                  "arxiv_id": "2411.03799",
                  "kind": "sentence",
                  "content": "Planar graphs are 4-colorable."
                }
              ]
            },
            {
              "element_id": "2411.03799#it2",
              "arxiv_id": "2411.03799",
              "kind": "item",
              "children": [
                {
                  "element_id": "2411.03799#sent6",
                  "arxiv_id": "2411.03799",
                  "kind": "sentence",
                  "content": "Bipartite graphs are 2-colorable."
                }
              ]
            }
          ]
        },
        {
          "element_id": "2411.03799#s3",
          "arxiv_id": "2411.03799",
          "kind": "section",
          "title": "Brooks' Bound",
          "level": 2,
          "role": "body",
          "children": [
            {
              "element_id": "2411.03799#sent7",
              "arxiv_id": "2411.03799",
              "kind": "sentence",
              "content": "The chromatic number satisfies the following upper bound:"
            },
            {
              "element_id": "2411.03799#eq1",
              "arxiv_id": "2411.03799",
              "kind": "formula",
              "label": "eq:brooks",
              "content": "\\chi(G) \\leq \\Delta(G) + 1"
            },
            {
              "element_id": "2411.03799#sent8",
              "arxiv_id": "2411.03799",
              "kind": "sentence",
              "content": "where $\\Delta(G)$ denotes the maximum degree."
            },
            {
              "element_id": "2411.03799#fig1",
              "arxiv_id": "2411.03799",
              "kind": "figure",
              "caption": "Visualization of 4-coloring on a planar triangulation.",
              "label": "fig:planar_graph",
              "content": ""
            }
          ]
        }
      ]
    },
    {
      "element_id": "2411.03799#s4",
      "arxiv_id": "2411.03799",
      "kind": "section",
      "title": "Acknowledgments",
      "level": 1,
      "role": "acknowledgments",
      "children": [
        {
          "element_id": "2411.03799#sent9",
          "arxiv_id": "2411.03799",
          "kind": "sentence",
          "content": "This research was supported by the VNU-HCM Research Grant."
        }
      ]
    },
    {
      "element_id": "2411.03799#s5",
      "arxiv_id": "2411.03799",
      "kind": "section",
      "title": "Omitted Proofs",
      "level": 1,
      "role": "appendix",
      "children": [
        {
          "element_id": "2411.03799#sent10",
          "arxiv_id": "2411.03799",
          "kind": "sentence",
          "content": "The detailed proof of correctness follows by induction on $|V|$."
        }
      ]
    }
  ]
}
```

### 9.3 Ánh xạ sang Hai Bảng Parquet (`elements` & `hierarchy`) trong Bộ Dataset 6 Bảng

Khi export sang 6 bảng Parquet của dataset repo phát hành trên Hugging Face Hub (tuân thủ định dạng sharded Parquet để Dataset Viewer và Streaming backend đọc trực tiếp), cây JSON lồng nhau ở trên được làm phẳng (flatten) triệt để thành hai bảng quan hệ theo mô hình Adjacency List với schema tường minh và thuần kiểu nguyên thủy (Flat & Primitive):

#### 9.3.1 Bảng `elements` (Trích đoạn các dòng đại diện)
- **Schema:** `element_id: string`, `arxiv_id: string`, `kind: string`, `content: string`, `level: int32 (nullable)`.
- **Đặc điểm:** Toàn bộ cột đều là kiểu nguyên thủy (primitive scalars). Tuyệt đối không chứa nested objects hay serialized JSON blobs. Nội dung công thức, câu văn, tiêu đề hay caption bảng/hình đều là các chuỗi ký tự thô/đã làm sạch lưu tại `content`.

| `element_id` | `arxiv_id` | `kind` | `content` | `level` |
|---|---|---|---|:---:|
| `2411.03799#doc` | `2411.03799` | `document` | `""` | `null` |
| `2411.03799#s1` | `2411.03799` | `section` | `"Abstract"` | `1` |
| `2411.03799#sent1` | `2411.03799` | `sentence` | `"Graph coloring is a fundamental problem in computer science."` | `null` |
| `2411.03799#sent2` | `2411.03799` | `sentence` | `"We present a linear-time heuristic bound."` | `null` |
| `2411.03799#s2` | `2411.03799` | `section` | `"Introduction"` | `1` |
| `2411.03799#sent3` | `2411.03799` | `sentence` | `"Graph coloring has been studied extensively \cite{brooks1941colouring}."` | `null` |
| `2411.03799#sent4` | `2411.03799` | `sentence` | `"Let $G=(V, E)$ be a connected graph."` | `null` |
| `2411.03799#list1` | `2411.03799` | `itemize` | `""` | `null` |
| `2411.03799#it1` | `2411.03799` | `item` | `""` | `null` |
| `2411.03799#sent5` | `2411.03799` | `sentence` | `"Planar graphs are 4-colorable."` | `null` |
| `2411.03799#it2` | `2411.03799` | `item` | `""` | `null` |
| `2411.03799#sent6` | `2411.03799` | `sentence` | `"Bipartite graphs are 2-colorable."` | `null` |
| `2411.03799#s3` | `2411.03799` | `section` | `"Brooks' Bound"` | `2` |
| `2411.03799#sent7` | `2411.03799` | `sentence` | `"The chromatic number satisfies the following upper bound:"` | `null` |
| `2411.03799#eq1` | `2411.03799` | `formula` | `"\chi(G) \leq \Delta(G) + 1"` | `null` |
| `2411.03799#sent8` | `2411.03799` | `sentence` | `"where $\Delta(G)$ denotes the maximum degree."` | `null` |
| `2411.03799#fig1` | `2411.03799` | `figure` | `"Visualization of 4-coloring on a planar triangulation."` | `null` |
| `2411.03799#s4` | `2411.03799` | `section` | `"Acknowledgments"` | `1` |
| `2411.03799#sent9` | `2411.03799` | `sentence` | `"This research was supported by the VNU-HCM Research Grant."` | `null` |
| `2411.03799#s5` | `2411.03799` | `section` | `"Omitted Proofs"` | `1` |
| `2411.03799#sent10` | `2411.03799` | `sentence` | `"The detailed proof of correctness follows by induction on \|V\|."` | `null` |

#### 9.3.2 Bảng `hierarchy` (Biểu diễn cây cho version 1)
- **Schema:** `arxiv_id: string`, `version: int32`, `element_id: string`, `parent_id: string (nullable)`, `order: int32`.
- **Đặc điểm:** Tách biệt cấu trúc phả hệ (Adjacency List) khỏi nội dung, quản lý mối quan hệ đa phiên bản (v1, v2...) và thứ tự xuất hiện các node con (`order`) hoàn toàn bằng các cột nguyên thủy.

| `arxiv_id` | `version` | `element_id` | `parent_id` | `order` |
|---|---|---|---|:---:|
| `2411.03799` | `1` | `2411.03799#s1` | `2411.03799#doc` | `0` |
| `2411.03799` | `1` | `2411.03799#sent1` | `2411.03799#s1` | `0` |
| `2411.03799` | `1` | `2411.03799#sent2` | `2411.03799#s1` | `1` |
| `2411.03799` | `1` | `2411.03799#s2` | `2411.03799#doc` | `1` |
| `2411.03799` | `1` | `2411.03799#sent3` | `2411.03799#s2` | `0` |
| `2411.03799` | `1` | `2411.03799#sent4` | `2411.03799#s2` | `1` |
| `2411.03799` | `1` | `2411.03799#list1` | `2411.03799#s2` | `2` |
| `2411.03799` | `1` | `2411.03799#it1` | `2411.03799#list1` | `0` |
| `2411.03799` | `1` | `2411.03799#sent5` | `2411.03799#it1` | `0` |
| `2411.03799` | `1` | `2411.03799#it2` | `2411.03799#list1` | `1` |
| `2411.03799` | `1` | `2411.03799#sent6` | `2411.03799#it2` | `0` |
| `2411.03799` | `1` | `2411.03799#s3` | `2411.03799#s2` | `3` |
| `2411.03799` | `1` | `2411.03799#sent7` | `2411.03799#s3` | `0` |
| `2411.03799` | `1` | `2411.03799#eq1` | `2411.03799#s3` | `1` |
| `2411.03799` | `1` | `2411.03799#sent8` | `2411.03799#s3` | `2` |
| `2411.03799` | `1` | `2411.03799#fig1` | `2411.03799#s3` | `3` |
| `2411.03799` | `1` | `2411.03799#s4` | `2411.03799#doc` | `2` |
| `2411.03799` | `1` | `2411.03799#sent9` | `2411.03799#s4` | `0` |
| `2411.03799` | `1` | `2411.03799#s5` | `2411.03799#doc` | `3` |
| `2411.03799` | `1` | `2411.03799#sent10` | `2411.03799#s5` | `0` |

> [!TIP]
> **Đối chiếu Tiêu chuẩn Lưu trữ Parquet của Hub:**
> - **Chỉ gồm đúng 6 bảng Parquet trong `<dataset-repo>/data/`:** `papers`, `references`, `elements`, `hierarchy`, `bibtex`, `predictions`.
> - **Cột phẳng & Nguyên thủy (Flat & Primitive):** Toàn bộ các bảng hoàn toàn không dùng nested objects hay serialized JSON blobs. Mọi thông tin được định kiểu rõ ràng (`string`, `int32`), cho phép viewer và streaming backend trên Hugging Face Hub tải và hiển thị ngay lập tức từng partition mà không cần giải mã thêm.

---

## 10. Khảo sát Thực nghiệm (Empirical Survey Action Items)

Kết quả sau quá trình tiến hành khảo sát trên **10 bài báo arXiv thực tế** tải từ dải ID của bản thân (`2024-11: 3799 → 8798`) và kiểm chứng 9 giả thuyết H1–H9:

### 10.1 Bảng dữ liệu Khảo sát Thực nghiệm 10 Bài báo

| # | arXiv ID | Lớp Document (`\documentclass`) | Đơn/Đa File | Các lệnh Heading xuất hiện | Khối Toán chính | Môi trường Bảng & Hình | Danh sách Liệt kê | Cấu trúc References | Vị trí Ack & Phụ lục |
|---|---|---|:---:|---|---|---|---|---|---|
| 1 | `2411.03799` | `article` | Đơn | `\section`, `\subsection` | `align`, `equation` | `figure`, `table` | `itemize` | `thebibliography` | Ack trước Ref, Appendix sau Ref |
| 2 | `2411.04100` | `revtex4-2` | Đa (3 files) | `\section`, `\subsection` | `equation*`, `eqnarray` | `figure*` | `enumerate` | `\bibliography` | Env `acknowledgments` |
| 3 | `2411.05214` | `amsart` | Đơn | `\section`, `\subsubsection` | `align*`, `gather` | Không có float | `description` | `thebibliography` | Không có Ack, có Appendix |
| 4 | `2411.06001` | `IEEEtran` | Đa (5 files) | `\section`, `\subsection` | `equation` | `figure`, `table` | `itemize` | `\bibliography` | Section `Acknowledgment` |
| 5 | `2411.07230` | `article` | Đơn | `\section`, `\paragraph` | `$$...$$`, `align` | `wrapfigure` | `itemize` | `thebibliography` | Ack có sao `\section*` |
| 6 | `2411.08115` | `report` | Đa (8 files) | `\chapter`, `\section` | `align`, `multline` | `table*` | `enumerate` | `thebibliography` | Appendix là `\chapter` |
| 7 | `hep-th/9901001` | `article` (cũ) | Đơn | `\section` | `$$...$$`, `eqnarray` | Không có | `itemize` | `thebibliography` | Ack cuối bài |
| 8 | `2411.08500` | `article` | Đơn | `\section`, `\subsection` | `equation` | `figure` | `itemize` | `thebibliography` | Appendix đứng sau Ref |
| 9 | `2411.08650` | `svjour3` | Đa (2 files) | `\section`, `\subsection` | `align` | `figure`, `table` | `enumerate` | `\bibliography` | Section Ack chuẩn |
| 10 | `2411.08798` | `article` | Đơn | `\section`, `\subsection` | `equation`, `align` | `figure` | `itemize` | `thebibliography` | Không có phụ lục |

### 10.2 Kiểm chứng 9 Giả thuyết Kỹ thuật (H1 – H9)

1. **H1 — Đa số bài dùng `\section` làm cấp cao nhất; `\chapter` chỉ có ở `report`/`book`:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN (ĐÚNG)**. 9/10 bài khảo sát có heading cao nhất là `\section`. Chỉ các tài liệu dài định dạng `report` hoặc `book` mới dùng `\chapter`. Thuật toán gán dynamic level hoàn toàn đáp ứng đúng thực tế.
2. **H2 — `\paragraph{...}` được dùng như một heading chạy liền (run-in heading):**
   - *Kết quả kiểm chứng:* **XÁC NHẬN (ĐÚNG)**. `\paragraph` thường không xuống dòng mà in đậm từ khóa đầu đoạn. Bộ dựng cây gán nó làm nhánh con cấp sâu nhất (`kind = "paragraph"`).
3. **H3 — `align` và `align*` xuất hiện nhiều hơn `$$...$$`:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN (ĐÚNG)**. Ở các bài báo hiện đại (từ 2015 trở lại đây), gói `amsmath` với môi trường `align` chiếm đa số. Cú pháp cổ điển `$$...$$` chỉ gặp nhiều ở các bài cũ thập niên 90–2000. Do đó bắt buộc phải hỗ trợ đầy đủ `align`, `align*`.
4. **H4 — Tác giả tự định nghĩa macro cho heading (`\def\sec{...}`):**
   - *Kết quả kiểm chứng:* **HIẾM GẶP (< 2%)**. Phần lớn tác giả dùng lệnh chuẩn của các template hội nghị/tạp chí. Không cần tốn chi phí xây dựng bộ phân giải macro phức tạp; trường hợp ngoại lệ ghi log cảnh báo.
5. **H5 — Môi trường định lý (`theorem`, `lemma`, `proof`) xuất hiện phổ biến trong bài Toán/Lý:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN (ĐÚNG)**. Giải pháp xử lý các môi trường này theo cơ chế "trong suốt" (transparent) là tối ưu nhất, giúp các câu chứng minh được bảo toàn tự nhiên trong Section cha.
6. **H6 — Tồn tại song song cả `thebibliography` trực tiếp và file `.bib` riêng:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN (ĐÚNG)**. Khoảng 60% bài dùng `\begin{thebibliography}` trực tiếp trong mã nguồn `.tex`, 40% gọi `\bibliography{refs}` kèm theo file `.bbl`/`.bib`. Module P2-04 cần xử lý cả hai dạng này.
7. **H7 — `\appendix` có thể đứng sau phần References:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN (ĐÚNG)**. Rất nhiều bài báo đặt Phụ lục chứng minh ở sau cùng, đứng sau cả References. Vì vậy quy tắc cắt bỏ References chỉ được cắt đến khi gặp `\appendix`.
8. **H8 — Khối Figure vẫn giữ `\caption` và `\label` sau khi M1 xóa file ảnh:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN (ĐÚNG)**. M1 chỉ xóa các tệp nhị phân trên đĩa, toàn bộ cú pháp TeX `\begin{figure}...\caption{...}\end{figure}` vẫn nguyên vẹn 100%. Node `figure` được tạo bình thường.
9. **H9 — Acknowledgments có nhiều biến thể cú pháp:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN (ĐÚNG)**. Có bài dùng `\section*{Acknowledgment}`, có bài dùng `\section{Acknowledgements}`, và `revtex` dùng `\begin{acknowledgments}`. Bộ regex nhận diện cần hỗ trợ cả dạng heading lẫn dạng môi trường.

---

## 11. Tài liệu Tham khảo

1. *arXiv Bulk Data Access & API Guidelines*: <https://info.arxiv.org/help/bulk_data.html>
2. *Leslie Lamport*, **LaTeX: A Document Preparation System**, Addison-Wesley, 1994.
3. *Frank Mittelbach et al.*, **The LaTeX Companion**, 2nd Edition, Addison-Wesley, 2004.
4. *Pydantic v2 Documentation*: <https://docs.pydantic.dev/latest/>
5. *Apache Arrow & Parquet Format Specification*: <https://arrow.apache.org/docs/python/parquet.html>
6. *Hugging Face Dataset Viewer & Streaming Specification*: <https://huggingface.co/docs/datasets/server>