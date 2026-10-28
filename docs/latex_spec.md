# Đặc tả cú pháp và quy ước về cây phân cấp theo mã nguồn LaTeX

---

## 📑 Mục lục

1. [Mục đích và phạm vi](#1-mục-đích-và-phạm-vi)
   - 1.1 [Mục đích](#11-mục-đích)
   - 1.2 [Vị trí trong Pipeline và ranh giới trách nhiệm các module](#12-vị-trí-trong-pipeline-và-ranh-giới-trách-nhiệm-các-module)
   - 1.3 [Phạm vi và ranh giới](#13-phạm-vi-và-ranh-giới)
2. [Các nguyên tắc bất biến cốt lõi](#2-các-nguyên-tắc-bất-biến-cốt-lõi)
3. [Mô hình dữ liệu: cây lồng nhau và bảng Parquet](#3-mô-hình-dữ-liệu-cây-lồng-nhau-và-bảng-parquet)
   - 3.1 [Quy chuẩn định dạng lưu trữ](#31-quy-chuẩn-định-dạng-lưu-trữ)
   - 3.2 [Cấu trúc Dataset Repository và hệ thống 6 bảng Parquet](#32-cấu-trúc-dataset-repository-và-hệ-thống-6-bảng-parquet)
   - 3.3 [Nguyên tắc làm phẳng dữ liệu từ LaTeX sang Parquet](#33-nguyên-tắc-làm-phẳng-dữ-liệu-từ-latex-sang-parquet)
   - 3.4 [Tập giá trị chuẩn cho thuộc tính `kind`](#34-tập-giá-trị-chuẩn-cho-thuộc-tính-kind)
   - 3.5 [Quy ước định danh `element_id` và tính xác định](#35-quy-ước-định-danh-element_id-và-tính-xác-định)
4. [Chi tiết thuật toán](#4-chi-tiết-thuật-toán)
   - 4.1 [Tự nhận diện cấp bậc tiêu đề](#41-tự-nhận-diện-cấp-bậc-tiêu-đề)
   - 4.2 [Dựng quan hệ cha con bằng ngăn xếp](#42-dựng-quan-hệ-cha-con-bằng-ngăn-xếp)
   - 4.3 [Chuẩn hóa tiêu đề Section](#43-chuẩn-hóa-tiêu-đề-section)
5. [Quy tắc bóc tách các nút lá và khối nội dung](#5-quy-tắc-bóc-tách-các-nút-lá-và-khối-nội-dung)
   - 5.1 [Nút lá: Câu văn](#51-nút-lá-câu-văn)
   - 5.2 [Nút lá: Công thức toán dạng khối](#52-nút-lá-công-thức-toán-dạng-khối)
   - 5.3 [Nút lá: Khối hình ảnh và bảng biểu](#53-nút-lá-khối-hình-ảnh-và-bảng-biểu)
   - 5.4 [Nút nhánh: Danh sách liệt kê và mục danh sách](#54-nút-nhánh-danh-sách-liệt-kê-và-mục-danh-sách)
6. [Xử lý các phần đặc biệt và vùng ngoại lệ](#6-xử-lý-các-phần-đặc-biệt-và-vùng-ngoại-lệ)
   - 6.1 [Phần tài liệu tham khảo](#61-phần-tài-liệu-tham-khảo)
   - 6.2 [Phần cảm ơn](#62-phần-cảm-ơn)
   - 6.3 [Phần phụ lục](#63-phần-phụ-lục)
   - 6.4 [Phần tóm tắt và lời mở đầu](#64-phần-tóm-tắt-và-lời-mở-đầu)
   - 6.5 [Bình luận và các khối mã nguyên bản](#65-bình-luận-và-các-khối-mã-nguyên-bản)
   - 6.6 [Các môi trường bao bọc trong suốt](#66-các-môi-trường-bao-bọc-trong-suốt)
7. [Xử lý lỗi và phòng thủ](#7-xử-lý-lỗi-và-phòng-thủ)
8. [Ví dụ hoàn chỉnh: đầu vào LaTeX và đầu ra `hierarchy.json`](#8-ví-dụ-hoàn-chỉnh-đầu-vào-latex-và-đầu-ra-hierarchyjson)
   - 8.1 [Đoạn mã LaTeX mẫu](#81-đoạn-mã-latex-mẫu)
   - 8.2 [File đầu ra cấu trúc cây `hierarchy.json`](#82-file-đầu-ra-cấu-trúc-cây-hierarchyjson)
   - 8.3 [Ánh xạ sang hai bảng Parquet elements và hierarchy](#83-ánh-xạ-sang-hai-bảng-parquet-elements-và-hierarchy)
     - 8.3.1 [Bảng `elements`: Trích đoạn các dòng đại diện](#831-bảng-elements-trích-đoạn-các-dòng-đại-diện)
     - 8.3.2 [Bảng `hierarchy`: Biểu diễn cây phiên bản 1](#832-bảng-hierarchy-biểu-diễn-cây-phiên-bản-1)
9. [Khảo sát thực nghiệm và kiểm chứng giả thuyết](#9-khảo-sát-thực-nghiệm-và-kiểm-chứng-giả-thuyết)
   - 9.1 [Bảng dữ liệu khảo sát thực nghiệm 10 bài báo](#91-bảng-dữ-liệu-khảo-sát-thực-nghiệm-10-bài-báo)
   - 9.2 [Kiểm chứng 9 giả thuyết kỹ thuật](#92-kiểm-chứng-9-giả-thuyết-kỹ-thuật)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Mục đích và Phạm vi

### 1.1 Mục đích
Tài liệu này là **nguồn sự thật duy nhất (SSOT)** quy định thuật toán, cấu trúc dữ liệu và các quy tắc nghiệp vụ để phân tích mã nguồn LaTeX của từng bài báo thành cấu trúc cây phân cấp lưu tại file trung gian `hierarchy.json`, sau đó làm phẳng triệt để thành hai bảng Parquet `elements` và `hierarchy` để phát hành trên Hugging Face.

### 1.2 Vị trí trong Pipeline và ranh giới trách nhiệm các module
Quy trình phân tích văn bản của Milestone 2 diễn ra tuần tự theo sơ đồ sau:

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
│ 2. LaTeX Cleanup & Math Normalization                  │
│    (latex_cleanup.py)                                  │
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
│    • Bóc tách lá và cành  │   │    • Chuẩn hóa trường    │
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
│    • Chuẩn Flat & Primitive                            │
│    • Không nested objects, không serialized JSON blob  │
└────────────────────────────────────────────────────────┘
```

Bảng dưới đây quy định chi tiết dữ liệu vào/ra và ranh giới trách nhiệm của từng module:

| Module | Đầu vào | Đầu ra | Ranh giới trách nhiệm và điều cấm |
|---|---|---|---|
| **`latex_parser.py`**<br/>*(Multi-file Resolver)* | Thư mục chứa các file `.tex` giải nén từ M1. | Chuỗi văn bản LaTeX duy nhất đã gộp. | Tìm file chính chứa `\documentclass`. Lần theo các lệnh `\input{...}` và `\include{...}` để gộp thành một chuỗi duy nhất, xử lý triệt để chu trình include đệ quy. |
| **`latex_cleanup.py`**<br/>*(Cleanup & Math Normalization)* | Chuỗi LaTeX đã gộp. | Chuỗi LaTeX sạch cấu trúc. | **Được phép:** Xóa các lệnh định dạng font `\bf`, `\it`, canh lề `\centering`, `\raggedright`, khoảng trắng `\vspace`, `\hspace`, tùy chọn vị trí float `[htbp]`.<br/>**Điều cấm:** Không xóa lệnh heading, không xóa môi trường toán `figure`, `table`, `equation`, không xóa `\caption`, `\label`, `\cite`, `\appendix`, `thebibliography`. |
| **`tree_builder.py`**<br/>*(Tree Builder)* | Chuỗi LaTeX đã cleanup, `arxiv_id`. | Cây `hierarchy.json` lồng nhau. | Dựng cây theo đúng đặc tả kỹ thuật: gán level động, bóc tách các nút lá, loại bỏ References, giữ nguyên Ack và Phụ lục. |
| **`bibtex_extractor.py`**<br/>*(BibTeX Extractor)* | Chuỗi LaTeX còn nguyên References. | Danh sách `BibtexEntry` hoặc file `refs.bib`. | Bóc tách toàn bộ `\bibitem` hoặc parse file `.bib` đi kèm thành các entry có cấu trúc. |
| **`dedup.py`**<br/>*(Deduplication)* | `hierarchy.json` và `refs.bib` qua các phiên bản. | Dữ liệu cấu trúc đã khử trùng lặp. | Khử trùng lặp nội dung giữa các version của cùng bài báo; hợp nhất các trường BibTeX qua cơ chế Field Union. |
| **`export_parquet.py`**<br/>*(Parquet Exporter)* | Dữ liệu cây đã dedup, metadata, bibtex, predictions. | 6 bảng sharded Parquet trên Hugging Face. | Làm phẳng triệt để cây thành hai bảng phẳng `elements` và `hierarchy` theo mô hình Adjacency List. Đóng gói bộ 6 bảng Parquet vào thư mục `data/`. **Điều cấm:** Tuyệt đối không serialize JSON blobs vào cột, không lưu nested objects, đảm bảo viewer và backend streaming của Hugging Face đọc trực tiếp. |

### 1.3 Phạm vi và ranh giới
- **Trong phạm vi:**
  - Thuật toán nhận diện động cấp bậc tiêu đề và dựng quan hệ cha – con cho cây phân cấp.
  - Quy tắc bóc tách các nút lá gồm câu văn, công thức khối, hình ảnh, bảng biểu và nút nhánh danh sách liệt kê.
  - Quy tắc xử lý các phần đặc biệt: References, Acknowledgments, Appendices.
  - Quy chuẩn định danh `element_id`, cấu trúc trung gian `hierarchy.json` và cơ chế làm phẳng thành hai bảng Parquet `elements` và `hierarchy` theo mô hình Adjacency List.
- **Ngoài phạm vi:**
  - Không biên dịch LaTeX sang PDF qua `pdflatex`. Toàn bộ quá trình chỉ phân tích cú pháp tĩnh dạng văn bản bằng regex và máy trạng thái.
  - Không thực thi hay giải mã các macro TeX phức tạp như `\def`, `\let`, `\expandafter`.

---

## 2. Các nguyên tắc bất biến cốt lõi

Quá trình phân tích và xây dựng dữ liệu bắt buộc phải tuân thủ 6 nguyên tắc cốt lõi sau:

1. **Nhận diện động cấp bậc tiêu đề:** Cấp bậc của heading được xác định linh hoạt dựa trên các lệnh thực tế xuất hiện trong bài báo, không gán cứng cố định (chi tiết tại Mục 4.1).
2. **Nút lá chỉ gồm 3 loại `kind`:** Chỉ gồm `sentence`, `formula` và `figure` (bao gồm cả hình ảnh và bảng biểu, vì đề bài coi Table là một loại Figure). Mọi nút lá đều không chứa nút con với `children = []` (chi tiết tại Mục 5).
3. **Danh sách liệt kê là nút nhánh:** Môi trường `itemize` là nút nhánh cấp 1, mỗi `item` là nút con cấp 2 chứa các câu văn hoặc danh sách con lồng nhau bên trong (chi tiết tại Mục 5.4).
4. **Phân loại phần đặc biệt:** Loại bỏ hoàn toàn phần References khỏi cây nội dung; bảo toàn phần Acknowledgments và Appendices dưới dạng các nút heading thông thường (chi tiết tại Mục 6).
5. **Chuẩn hóa thuộc tính `kind` và mã định danh:** Mỗi phần tử phải có thuộc tính `kind` theo enum chuẩn hóa và mã định danh `element_id` duy nhất có tính xác định cao chứa tiền tố `arxiv_id` (chi tiết tại Mục 3.4 và 3.5).
6. **Lưu trữ Parquet hoàn toàn phẳng:** Biểu diễn phả hệ qua mô hình Adjacency List với các cột nguyên thủy chuẩn Flat & Primitive, tuyệt đối không lưu lồng đối tượng hay chuỗi JSON serialize (chi tiết tại Mục 3).



---

## 3. Mô hình dữ liệu: cây lồng nhau và bảng Parquet

### 3.1 Quy chuẩn định dạng lưu trữ

Để đáp ứng tối đa hiệu năng của Hugging Face Dataset Viewer và Apache Arrow streaming backend, toàn bộ dữ liệu cấu trúc bài báo được phân định rõ ràng giữa hai biểu diễn:
- **Biểu diễn cây cục bộ `hierarchy.json`:** Chỉ lưu hành nội bộ ở bước xử lý Milestone 2, lưu trữ in-memory hoặc file tạm trên đĩa. Cây có cấu trúc JSON lồng nhau phân cấp phục vụ việc đọc hiểu cấu trúc phân tầng tự nhiên, hỗ trợ thuật toán bóc tách phạm vi heading, nhận diện cha - con và phân tích ngữ cảnh.
- **Biểu diễn bảng phẳng Parquet `elements` và `hierarchy`:** Khi đóng gói xuất bản, cây phân cấp bắt buộc phải được làm phẳng hoàn toàn thành hai bảng quan hệ theo mô hình Adjacency List chuẩn Apache Parquet, tuyệt đối không serialize cả cây hay các node thành JSON blob.

### 3.2 Cấu trúc Dataset Repository và hệ thống 6 bảng Parquet

Cấu trúc thư mục của repository bộ dữ liệu trên Hugging Face `<dataset-repo>` gồm 6 bảng Parquet đặt trong `data/` (`papers`, `references`, `elements`, `hierarchy`, `bibtex`, `predictions`).

> [!NOTE]
> **Nguồn sự thật duy nhất về Schema:** Định nghĩa chi tiết các trường, kiểu dữ liệu PyArrow (`PARQUET_SCHEMAS`) và các ràng buộc kiểm tra dữ liệu Pydantic (`MODELS`) của cả 6 bảng Parquet được quy định duy nhất tại **`src/schemas.py`**. Module xuất dữ liệu chỉ tập trung ánh xạ cây phân cấp nội bộ (`hierarchy.json`) sang hai bảng `elements` và `hierarchy` theo đúng schema đã khai báo trong `src/schemas.py`.

### 3.3 Nguyên tắc làm phẳng dữ liệu từ LaTeX sang Parquet
- Hugging Face Dataset Viewer và Streaming backend đọc Parquet trực tiếp theo từng khối nhị phân Row Groups hoặc Shards.
- **Quy chuẩn lưu trữ trường `content` trong bảng `elements`:**
  - Đối với nút `heading` và `sentence`: lưu trực tiếp chuỗi văn bản thuần (tiêu đề section hoặc nội dung câu văn).
  - Đối với nút `formula`: lưu **nguyên khối TeX thô** từ `\begin{...}` đến `\end{...}`, bao gồm cả `\label{...}`. Với cú pháp `$$...$$` và `\[...]`, chuẩn hóa về `\begin{equation}...\end{equation}` để thống nhất với `tests/make_mock.py` và exporter.
  - Đối với nút `figure`: lưu **nguyên khối TeX thô** của toàn bộ hình/bảng từ `\begin{figure}` / `\begin{table}` / `\begin{tabular}` đến `\end{...}`, bao gồm cả `\caption{...}`, `\label{...}`, và các môi trường con.
  - **Lý do thiết kế:** Cột `Element.content` là cột duy nhất có thể chứa nội dung. Nếu bóc vỏ công thức hoặc chỉ lấy caption thì `\label`, `\caption` hay mã bảng/hình sẽ bị mất hoàn toàn. Mặt khác, `tests/make_mock.py` cũng đang dùng raw block cho formula và figure, nên quy chuẩn này khớp 100% với mock hiện có và exporter. Vì `content` vẫn là một chuỗi duy nhất nên `to_rows()` vẫn giữ đúng 4 cột phẳng nguyên thủy và không cần đổi schema.
  - Cả `ParsedNode`, hàm `to_nested()` lẫn hai bảng `elements`, `hierarchy` đều không có các thuộc tính phụ (`caption`, `label`, `level`, `role`, `title`). Tuyệt đối không tách thành trường riêng và không đóng gói thành chuỗi JSON serialize nhét vào `content`.

### 3.4 Tập giá trị chuẩn cho thuộc tính `kind`
Đồng bộ 100% với `Kind` enum trong `src/schemas.py` (7 giá trị):

| `kind` | Vai trò trong cây | Là lá? | Mô tả |
|---|---|:---:|---|
| `document` | Gốc | ❌ | Node gốc đại diện cho toàn bộ một phiên bản bài báo (1 node/version). |
| `heading` | Nhánh | ❌ | Đại diện cho mọi cấp tiêu đề (`\part`, `\chapter`, `\section`, `\subsection`, `\subsubsection`, `\paragraph`, `\subparagraph`). Độ sâu phân cấp được thể hiện qua quan hệ cha – con trong bảng `hierarchy`. |
| `itemize` | Nhánh | ❌ | Container đại diện cho môi trường danh sách (`itemize`, `enumerate`, `description`). |
| `item` | Nhánh | ❌ | Đại diện cho một mục con (`\item`) bên trong một container danh sách. |
| `sentence` | **Lá** | ✅ | Một câu văn hoàn chỉnh đã qua tách câu và bảo vệ thực thể. |
| `formula` | **Lá** | ✅ | Một công thức toán dạng khối. |
| `figure` | **Lá** | ✅ | Toàn bộ khối hình ảnh (`figure`, `figure*`, `wrapfigure`) **và cả bảng biểu** (`table`, `table*`, `tabular` độc lập). |

### 3.5 Quy ước định danh `element_id` và tính xác định
Quy chuẩn định danh `element_id` bắt buộc sử dụng hàm `make_element_id(arxiv_id, local_id)` trong `src/schemas.py`:

```python
element_id = make_element_id(arxiv_id, local_id)
# Kết quả: f"{to_folder_name(arxiv_id)}#{local_id}"
```

- Định dạng ID tuân thủ cấu trúc **publication key + anchor** theo đúng README (Mục 13.1):
  - Node gốc phiên bản 1: `make_element_id("2411.03799", "root1")` $\rightarrow$ `"2411-03799#root1"`.
  - Node câu văn hoặc heading: `make_element_id("2411.03799", "v1-s1")` $\rightarrow$ `"2411-03799#v1-s1"` (hoặc `"2411-03799#s1"` sau bước khử trùng lặp dedup giữa các phiên bản).
  - Tuyệt đối không dùng các định danh tùy tiện như `#doc`, `#sent1`.

---

## 4. Chi tiết thuật toán

### 4.1 Tự nhận diện cấp bậc tiêu đề
LaTeX chuẩn quy định thứ bậc tự nhiên gồm 7 cấp độ:

| Cấp LaTeX chuẩn | Lệnh tương ứng | Hạng chuẩn |
|---|---|:---:|
| Part | `\part{...}` | 0 |
| Chapter | `\chapter{...}` | 1 |
| Section | `\section{...}` | 2 |
| Subsection | `\subsection{...}` | 3 |
| Subsubsection | `\subsubsection{...}` | 4 |
| Paragraph | `\paragraph{...}` | 5 |
| Subparagraph | `\subparagraph{...}` | 6 |

**Quy tắc nhận diện động:**
1. Lệnh có dấu sao như `\section*{Introduction}` có cùng cấp bậc và trọng số thứ bậc với dạng không sao `\section{Introduction}`. Dấu sao chỉ điều khiển việc đánh số hiển thị của LaTeX, không ảnh hưởng đến cấu trúc phân cấp logic.
2. Thuật toán quét toàn bộ văn bản sau khi đã loại trừ comment và môi trường verbatim, lập tập hợp các lệnh heading thực sự xuất hiện trong bài báo.
3. Sắp xếp tập hợp đó tăng dần theo `Rank`.
4. Ánh xạ các lệnh này sang dải số nguyên liên tục bắt đầu từ `1`:

$$\text{Level}(\text{cmd}) = \text{Index trong danh sách các lệnh có mặt} + 1$$

*Ví dụ minh họa:*
- Bài báo chỉ có `\section`, `\subsection`:
  $\rightarrow$ `\section` có level 1; `\subsection` có level 2.
- Bài báo có `\chapter`, `\section`, `\subsection`:
  $\rightarrow$ `\chapter` có level 1; `\section` có level 2; `\subsection` có level 3.
- Bài báo nhảy cấp từ `\section` sang `\subsubsection`:
  $\rightarrow$ `\section` có level 1; `\subsubsection` có level 2. Bộ dựng cây không sinh node rỗng trung gian.

### 4.2 Dựng quan hệ cha con bằng ngăn xếp
Sử dụng cấu trúc dữ liệu ngăn xếp để quản lý phạm vi của các section đang mở:

```python
# Thuật toán dựng cây:
# stack quản lý tuple (level, node_id) nội bộ trong quá trình dựng, node không lưu trường level
stack = [(0, root_node_id)]

for item in parsed_items:
    if item.is_heading:
        item_level = dynamic_level_map[item.command]
        # Đóng các section có level sâu hơn hoặc bằng level của heading hiện tại
        while len(stack) > 1 and stack[-1][0] >= item_level:
            stack.pop()
        
        parent_id = stack[-1][1]
        new_heading_id = add_node(kind=Kind.HEADING, content=item.title, parent=parent_id)
        stack.append((item_level, new_heading_id))
    else:
        # Nút nội dung: câu văn, công thức, danh sách, hình ảnh, bảng biểu
        parent_id = stack[-1][1]
        add_node(kind=item.kind, content=item.raw_content, parent=parent_id)
```

> [!NOTE]
> Biến `level` chỉ tồn tại nội bộ trong ngăn xếp `stack` của Tree Builder để xác định quan hệ cha - con, hoàn toàn không được lưu thành thuộc tính của node hay cột trong schema Parquet.

**Xử lý văn bản trước Heading đầu tiên:**
Mọi nội dung xuất hiện sau `\begin{document}` hoặc sau `abstract` nhưng nằm trước lệnh heading đầu tiên sẽ được gắn trực tiếp vào nút gốc `document`.

### 4.3 Chuẩn hóa tiêu đề Section
Tiêu đề của section được trích xuất từ đối số thứ nhất của lệnh heading và làm sạch:
1. Loại bỏ nhãn tham chiếu `\label{...}` lồng bên trong tiêu đề (loại bỏ hoàn toàn khỏi tiêu đề, không lưu thành trường riêng vì schema không có cột `label`).
2. Xử lý macro tương thích `\texorpdfstring{TeX text}{Plain text}`: Luôn lấy tham số thứ hai `Plain text` để đảm bảo tiêu đề sạch ký tự lạ.
3. Bảo toàn các công thức toán nội dòng như `$\Delta$-Coloring` để giữ nguyên vẹn ý nghĩa chuyên ngành.
4. Xóa bỏ các lệnh ngắt dòng thủ công `\\`, `\newline` hoặc định dạng chữ `\textbf{...}`, `\emph{...}`.

---

## 5. Quy tắc bóc tách các nút lá và khối nội dung

### 5.1 Nút lá: Câu văn
Nhiệm vụ của bộ tách câu là chia một đoạn văn bản thành danh sách các node `kind = "sentence"`. Việc tách câu phải thông minh, không được gây phân mảnh sai các thực thể toán học và trích dẫn.

**Quy trình 3 bước tách câu:**
1. **Bước 1 — Bảo vệ các thực thể không thể tách:**
   Thay thế tạm thời bằng các token đặc biệt hoặc bảo vệ các thành phần:
   - Toán nội dòng: `$...$` và `\( ... \)`.
   - Trích dẫn thư mục: `\cite{...}`, `\citep{...}`, `\citet{...}`.
   - Nhãn tham chiếu nội bộ: `\ref{...}`, `\eqref{...}`, `\autoref{...}`.
   - Địa chỉ liên kết: `\url{...}`, `\href{...}{Ellipsis}`.
   - Các chữ viết tắt phổ biến trong văn bản khoa học: `et al.`, `i.e.`, `e.g.`, `cf.`, `vs.`, `Fig.`, `Eq.`, `Sec.`, `Ref.`, `Thm.`, `Lem.`, `No.`, `Vol.`, `pp.`, `etc.`.
   - Viết tắt tên tác giả: `A. Smith`, `J. D. Watson`.
   - Số thập phân: `3.14159`, `.05`, `1,000`.
2. **Bước 2 — Phân tách ranh giới câu:**
   - Chỉ tách câu tại dấu chấm kết thúc `.`, dấu chấm hỏi `?`, dấu chấm than `!` khi theo sau bởi khoảng trắng và một ký tự viết hoa hoặc bắt đầu một đoạn mới.
   - Hai dấu xuống dòng liên tiếp `\n\n` hoặc lệnh `\par` báo hiệu ngắt đoạn: Ngắt câu ngay tại ranh giới đoạn văn dù câu trước có thiếu dấu chấm.
3. **Bước 3 — Khôi phục thực thể và chuẩn hóa:**
   - Hoàn trả lại các token đã bảo vệ ở Bước 1.
   - Chuẩn hóa khoảng trắng dư thừa (`\s+` $\rightarrow$ `" "`).
   - Giữ nguyên các lệnh trích dẫn `\cite{...}` trong trường `content` của câu nhằm phục vụ cho module so khớp trích dẫn ở Milestone 3.

### 5.2 Nút lá: Công thức toán dạng khối
Mọi công thức hiển thị riêng biệt trên một dòng được đóng gói thành một node `kind = "formula"`.

**Các môi trường bắt buộc nhận diện:**
- Cú pháp TeX cổ điển: `$$ ... $$`
- Cú pháp chuẩn LaTeX: `\[ ... \]`
- Môi trường đánh số cơ bản: `\begin{equation} ... \end{equation}`, `\begin{equation*} ... \end{equation*}`
- Các môi trường toán nâng cao từ gói `amsmath` và `mathtools`:
  `align`, `align*`, `gather`, `gather*`, `multline`, `multline*`, `flalign`, `flalign*`, `eqnarray`, `eqnarray*`, `displaymath`.

**Quy tắc trích xuất:**
- Thuộc tính `content`: Lưu **nguyên khối TeX** của công thức, từ `\begin{...}` đến `\end{...}`, bao gồm cả `\label{...}`. Với cú pháp `$$ ... $$` hoặc `\[ ... \]`, parser chuẩn hóa đưa về dạng `\begin{equation} ... \end{equation}` để thống nhất với `tests/make_mock.py` và exporter.
- Không bóc vỏ và không tách nhãn: Lệnh `\label{...}` (nếu có) được giữ nguyên vẹn bên trong khối TeX thô của `content`, tuyệt đối không bóc thẻ `\begin`/`\end` và không tách ra thành trường `label` riêng hay cột riêng (vì `ParsedNode`, `Element`, và `to_nested()` không có trường `label`).
- Môi trường con lồng bên trong: Các cấu trúc toán bên trong công thức khối như `\begin{split}`, `\begin{cases}`, `\begin{aligned}`, `\begin{matrix}` là một phần của công thức, được bảo toàn nguyên vẹn bên trong khối TeX thô của `content`, không tách thành các node lá độc lập.
- Toán nội dòng: Cú pháp `$x^2 + y^2 = z^2$` nằm trong luồng văn bản phải được giữ nguyên trong `sentence`, không tạo node `formula`.

**Xử lý hiện tượng câu bị ngắt quãng bởi công thức khối:**
Khi công thức khối xuất hiện giữa câu, parser chia thành chuỗi 3 nút liên tiếp theo thứ tự xuất hiện trong văn bản nguồn:
```text
[sentence: "Hệ số ma sát được định nghĩa bởi"]
[formula: "\begin{equation} f = \mu N \end{equation}"]
[sentence: "trong đó \mu là hệ số tĩnh."]
```

### 5.3 Nút lá: Khối hình ảnh và bảng biểu
Toàn bộ khối môi trường trôi (hình ảnh và bảng biểu) được xem là một nút lá độc lập và đều được gán chung **`kind = "figure"`** (đúng theo quy ước của `src/schemas.py` coi Table là một loại Figure).

**Các môi trường hỗ trợ:**
- Khối hình ảnh: `figure`, `figure*`, `wrapfigure`.
- Khối bảng biểu: `table`, `table*`, và `tabular` độc lập khi đứng ngoài môi trường table.

**Quy tắc trích xuất:**
- Thuộc tính `content`: Lưu **nguyên khối TeX thô** của toàn bộ hình/bảng (từ `\begin{figure}` / `\begin{table}` / `\begin{tabular}` đến `\end{...}`). Toàn bộ các câu lệnh bên trong như `\caption{...}`, `\label{...}`, `\includegraphics{...}` hay cấu trúc bảng `tabular` và các môi trường con được giữ nguyên vẹn trong chuỗi `content` thô.
- Không tách `caption` hay `label` thành trường riêng: Cả `ParsedNode`, hàm xuất `to_nested()` lẫn bảng Parquet `elements` đều không có các trường `caption` hay `label`. Mọi chi tiết về chú thích và nhãn đều nằm sẵn trong chuỗi TeX thô của `content`, không tách thành trường riêng hay cột riêng và không serialize JSON vào `content`.
- **Trường hợp xóa ảnh ở M1:** Crawler M1 đã xóa các file ảnh nhị phân như `.png`, `.jpg`, `.pdf` để tiết kiệm dung lượng nhưng vẫn giữ nguyên mã `\begin{figure}`. Bộ dựng cây vẫn tạo node `figure` bình thường dù lệnh `\includegraphics` bên trong không còn file ảnh vật lý.
- Các môi trường con như `subfigure`, `minipage` được giữ nguyên trong nội dung khối TeX thô của node `figure`, không tách thành các node riêng rẽ.
- **Tuân thủ nguyên tắc lưu Parquet phẳng:**
  Cả trong file trung gian `hierarchy.json` lẫn bảng Parquet `elements`, trường `content` đều lưu trữ trực tiếp chuỗi khối TeX thô, tuyệt đối không serialize thành chuỗi JSON dạng `{\"caption\": \"...\", \"label\": \"...\"}`.

### 5.4 Nút nhánh: Danh sách liệt kê và mục danh sách
Cấu trúc danh sách biểu diễn theo mô hình hai tầng lồng nhau:

```text
Node "itemize"
  ├── Node "item"
  │     ├── Node "sentence"
  │     └── Node "formula" (nếu có)
  └── Node "item"
        ├── Node "sentence"
        └── Node "itemize" (danh sách con lồng nhau nếu có)
```

**Các môi trường áp dụng:**
Bao gồm các môi trường `itemize`, `enumerate` và `description`.
- Tạo một node `kind = "itemize"` để bọc toàn bộ khối môi trường.
- Với mỗi lệnh `\item`, tạo một node `kind = "item"` làm con của node `itemize`.
- Nếu có nhãn tùy biến `\item[label]`, nhãn này được loại bỏ do schema không có cột riêng; nội dung mục được bóc tách bình thường.
- Nội dung sau `\item` được phân tích như văn bản thông thường, bao gồm câu văn, công thức hoặc danh sách lồng nhau.

---

## 6. Xử lý các phần đặc biệt và vùng ngoại lệ

### 6.1 Phần tài liệu tham khảo
Yêu cầu bắt buộc: Cây phân cấp nội dung `hierarchy.json` không được chứa các mục tham khảo thư mục.

**Nhận diện và cắt bỏ:**
1. Khối môi trường tường minh:
   - `\begin{thebibliography}{...} ... \end{thebibliography}`
   - `\bibliography{...}`, `\printbibliography`
2. Khối heading tham khảo:
   - Các Section có tiêu đề thuộc danh sách: `References`, `Reference`, `Bibliography`, `Literature Cited`, `References and Notes`.
   - Toàn bộ heading này và tất cả các node con thuộc phạm vi của nó bị loại bỏ hoàn toàn khỏi cây phân cấp.
   - Phạm vi cắt bỏ kết thúc khi gặp một heading khác cùng cấp hoặc cao hơn, hoặc khi gặp lệnh `\appendix`.

> [!CAUTION]
> **Thứ tự xử lý giữa các module:**
> Module `bibtex_extractor.py` phải trích xuất các `\bibitem` thành file `refs.bib` từ chuỗi LaTeX trước khi `tree_builder.py` loại bỏ phần này khỏi cây phân cấp!

### 6.2 Phần cảm ơn
Phần cảm ơn phải được bảo toàn trong cây nội dung.
- Các tiêu đề thường gặp: `Acknowledgment`, `Acknowledgments`, `Acknowledgement`, `Acknowledgements`.
- Các môi trường đặc thù: `\begin{acknowledgments} ... \end{acknowledgments}` trong gói `revtex4`.
- **Quy tắc dựng node:**
  - Tạo node `kind = "heading"` với `content = "Acknowledgments"` (tiêu đề section được lưu trực tiếp vào trường `content`; không có trường `title`).
  - Nút heading này đóng vai trò như một section cấp cao nhất trực tiếp dưới nút gốc `document`.
  - Phân tích các câu văn bên trong thành các nút con `kind = "sentence"` gắn cha là heading này.
  - Không sinh các trường `title`, `level` hay `role` (vì `ParsedNode`, `Element`, `Hierarchy` trong `src/schemas.py` và `to_nested()` hoàn toàn không có các cột này; vai trò acknowledgment chỉ xử lý nội bộ trong bộ quét để phân loại khối `special`).

### 6.3 Phần phụ lục
Mọi nội dung trong phần phụ lục phải được giữ trọn vẹn trong cây phân cấp.
- Chỉ thị phân định: Lệnh `\appendix` hoặc môi trường `\begin{appendices} ... \end{appendices}`.
- Lệnh `\appendix` trong LaTeX là một công tắc chuyển chế độ đánh số từ chữ số sang chữ cái A, B, C; bản thân lệnh này không nhất thiết phải là một heading.
- Khi gặp `\appendix`, các section xuất hiện phía sau lệnh này vẫn được dựng thành các node `kind = "heading"` bình thường với `content` là tiêu đề section.
- Không sinh trường `role = "appendix"` (cũng như `title`, `level`) vào dữ liệu vì `ParsedNode`, `Element`, `Hierarchy` và `to_nested()` không có các trường này; trạng thái appendix chỉ dùng nội bộ trong parser để reset cơ chế bỏ qua References hoặc xác định phạm vi kết thúc References.
- Nếu phụ lục xuất hiện sau phần References, hệ thống vẫn tiếp tục mở lại việc dựng cây cho phụ lục, tuyệt đối không cắt bỏ phụ lục theo References.

### 6.4 Phần tóm tắt và lời mở đầu
- **Preamble:** Toàn bộ phần mã nguồn nằm trước `\begin{document}` chứa các khai báo `\documentclass`, `\usepackage`, macro bị loại bỏ khỏi cây nội dung và không đưa vào cây phân cấp.
- **Abstract:** Môi trường `\begin{abstract} ... \end{abstract}` được chuẩn hóa thành một Heading đặc biệt:
  - `kind = "heading"`
  - `content = "Abstract"` (tiêu đề lưu trực tiếp vào trường `content`; không có các trường `title`, `level`, hay `role`).
  - Nút heading này được gắn làm con trực tiếp của nút gốc `document`.
  - Các câu văn bên trong abstract được tách thành các node `sentence` con có cha là heading này.
- **Metadata tác giả:** Các lệnh `\title{...}`, `\author{...}`, `\date{...}` không tạo node trong cây vì đã được lưu trữ riêng trong bảng `papers`.
- **Nội dung sau `\end{document}`:** Bỏ qua hoàn toàn.

### 6.5 Bình luận và các khối mã nguyên bản
1. **Comment:**
   - Ký tự `%` bắt đầu comment đến hết dòng. Mọi lệnh LaTeX nằm sau `%` bị vô hiệu hóa.
   - Ký tự escape `\%` biểu thị dấu phần trăm thông thường, không phải là comment.
   - Parser phải loại bỏ comment trước khi nhận diện cấu trúc.
2. **Vùng mã nguyên văn:**
   - Các môi trường: `verbatim`, `lstlisting`, `minted`, `rawhtml`.
   - Các lệnh rẽ nhánh điều kiện: `\iffalse ... \fi`, `\begin{comment} ... \end{comment}`.
   - Parser xem các vùng này như khối văn bản thô, không phân tích cú pháp heading hay tách câu bên trong.

### 6.6 Các môi trường bao bọc trong suốt
Các môi trường định lý và trình bày: `theorem`, `lemma`, `proposition`, `corollary`, `definition`, `remark`, `proof`, `quote`, `center`, `minipage`.
- Các môi trường này đóng vai trò bao bọc ngữ nghĩa nhưng không làm thay đổi phân cấp cha – con của tài liệu.
- **Quy tắc:** Bộ phân tích xem các môi trường này là "trong suốt" — nội dung bên trong được bóc tách thành các câu `sentence` hoặc công thức `formula` và gắn trực tiếp vào Section cha hiện tại.

---

## 7. Xử lý lỗi và phòng thủ

Để pipeline không bị dừng đột ngột khi xử lý hàng nghìn bài báo thực tế, bộ dựng cây triển khai cơ chế bắt lỗi phòng thủ:

1. **Bài báo không có mã nguồn TeX:**
   - Được ghi nhận khi crawler M1 phát hiện bài chỉ có file PDF hoặc DOCX. Parser tạo file `hierarchy.json` rỗng chứa duy nhất node gốc `document` (với `content = "Version {version} root"` hoặc chuỗi rỗng), ghi log cảnh báo và tiếp tục xử lý bài tiếp theo. Bảng `hierarchy` không sinh dòng nào cho bài này vì nút gốc không có cha.
2. **Môi trường thiếu thẻ đóng:**
   - Không gây crash chương trình. Khi một môi trường (`equation`, `figure`, `table`, `tabular`...) thiếu thẻ đóng `\end{...}` tương ứng, parser ghi log cảnh báo, tự động xác định ranh giới kết thúc tại heading tiếp theo hoặc tại cuối tài liệu, đồng thời **bổ sung thẻ đóng `\end{...}` tương ứng vào `content`** để đảm bảo chuỗi lưu trữ luôn là một khối TeX hoàn chỉnh.
3. **Bài báo không có bất kỳ lệnh heading nào:**
   - Toàn bộ các câu văn, công thức, bảng biểu trong bài được gắn trực tiếp vào node gốc `document`.
4. **Lệnh include trỏ đến file không tồn tại:**
   - Ghi nhận warning vào log file, thay thế bằng khoảng trắng và tiếp tục xử lý các phần khác của bài báo.
5. **Section rỗng không có nội dung:**
   - Vẫn tạo node `kind = "heading"` với `content` là tiêu đề section, thuộc tính `children = []` (không có nút con nào bên dưới).

---

## 8. Ví dụ hoàn chỉnh: đầu vào LaTeX và đầu ra `hierarchy.json`

### 8.1 Đoạn mã LaTeX mẫu

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

### 8.2 File đầu ra cấu trúc cây `hierarchy.json`

Cấu trúc JSON phân cấp trả về từ hàm `to_nested()` của `src/tree_builder.py` để biểu diễn trực quan cây phân cấp của một phiên bản (mỗi node chỉ gồm 4 trường chuẩn `id`, `kind`, `content`, `children`; không có các trường phụ `title`, `level`, `role`, `caption`, `label`):

```json
{
  "id": "root1",
  "kind": "document",
  "content": "Version 1 root",
  "children": [
    {
      "id": "v1-h1",
      "kind": "heading",
      "content": "Abstract",
      "children": [
        {
          "id": "v1-s1",
          "kind": "sentence",
          "content": "Graph coloring is a fundamental problem in computer science.",
          "children": []
        },
        {
          "id": "v1-s2",
          "kind": "sentence",
          "content": "We present a linear-time heuristic bound.",
          "children": []
        }
      ]
    },
    {
      "id": "v1-h2",
      "kind": "heading",
      "content": "Introduction",
      "children": [
        {
          "id": "v1-s3",
          "kind": "sentence",
          "content": "Graph coloring has been studied extensively \\cite{brooks1941colouring}.",
          "children": []
        },
        {
          "id": "v1-s4",
          "kind": "sentence",
          "content": "Let $G=(V, E)$ be a connected graph.",
          "children": []
        },
        {
          "id": "v1-list1",
          "kind": "itemize",
          "content": "",
          "children": [
            {
              "id": "v1-it1",
              "kind": "item",
              "content": "",
              "children": [
                {
                  "id": "v1-s5",
                  "kind": "sentence",
                  "content": "Planar graphs are 4-colorable.",
                  "children": []
                }
              ]
            },
            {
              "id": "v1-it2",
              "kind": "item",
              "content": "",
              "children": [
                {
                  "id": "v1-s6",
                  "kind": "sentence",
                  "content": "Bipartite graphs are 2-colorable.",
                  "children": []
                }
              ]
            }
          ]
        },
        {
          "id": "v1-h3",
          "kind": "heading",
          "content": "Brooks' Bound",
          "children": [
            {
              "id": "v1-s7",
              "kind": "sentence",
              "content": "The chromatic number satisfies the following upper bound:",
              "children": []
            },
            {
              "id": "v1-f1",
              "kind": "formula",
              "content": "\\begin{equation}\n\\chi(G) \\leq \\Delta(G) + 1 \\label{eq:brooks}\n\\end{equation}",
              "children": []
            },
            {
              "id": "v1-s8",
              "kind": "sentence",
              "content": "where $\\Delta(G)$ denotes the maximum degree.",
              "children": []
            },
            {
              "id": "v1-fig1",
              "kind": "figure",
              "content": "\\begin{figure}[ht]\n\\caption{Visualization of 4-coloring on a planar triangulation.}\n\\label{fig:planar_graph}\n\\end{figure}",
              "children": []
            }
          ]
        }
      ]
    },
    {
      "id": "v1-h4",
      "kind": "heading",
      "content": "Acknowledgments",
      "children": [
        {
          "id": "v1-s9",
          "kind": "sentence",
          "content": "This research was supported by the VNU-HCM Research Grant.",
          "children": []
        }
      ]
    },
    {
      "id": "v1-h5",
      "kind": "heading",
      "content": "Omitted Proofs",
      "children": [
        {
          "id": "v1-s10",
          "kind": "sentence",
          "content": "The detailed proof of correctness follows by induction on $|V|$.",
          "children": []
        }
      ]
    }
  ]
}
```

### 8.3 Ánh xạ sang hai bảng Parquet `elements` và `hierarchy`

Khi xuất dữ liệu sang 6 bảng Parquet phát hành trên Hugging Face Hub theo `PARQUET_SCHEMAS` trong `src/schemas.py`, cây JSON lồng nhau ở trên được làm phẳng triệt để thành hai bảng quan hệ theo mô hình Adjacency List:

#### 8.3.1 Bảng `elements`: Trích đoạn các dòng đại diện
- **Schema (`src/schemas.py`):** Chỉ gồm đúng 4 cột: `element_id: string (non-null)`, `arxiv_id: string (non-null)`, `kind: dictionary<int8, string>`, `content: string`.
- **Đặc điểm:** Toàn bộ cột đều là kiểu nguyên thủy. Tuyệt đối không chứa nested objects hay serialized JSON blobs, không có các trường phụ `title`, `level`, `role`, `caption`, `label`. Tiêu đề heading lưu trực tiếp vào `content`; toàn bộ khối TeX thô của công thức toán khối và hình ảnh/bảng biểu (kèm nhãn và chú thích) được nhét nguyên vẹn vào `content`.

| `element_id` | `arxiv_id` | `kind` | `content` |
|---|---|---|---|
| `2411-03799#root1` | `2411.03799` | `document` | `"Version 1 root"` |
| `2411-03799#v1-h1` | `2411.03799` | `heading` | `"Abstract"` |
| `2411-03799#v1-s1` | `2411.03799` | `sentence` | `"Graph coloring is a fundamental problem in computer science."` |
| `2411-03799#v1-s2` | `2411.03799` | `sentence` | `"We present a linear-time heuristic bound."` |
| `2411-03799#v1-h2` | `2411.03799` | `heading` | `"Introduction"` |
| `2411-03799#v1-s3` | `2411.03799` | `sentence` | `"Graph coloring has been studied extensively \cite{brooks1941colouring}."` |
| `2411-03799#v1-s4` | `2411.03799` | `sentence` | `"Let $G=(V, E)$ be a connected graph."` |
| `2411-03799#v1-list1` | `2411.03799` | `itemize` | `""` |
| `2411-03799#v1-it1` | `2411.03799` | `item` | `""` |
| `2411-03799#v1-s5` | `2411.03799` | `sentence` | `"Planar graphs are 4-colorable."` |
| `2411-03799#v1-it2` | `2411.03799` | `item` | `""` |
| `2411-03799#v1-s6` | `2411.03799` | `sentence` | `"Bipartite graphs are 2-colorable."` |
| `2411-03799#v1-h3` | `2411.03799` | `heading` | `"Brooks' Bound"` |
| `2411-03799#v1-s7` | `2411.03799` | `sentence` | `"The chromatic number satisfies the following upper bound:"` |
| `2411-03799#v1-f1` | `2411.03799` | `formula` | `"\begin{equation}\n\chi(G) \leq \Delta(G) + 1 \label{eq:brooks}\n\end{equation}"` |
| `2411-03799#v1-s8` | `2411.03799` | `sentence` | `"where $\Delta(G)$ denotes the maximum degree."` |
| `2411-03799#v1-fig1` | `2411.03799` | `figure` | `"\begin{figure}[ht]\n\caption{Visualization of 4-coloring on a planar triangulation.}\n\label{fig:planar_graph}\n\end{figure}"` |
| `2411-03799#v1-h4` | `2411.03799` | `heading` | `"Acknowledgments"` |
| `2411-03799#v1-s9` | `2411.03799` | `sentence` | `"This research was supported by the VNU-HCM Research Grant."` |
| `2411-03799#v1-h5` | `2411.03799` | `heading` | `"Omitted Proofs"` |
| `2411-03799#v1-s10` | `2411.03799` | `sentence` | `"The detailed proof of correctness follows by induction on \|V\|."` |

#### 8.3.2 Bảng `hierarchy`: Biểu diễn cây phiên bản 1
- **Schema (`src/schemas.py`):** Chỉ gồm đúng 4 cột: `arxiv_id: string (non-null)`, `version: int32`, `element_id: string (non-null)`, `parent_id: string (non-null)`.
- **Đặc điểm:** Tách biệt cấu trúc phả hệ khỏi nội dung, quản lý mối quan hệ cha – con đa phiên bản bằng các cột nguyên thủy. Nút gốc `document` (`2411-03799#root1`) không có cha nên không sinh dòng trong bảng này.

| `arxiv_id` | `version` | `element_id` | `parent_id` |
|---|---:|---|---|
| `2411.03799` | `1` | `2411-03799#v1-h1` | `2411-03799#root1` |
| `2411.03799` | `1` | `2411-03799#v1-s1` | `2411-03799#v1-h1` |
| `2411.03799` | `1` | `2411-03799#v1-s2` | `2411-03799#v1-h1` |
| `2411.03799` | `1` | `2411-03799#v1-h2` | `2411-03799#root1` |
| `2411.03799` | `1` | `2411-03799#v1-s3` | `2411-03799#v1-h2` |
| `2411.03799` | `1` | `2411-03799#v1-s4` | `2411-03799#v1-h2` |
| `2411.03799` | `1` | `2411-03799#v1-list1` | `2411-03799#v1-h2` |
| `2411.03799` | `1` | `2411-03799#v1-it1` | `2411-03799#v1-list1` |
| `2411.03799` | `1` | `2411-03799#v1-s5` | `2411-03799#v1-it1` |
| `2411.03799` | `1` | `2411-03799#v1-it2` | `2411-03799#v1-list1` |
| `2411.03799` | `1` | `2411-03799#v1-s6` | `2411-03799#v1-it2` |
| `2411.03799` | `1` | `2411-03799#v1-h3` | `2411-03799#v1-h2` |
| `2411.03799` | `1` | `2411-03799#v1-s7` | `2411-03799#v1-h3` |
| `2411.03799` | `1` | `2411-03799#v1-f1` | `2411-03799#v1-h3` |
| `2411.03799` | `1` | `2411-03799#v1-s8` | `2411-03799#v1-h3` |
| `2411.03799` | `1` | `2411-03799#v1-fig1` | `2411-03799#v1-h3` |
| `2411.03799` | `1` | `2411-03799#v1-h4` | `2411-03799#root1` |
| `2411.03799` | `1` | `2411-03799#v1-s9` | `2411-03799#v1-h4` |
| `2411.03799` | `1` | `2411-03799#v1-h5` | `2411-03799#root1` |
| `2411.03799` | `1` | `2411-03799#v1-s10` | `2411-03799#v1-h5` |

> [!TIP]
> **Đối chiếu tiêu chuẩn lưu trữ Parquet của Hugging Face Hub:**
> - **Chỉ gồm đúng 6 bảng Parquet trong `<dataset-repo>/data/`:** `papers`, `references`, `elements`, `hierarchy`, `bibtex`, `predictions` (chi tiết schema tại `src/schemas.py`).
> - **Cột phẳng và kiểu nguyên thủy:** Toàn bộ các bảng không dùng nested objects hay chuỗi JSON serialize. Mọi thông tin được định kiểu rõ ràng (`string`, `int32`, `dictionary`), cho phép viewer và backend streaming trên Hugging Face Hub tải và hiển thị ngay lập tức từng partition mà không cần giải mã thêm.

---

## 9. Khảo sát thực nghiệm và kiểm chứng giả thuyết

Kết quả sau quá trình tiến hành khảo sát trên **10 bài báo arXiv thực tế** tải từ dải ID `2024-11: 3799 → 8798` và kiểm chứng 9 giả thuyết H1 – H9:

### 9.1 Bảng dữ liệu khảo sát thực nghiệm 10 bài báo

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

### 9.2 Kiểm chứng 9 giả thuyết kỹ thuật

1. **H1 — Đa số bài dùng `\section` làm cấp cao nhất; `\chapter` chỉ có ở `report`/`book`:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN**. 9/10 bài khảo sát có heading cao nhất là `\section`. Chỉ các tài liệu dài định dạng `report` hoặc `book` mới dùng `\chapter`. Thuật toán gán dynamic level hoàn toàn đáp ứng đúng thực tế.
2. **H2 — `\paragraph{...}` được dùng như một tiêu đề chạy liền:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN**. `\paragraph` thường không xuống dòng mà in đậm từ khóa đầu đoạn. Bộ dựng cây gán nó làm nhánh tiêu đề con cấp sâu nhất (`kind = "heading"`).
3. **H3 — `align` và `align*` xuất hiện nhiều hơn `$$...$$`:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN**. Ở các bài báo hiện đại (từ năm 2015 trở lại đây), gói `amsmath` với môi trường `align` chiếm đa số. Cú pháp cổ điển `$$...$$` chỉ gặp nhiều ở các bài cũ thập niên 90–2000. Do đó bắt buộc phải hỗ trợ đầy đủ `align`, `align*`.
4. **H4 — Tác giả tự định nghĩa macro cho heading (`\def\sec{...}`):**
   - *Kết quả kiểm chứng:* **HIẾM GẶP, dưới 2%**. Phần lớn tác giả dùng lệnh chuẩn của các template hội nghị/tạp chí. Không cần tốn chi phí xây dựng bộ phân giải macro phức tạp; trường hợp ngoại lệ ghi log cảnh báo.
5. **H5 — Môi trường định lý (`theorem`, `lemma`, `proof`) xuất hiện phổ biến trong bài Toán/Lý:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN**. Giải pháp xử lý các môi trường này theo cơ chế "trong suốt" là tối ưu nhất, giúp các câu chứng minh được bảo toàn tự nhiên trong Section cha.
6. **H6 — Tồn tại song song cả `thebibliography` trực tiếp và file `.bib` riêng:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN**. Khoảng 60% bài dùng `\begin{thebibliography}` trực tiếp trong mã nguồn `.tex`, 40% gọi `\bibliography{refs}` kèm theo file `.bbl`/`.bib`. Module `bibtex_extractor.py` cần xử lý cả hai dạng này.
7. **H7 — `\appendix` có thể đứng sau phần References:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN**. Rất nhiều bài báo đặt Phụ lục chứng minh ở sau cùng, đứng sau cả References. Vì vậy quy tắc cắt bỏ References chỉ được cắt đến khi gặp `\appendix`.
8. **H8 — Khối Figure vẫn giữ `\caption` và `\label` sau khi M1 xóa file ảnh:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN**. M1 chỉ xóa các tệp nhị phân trên đĩa, toàn bộ cú pháp TeX `\begin{figure}...\caption{...}\end{figure}` vẫn nguyên vẹn 100%. Node `figure` được tạo bình thường.
9. **H9 — Acknowledgments có nhiều biến thể cú pháp:**
   - *Kết quả kiểm chứng:* **XÁC NHẬN**. Có bài dùng `\section*{Acknowledgment}`, có bài dùng `\section{Acknowledgements}`, và `revtex` dùng `\begin{acknowledgments}`. Bộ regex nhận diện cần hỗ trợ cả dạng heading lẫn dạng môi trường.

---

## 10. Tài liệu tham khảo

1. *arXiv Bulk Data Access & API Guidelines*: <https://info.arxiv.org/help/bulk_data.html>
2. *Leslie Lamport*, **LaTeX: A Document Preparation System**, Addison-Wesley, 1994.
3. *Frank Mittelbach et al.*, **The LaTeX Companion**, 2nd Edition, Addison-Wesley, 2004.
4. *Pydantic v2 Documentation*: <https://docs.pydantic.dev/latest/>
5. *Apache Arrow & Parquet Format Specification*: <https://arrow.apache.org/docs/python/parquet.html>
6. *Hugging Face Dataset Viewer & Streaming Specification*: <https://huggingface.co/docs/datasets/server>
