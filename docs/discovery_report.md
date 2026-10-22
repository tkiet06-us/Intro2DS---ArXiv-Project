# Khảo sát Discovery & Phương pháp Crawling (P1-00)

> **Dải ID:** `2506.00788` → `2506.05787` (5.000 bài)
> **Mẫu thử:** API và Kaggle dùng 50 ID liên tiếp `2506.00788` → `2506.00837` (Kaggle quét cả dải 5.000 ID); OAI-PMH dùng 20 ID đầu của mẫu đó · **Môi trường đo:** máy cá nhân, Windows 11, Python 3.14.0
> **Số liệu chính thức:** (1) **Chạy thật toàn dải** (`discovery.py --mode full`, 1 lần, 100 request API + quét Kaggle; chạy lúc `⟨điền: environment.run_utc trong discovery_full_result.json⟩`) là nguồn của mọi con số "tổng thời gian discovery" và của số đo API toàn dải; (2) **phiên A** (`discovery.py` benchmark, 3 lần lặp, mỗi lần một tiến trình riêng, chạy lúc 13:53 UTC ngày 20/10/2026, tức 20:53 giờ Việt Nam) là nguồn của số đo mẫu OAI-PMH, API và Kaggle. **Phiên B** là một lần chạy trước đó bằng bản script cũ, không ghi ngày và không có file kết quả đi kèm nên **không kiểm chứng được**: chỉ giữ làm ghi chú ở mục 2.4, không dùng làm số chính thức. Các lần đo đơn lẻ ngày 14/10 và 16/10 chỉ đo một lần nên không dùng.
>
> Ghi chú đọc số: thời gian phụ thuộc máy, mạng và **thời điểm đo**. Số của API và của tổng discovery là **số đo thật** từ lần chạy toàn dải; số "ước lượng 5.000 ID" của OAI được ngoại suy từ số đo mẫu (theo số ID). Thời gian API và OAI **đã gồm** sleep 3 s giữa các request; thời gian Kaggle **không gồm** thời gian tải file.

## 1. Mục tiêu

Chọn cách "discovery" (xác định bài nào tồn tại trong dải ID được giao, có bao nhiêu version, ngày của từng version, metadata cơ bản) cho Milestone 1. Hệ quả: đầu vào cho crawler (P2-01) và đợt crawl thật (22/10 → 28/10).

**Phạm vi theo đề.** Đề nêu như sau:

> The first step in any arXiv crawling workflow is to identify which papers to collect. In this project the assigned range arrives as plain arXiv IDs, so these methods are a toolbox rather than a task: the API resolves the assigned IDs into paper metadata and reveals each paper’s version list, while OAI-PMH and the Kaggle snapshot show how the same discovery is done at repository scale. Implementing a full repository-scale harvest is not part of this milestone.

Hệ quả cho báo cáo này: (1) dải ID được giao đã là danh sách ID cụ thể, nên discovery ở đây là *phân giải* các ID đó thành metadata và danh sách version, không phải tìm bài từ đầu; (2) arXiv API là công cụ chính cho việc này; (3) OAI-PMH và Kaggle snapshot được khảo sát để so sánh cách làm ở quy mô kho lưu trữ, nên báo cáo chỉ đo thử và ngoại suy, **không thực hiện thu hoạch toàn kho** (không chạy `ListRecords` toàn bộ, không xử lý snapshot ngoài dải được giao).

Yêu cầu của đề cần thỏa:
- Crawl **mọi version** `v1, v2, …` của mỗi bài.
- `metadata.json` có: tiêu đề, tác giả, ngày nộp v1 và **ngày của các lần revised** (ISO).
- Báo cáo M1 cần số liệu **tổng thời gian discovery**.

## 2. Ba phương pháp

### 2.1 arXiv API – tra theo lô `id_list`

- **Cách hoạt động:** `GET http://export.arxiv.org/api/query?id_list=id1,id2,...&max_results=N` trả về feed Atom, mỗi bài một `<entry>`.
- **Dữ liệu có:** tiêu đề, tác giả, abstract, category, `published` (= ngày **v1**), `updated` (= ngày **version mới nhất**). `<id>` có dạng `.../abs/2506.00788v3` → **v3 là version mới nhất ⇒ bài có 3 version**.
- **Không có:** ngày của các version trung gian (ví dụ v2 khi bài có v3).
- **Ưu:** truy vấn thẳng theo ID nên khớp hoàn toàn với việc được giao theo dải ID; dữ liệu luôn mới (nguồn chuẩn cho "bài có tồn tại không" và "đang ở version nào"); một request chứa được nhiều ID.
- **Nhược:** giới hạn ≥ 3 giây/request, 1 kết nối; ID không tồn tại thì bị bỏ qua trong feed, cần so khớp với danh sách đã hỏi; thiếu ngày các version trung gian; thời gian phản hồi dao động mạnh theo thời điểm.
- **Thời gian đo được (phiên A, 3 lần lặp, mỗi lần 1 request cho 50 ID):** **0,99 ± 0,26 s** (khoảng [0,71; 1,22] s, median 1,02 s), tìm thấy 50/50 ID. Mỗi lần chỉ có một request nên con số này không gồm sleep. RAM đỉnh 46,5 ± 0,2 MB (RAM nền 40,8 MB).
  - Cùng một request, phiên B (mục 2.4, không kiểm chứng được) cho 12,71 ± 10,67 s. Thời gian phản hồi của server phụ thuộc mạnh vào thời điểm đo; lần chạy thật 100 request bên dưới cho thấy đúng hiện tượng này.
- **Đo thật toàn dải 5.000 ID (`--mode full`, 1 lần):** rate limit tính theo **số request**, không theo số ID; batch 50 ⇒ **100 request**, mỗi request = thời gian xử lý + 3 s sleep bắt buộc.
  - **Tổng thời gian API: 784,2 s ≈ 13,1 phút** (wall-clock, gồm sleep, parse feed và 1 lần tự thử lại do `ConnectionResetError`). Nhận đủ 5.000/5.000 bài.
  - **Thời gian mỗi request (50 ID, 100 request):** trung bình 2,57 s, median 1,35 s, std 5,17 s, [0,94; 43,40] s. Phân bố lệch phải: 95/100 request dưới 5 s; 5 request chậm (7,7; 16,9; 17,5; 22,4 và 43,4 s) chiếm khoảng 108 s trong tổng 256,9 s thời gian request.
  - **Cấu thành:** 256,9 s (tổng thời gian request, theo log) + 297 s (99 lần sleep 3 s danh nghĩa) = 553,9 s; **còn ≈ 230 s (≈ 2,3 s mỗi vòng lặp) chưa giải thích được**. Báo cáo dùng số wall-clock 784,2 s vì đó là số đo thật.
  - **So với ước lượng từ mẫu:** phiên A ước lượng 100 × (0,99 + 3) ≈ 399 s ≈ 6,6 phút, nên số đo thật **gấp khoảng 2 lần**. Mẫu 3 request không bắt được đuôi chậm của phân bố, nên ngoại suy từ mẫu nhỏ đã đánh giá thấp thời gian thật.

### 2.2 OAI-PMH (thư viện `sickle`)

- **Endpoint:** `https://oaipmh.arxiv.org/oai`. Endpoint cũ `export.arxiv.org/oai2` đã được thay (trang hướng dẫn của arXiv ghi rõ base URL đã đổi; theo tài liệu môn, từ 03/2025).
- **Cách hoạt động:** hai verb liên quan:
  - `GetRecord` (identifier `oai:arXiv.org:<id>`) – lấy **một** bài.
  - `ListRecords` – thu hoạch hàng loạt, lọc theo **ngày (`from`/`until`) và `set`**, phân trang bằng `resumptionToken`. `from`/`until` lọc theo *datestamp* (lần sửa metadata gần nhất), **không phải ngày nộp bài**; arXiv ghi rõ OAI-PMH không hỗ trợ thu hoạch chọn lọc theo ngày nộp.
- **Format `arXivRaw`:** chứa danh sách `<version version="vN">` với **ngày và kích thước từng version** ⇒ nguồn chính xác nhất cho "số version" và "ngày từng version".
- **Ưu:** đủ ngày + size mọi version trong một nguồn duy nhất; là chuẩn thu hoạch metadata.
- **Nhược:** `ListRecords` **không lọc được theo dải ID**, chỉ theo ngày/set; muốn lấy dải 788 → 5787 của 2025-06 phải quét cả tháng rồi bỏ phần thừa (đây chính là thu hoạch quy mô kho mà đề không yêu cầu). `GetRecord` thì 1 request/ID, nên 5.000 ID là 5.000 request, bị chặn dưới bởi sleep 3 s. arXiv dùng `503 Retry-After` để điều tiết.
- **Thời gian đo được (phiên A, endpoint mới, 3 lần lặp, mỗi lần 20 `GetRecord`):** **73,78 ± 6,72 s** (khoảng [66,18; 78,95] s), tìm thấy 20/20 ID, đã gồm 19 lần sleep 3 s (57 s). Thời gian request thực trung bình ≈ **0,84 s**, nên chi phí mỗi ID ≈ **3,84 s** (gồm sleep). RAM đỉnh 46,6 ± 0,2 MB (RAM nền 41,9 MB).
- **So với endpoint cũ:** lần đo trước (endpoint `export.arxiv.org/oai2`) là 14,37 s/ID (431,14 s cho 30 ID). Endpoint mới nhanh hơn khoảng **3,7 lần** (3,84 s/ID, phiên A).
- **Ước lượng 5.000 ID:**
  - Từ số đo phiên A: 5.000 × 3,84 ≈ **19.192 s ≈ 5,3 giờ**. Khoảng tin cậy 95% của trung bình: 18.423–19.962 s (5,1–5,5 giờ); khoảng [min, max] (cực đoan): 16.571–27.032 s (4,6–7,5 giờ).
  - Cận dưới do sleep: 5.000 × 3 s ≈ 4,2 giờ. Với endpoint cũ là ≈ 20 giờ.

### 2.3 Kaggle dataset (`Cornell-University/arxiv`, đọc bằng pandas)

- **Cách hoạt động:** tải file `arxiv-metadata-oai-snapshot.json` (JSON Lines), đọc bằng `pd.read_json(lines=True, chunksize=...)`, lọc `id` bắt đầu bằng `2506.` và phần số trong `[788, 5787]`.
- **Dữ liệu có:** `id, title, authors, categories, doi, journal-ref, update_date` và `versions: [{version, created}]` ⇒ **có số version và ngày từng version** trong một lần đọc.
- **Ưu:** không tốn request nào tới arXiv, không dính giới hạn 3 s; lọc dải ID chỉ là thao tác pandas; một lần đọc là xong cả 5.000 bài.
- **Nhược:** là snapshot (cập nhật định kỳ), bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu; file lớn (1,73 GB nén, 5,61 GB sau khi giải nén), cần tài khoản/API key Kaggle; không có thông tin source `.tar.gz`.
- **Thời gian đo được:**
  - **Tải file:** **1 giờ 23 phút** (≈ 4.980 s) cho file 1,73 GB nén, tức tốc độ trung bình ≈ **350 kB/s** (1,73 GB / 4.980 s). Chi phí **một lần**, đo một lần duy nhất, phụ thuộc mạng.
  - **Đọc + lọc dải 5.000 ID (phiên A, 3 lần lặp):** **66,76 ± 7,58 s** (khoảng [61,44; 75,44] s, median 63,40 s). Tìm thấy **5.000/5.000** ID trong dải ở cả 3 lần, và 50/50 ID của mẫu.
  - Lần lặp đầu (75,44 s) chậm hơn rõ so với hai lần sau (61,44 s và 63,40 s), nhiều khả năng do cache hệ điều hành chưa chứa file 5,61 GB. Đọc "nóng" ≈ 61–64 s; lần đọc đầu có thể chậm hơn khoảng 10 s. Phiên B (64,44 ± 0,72 s) không có hiện tượng này.
  - **Tổng end-to-end:** ≈ 5.047 s ≈ **1 giờ 24 phút** (gồm cả tải).
  - **RAM đỉnh** khi đọc (chunksize `200_000`): **2061,2 ± 4,0 MB** (mean ± std của 3 lần lặp; lần cao nhất 2064,1 MB). RAM nền sau khi import pandas: 78,2 MB.
  - **Chạy thật toàn dải (`--mode full`, 1 lần):** quét + lọc **73,3 s** (≈ 76,5 MB/s), tìm thấy **5.000/5.000** ID; RAM đỉnh **3.264,0 MB** (RAM nền 80,4 MB, chunksize `200_000`). Cao hơn khoảng 1,2 GB so với 2.061,2 MB của benchmark phiên A và **chưa giải thích được** chênh lệch này; chế độ full gọi `read_json(..., convert_dates=False)` còn benchmark thì không. Cả hai đều dưới ngưỡng 6.000 MB.
- **Đối chiếu giữa các nguồn trên mẫu:** 50 ID được đối chiếu số version, **0 ID lệch**, và 0 ID đổi version giữa các lần lặp của cùng một nguồn (API và Kaggle phủ 50 ID; OAI phủ 20 ID đầu). Đây chỉ là mẫu 1% dải; kết quả đối chiếu toàn dải 5.000 ID (33 ID lệch `n_versions`) nằm ở mục 4.1.

### 2.4 Độ nhạy theo thời điểm đo

Phần chạy cục bộ (đọc Kaggle) ổn định, còn phần phụ thuộc server arXiv (API, OAI) dao động mạnh theo thời điểm. **Phiên B chỉ là ghi chú** ("lần đo trước chậm hơn nhiều"): chạy bằng script cũ, không ghi ngày, không có file kết quả đi kèm nên không kiểm chứng được, và **không** dùng làm cận trên chính thức.

| Chỉ số | Phiên A (20/10, benchmark) | Phiên B (script cũ, không ghi ngày) | Chạy thật toàn dải (mục 2.1, 4.1) |
|---|---|---|---|
| API, thời gian 1 request (50 ID) | 0,99 ± 0,26 s (3 request) | 12,71 ± 10,67 s | TB 2,57 s, median 1,35 s, [0,94; 43,40] s (100 request) |
| OAI, thời gian 1 request | ≈ 0,84 s (3,84 s/ID gồm sleep) | ≈ 2,94 s (5,94 s/ID) | không chạy |
| Kaggle, đọc + lọc cả dải | 66,76 ± 7,58 s | 64,44 ± 0,72 s | 73,3 s (1 lần) |
| Kaggle, RAM đỉnh | 2061,2 ± 4,0 MB | 2048,7 ± 18,9 MB | 3264,0 MB |
| API, 5.000 ID | 399 s ≈ 6,6 phút (ước lượng) | 1.571 s ≈ 26,2 phút (ước lượng) | **784,2 s ≈ 13,1 phút (đo thật)** |
| OAI, 5.000 ID (ước lượng) | 19.192 s ≈ 5,3 giờ | 29.686 s ≈ 8,2 giờ | không chạy |
| Discovery kết hợp, chưa tính tải | ≈ 465 s ≈ 7,8 phút (ước lượng) | ≈ 1.635 s ≈ 27,3 phút (ước lượng) | **857,5 s ≈ 14,3 phút (đo thật)** |
| Discovery kết hợp, tính cả tải | ≈ 5.445 s ≈ 1 giờ 31 (ước lượng) | ≈ 6.615 s ≈ 1 giờ 50 (ước lượng) | **≈ 5.837,5 s ≈ 1 giờ 37 (đo thật)** |

Nhận xét:
- Thời gian đọc Kaggle ổn định giữa các lần đo (61–75 s); RAM đỉnh thì khác giữa benchmark và chạy thật (mục 2.3).
- Ước lượng từ 3 request (399 s) thấp hơn số đo thật (784,2 s) khoảng 2 lần, vì mẫu nhỏ không bắt được các request chậm (đến 43 s).
- Độ lệch chuẩn *trong* một phiên (và khoảng tin cậy tính từ nó) nhỏ hơn nhiều so với chênh lệch *giữa* các lần đo, nên khoảng tin cậy của một phiên đơn lẻ đánh giá thấp độ bất định thật.
- **Kết luận chọn phương án không đổi:** OAI một mình (≈ 19.192 s, ước lượng phiên A) đắt hơn phương án kết hợp đo thật khoảng **22 lần** khi chưa tính tải và **3,3 lần** khi tính cả tải lần đầu.
- **Con số đưa vào báo cáo M1:** discovery ≈ **14,3 phút** (857,5 s, chưa tính tải) và ≈ **1 giờ 37 phút** (≈ 5.837,5 s) nếu tính cả tải lần đầu. Đây là kết quả của một lần chạy; thời gian API phụ thuộc thời điểm nên cần chạy lặp ở các thời điểm khác nhau để có mean ± std.

## 3. Bảng so sánh

| Tiêu chí | arXiv API (`id_list`) | OAI-PMH (`sickle`) | Kaggle snapshot |
|---|---|---|---|
| Truy vấn theo dải ID | Có, trực tiếp | Không (`ListRecords`); 1 ID/request (`GetRecord`) | Có, lọc bằng pandas |
| Số version | Có, từ hậu tố `vN` của `<id>` | Có, đếm `<version>` | Có, `len(versions)` |
| Ngày từng version | Không, chỉ v1 và mới nhất | Có (kèm size) | Có (`created`) |
| Phụ thuộc rate limit 3 s | Có | Có | Không |
| Độ mới dữ liệu | Realtime | Realtime | Theo snapshot |
| Số request cho 5.000 bài | ~100 (batch 50) | ~5.000 (`GetRecord`) | 0 |
| Thời gian đo (phiên A, mean ± std, 3 lần) | 0,99 ± 0,26 s (50 ID, 1 request) | 73,78 ± 6,72 s (20 ID, 20 request) | 66,76 ± 7,58 s (đọc + lọc cả dải 5.000 ID) |
| RAM đỉnh khi đo | 46,5 MB | 46,6 MB | 2061,2 ± 4,0 MB (benchmark); 3264,0 MB (chạy thật) |
| Thời gian chuẩn bị / tải | 0 | 0 | 4.980 s (1 giờ 23 phút), một lần |
| Cho 5.000 ID (đo thật hoặc ước lượng) | **784,2 s ≈ 13,1 phút (đo thật, 100 request)**; ước lượng phiên A từ mẫu: 399 s | ≈ 19.192 s (≈ 5,3 giờ), ước lượng phiên A | **73,3 s (đo thật, chưa tính tải)**; ≈ 5.053 s (tính cả tải) |

> Chú thích: cột Kaggle phải đọc kèm dòng "thời gian chuẩn bị / tải". Chi phí tải (1 giờ 23 phút) là một lần và có thể chia sẻ cho cả nhóm qua Drive, nhưng vẫn là chi phí thật của phương pháp này. Số mẫu của ba cột khác nhau (50 ID / 20 ID / cả dải) nên chỉ so sánh được qua dòng "Cho 5.000 ID".

## 4. Kết luận & lựa chọn

### 4.1 Phương án chọn

**Kết hợp hai nguồn trên toàn bộ 5.000 ID: arXiv API là nguồn chuẩn cho "bài tồn tại" và "số version"; Kaggle snapshot cung cấp ngày từng version và metadata; OAI-PMH `GetRecord` chỉ vá các ca lệch.** Khi hai nguồn mâu thuẫn, lấy API làm chuẩn.

Quy trình:
1. **Quét Kaggle (1 lần):** đọc snapshot bằng pandas theo chunk, lọc `2506.00788 → 2506.05787`. Thu được ID, số version và ngày từng version cho cả 5.000 bài (đo thật 73,3 s).
2. **Chạy arXiv API cho toàn bộ 5.000 ID** (100 request, batch 50, đo thật 784,2 s ≈ 13 phút): xác nhận bài tồn tại và lấy `vN` mới nhất.
3. **So `n_versions` giữa API và Kaggle, lấy API làm chuẩn:**
   - Khớp ⇒ dùng ngày version từ Kaggle.
   - API có `vN` mới hơn Kaggle (snapshot cũ) hoặc bài chỉ có ở API ⇒ gọi OAI `GetRecord` (`arXivRaw`) để lấy ngày các version còn thiếu.
   - Lý do chạy toàn bộ thay vì chỉ lấy mẫu: nếu snapshot thiếu `v(N+1)` thì crawler sẽ bỏ sót version, vi phạm yêu cầu "mọi version". Chi phí chạy hết đo thật ≈ 13 phút nên không đáng để tiết kiệm.

**Tổng thời gian discovery (số liệu cho báo cáo M1), 5.000 ID, đo thật một lần chạy `--mode full`:**

| Thành phần | Thời gian đo thật | Ghi chú |
|---|---|---|
| Quét Kaggle | 73,3 s | một lần quét cả dải, ≈ 76,5 MB/s |
| arXiv API toàn dải | 784,2 s (≈ 13,1 phút) | 100 request, gồm sleep 3 s; thời gian request trung bình 2,57 s (mục 2.1) |
| Ghi `discovery.jsonl` | ≈ 0 s | 5.000 dòng |
| **Cộng (chưa tính tải)** | **857,5 s ≈ 14,3 phút** | wall-clock end-to-end của một tiến trình |
| Tải file Kaggle | 4.980 s (1 giờ 23 phút) | một lần, chia sẻ cho nhóm |
| **Cộng (lần đầu, tính cả tải)** | **≈ 5.837,5 s ≈ 1 giờ 37 phút** | |
| Vá bằng OAI | ≈ 3,84 s/bài (phiên A) | tối đa 33 bài ≈ 127 s ≈ 2,1 phút; ví dụ 5% (250 bài) ≈ 16 phút |

Số đo này thay cho khoảng ước lượng "8–27 phút" ở bản trước: cận trên 27 phút dựa trên phiên B không kiểm chứng được nên đã bỏ. Đây là một lần chạy; thời gian API phụ thuộc thời điểm (mục 2.4) nên cần chạy lặp để có mean ± std.

**Số bài lệch API–Kaggle trên toàn dải: 33/5.000 (0,66%)** lệch `n_versions` (lấy API làm chuẩn); lệch ngày v1 (`published` của API so với `v1` của Kaggle): **0 ID**. Mười ID lệch đầu danh sách: `2506.00948`, `2506.01228`, `2506.01448`, `2506.01495`, `2506.01518`, `2506.01540`, `2506.01589`, `2506.02434`, `2506.02514`, `2506.02620`; danh sách đầy đủ nằm trong `discovery_full_result.json` (`check.n_versions_mismatch_api_vs_kaggle`). Hướng lệch: `⟨điền: bao nhiêu/33 ID có Kaggle ít version hơn API⟩`. Kết quả này nhất quán với 0/50 ở mẫu ban đầu vì ID lệch nhỏ nhất là `2506.00948`, nằm ngoài mẫu 50 ID đầu (`2506.00788` → `2506.00837`).

### 4.2 Ưu điểm của phương án này

| # | Ưu điểm | Giải thích |
|---|---|---|
| 1 | **Khớp bài toán "theo dải ID"** | Cả API và Kaggle đều lấy được đúng dải. OAI-PMH `ListRecords` chỉ lọc theo ngày/set. |
| 2 | **Đủ dữ liệu mà ít request** | Kaggle cho ngày từng version với 0 request. API chỉ cần khoảng 100 request cho 5.000 bài, thay vì 5.000 request của OAI `GetRecord`. |
| 3 | **Không bỏ sót version** | API là nguồn realtime làm chuẩn nên bắt được version mới hơn snapshot. Đây là điều kiện để đáp ứng "crawl mọi version". |
| 4 | **Tiết kiệm thời gian ở bước crawl thật** | Biết trước `n_versions` thì crawler tải đúng `v1..vN`, không phải dò `v(N+1)` để gặp 404. Mỗi lần dò tốn ≥ 3 s, nhân 5.000 bài ≈ 4 giờ lãng phí. |
| 5 | **Có kiểm chứng chéo** | Hai nguồn độc lập đối chiếu nhau trên toàn bộ dải, không chỉ trên mẫu. |
| 6 | **Nhanh hơn OAI-only** | ≈ 14,3 phút (chưa tính tải) hoặc ≈ 1,6 giờ (tính cả tải lần đầu), so với ≈ 5,3 giờ (ước lượng phiên A) nếu chỉ dùng OAI `GetRecord` cho cả dải. |

### 4.3 Nhược điểm và cách giảm thiểu

| # | Nhược điểm | Cách giảm thiểu |
|---|---|---|
| 1 | **Tải file Kaggle rất lâu** (1 giờ 23 phút, tốc độ trung bình ≈ 350 kB/s), file 1,73 GB nén (5,61 GB giải nén), cần tài khoản/API key. | Chỉ một người tải một lần và chia sẻ qua Drive chung. Báo cáo ghi rõ chi phí này, không gộp vào thời gian đọc. Nếu chỉ tính riêng dải 5.000 ID của một người thì API một mình (≈ 13 phút, đo thật) rẻ hơn tổng end-to-end Kaggle (≈ 84 phút: 73,3 s quét + 4.980 s tải); Kaggle có lợi nhờ ngày version và việc dùng lại cho nhiều thành viên. |
| 2 | **Snapshot có thể cũ**: bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu hoặc lệch. | Chạy API toàn dải, lấy API làm chuẩn, vá ngày version bằng OAI `GetRecord`. Trên toàn dải đo được 33/5.000 (0,66%) bài lệch `n_versions` (mục 4.1). Ghi rõ phiên bản và ngày snapshot (mục 7.2). |
| 3 | **Vá bằng OAI tốn thời gian** (≈ 3,84 s/bài, phiên A). | Chỉ áp dụng cho bài lệch; ước lượng chi phí theo tỉ lệ lệch ở mục 4.5. |
| 4 | **Thời gian phản hồi của API dao động mạnh theo thời điểm**: trong lần chạy thật, một request 50 ID mất từ 0,94 s đến 43,4 s (median 1,35 s; 5/100 request trên 5 s) và có 1 lần `ConnectionResetError` được thử lại thành công. | Lập kế hoạch theo số đo thật ≈ 13 phút cho API và dự phòng gấp đôi; `discovery.py` tự thử lại tối đa 4 lần với backoff; crawler có Checkpoint/Resume. |
| 5 | **Không có thông tin source** (`.tar.gz`/`.gz`). | Không thuộc phạm vi discovery. Source vẫn lấy từ `export.arxiv.org/src/{id}v{n}` ở bước crawl. |
| 6 | **Ngày ở dạng RFC 2822**, có thể lệch múi giờ giữa nguồn. | Chuẩn hóa sang ISO 8601 (UTC) bằng một hàm duy nhất, kiểm tra khớp trên mẫu. |
| 7 | **Phụ thuộc nguồn bên thứ ba** (Kaggle/Cornell). | Nếu Kaggle không dùng được, API + OAI `GetRecord` thay thế (chậm hơn, ≈ 5,3 giờ cho phần OAI). |

### 4.4 Vì sao không chọn một phương pháp đơn lẻ

- **Chỉ arXiv API:** nhanh (≈ 13 phút, đo thật) và chuẩn về số version, nhưng thiếu ngày version trung gian. Đó là lý do cần nguồn thứ hai.
- **Chỉ OAI-PMH:** `arXivRaw` đầy đủ nhất và chỉ cần một nguồn (đơn giản hơn). Nhưng `ListRecords` không lọc được theo dải ID và buộc quét cả tháng (thu hoạch quy mô kho, ngoài phạm vi milestone); còn `GetRecord` đo được ≈ 3,84 s/ID (phiên A), ngoại suy ≈ 5,3 giờ cho 5.000 ID (nhanh hơn khoảng 3,7 lần so với ≈ 20 giờ ở endpoint cũ, nhưng vẫn gấp khoảng 22 lần phương án kết hợp đo thật khi chưa tính tải, và khoảng 3,3 lần khi tính cả tải lần đầu). Phù hợp làm phương án dự phòng và để vá lẻ.
- **Chỉ Kaggle:** đủ dữ liệu nhưng không đảm bảo độ mới; rủi ro bỏ sót version.

### 4.5 Khi nào đổi phương án

Chi phí vá bằng OAI tăng tuyến tính theo số bài lệch (≈ 3,84 s/bài, phiên A). Ví dụ 5% (250 bài) ≈ 16 phút, 10% (500 bài) ≈ 32 phút; với tỉ lệ lệch đo được 0,66% (33 bài) chỉ ≈ 2,1 phút. **Trường hợp xấu nhất là mọi bài đều lệch:** tổng ≈ 857,5 + 19.192 ≈ 20.050 s ≈ 5,6 giờ, chỉ hơn OAI `GetRecord` một mình khoảng 14 phút (phần API + quét Kaggle).

Về thời gian, phương án kết hợp vẫn rẻ hơn OAI một mình cho tới khi tỉ lệ lệch đạt khoảng **95%** (chưa tính tải Kaggle) hoặc khoảng **70%** (tính cả tải lần đầu); tính từ số đo thật và số đo OAI phiên A. Với tỉ lệ lệch đo được 0,66%, ngưỡng này còn rất xa. Vì vậy việc bỏ bước Kaggle khi tỉ lệ lệch cao (ví dụ trên 50%) là quyết định vì **đơn giản hóa quy trình**, không phải vì tiết kiệm thời gian. Ngưỡng ngoại suy từ số đo OAI mẫu, nên mang độ bất định của mẫu 20 ID.

## 5. Ghi chú kỹ thuật phát hiện được

- **Số version của một bài:** từ API lấy hậu tố `vN` trong `<id>` của entry; từ Kaggle `len(versions)`; từ OAI đếm các phần tử `<version>`.
- **Ngày từng version:** API chỉ có `published` (v1) và `updated` (mới nhất); Kaggle `versions[i].created` và OAI `arXivRaw` `<version><date>` đều có đủ. Định dạng là RFC 2822 (`Mon, 2 Jun 2025 12:34:56 GMT`), cần đổi sang **ISO 8601** khi ghi `metadata.json`.
- **Rate limit:** ≥ 3 giây giữa hai request tới arXiv, 1 kết nối tại một thời điểm, dùng `export.arxiv.org` (không dùng `arxiv.org`) và **User-Agent mô tả rõ** kèm email liên hệ. Semantic Scholar: delay 1 s + API key.
- **Rate limit tính theo request, không theo ID:** batch 30 hay 50 ID thì số request mới quyết định thời gian, nên ước lượng tổng thời gian dựa trên số request (`ceil(5000/50) = 100`). Tuy nhiên thời gian xử lý mỗi request dao động mạnh theo thời điểm (0,94 s đến 43,4 s trong lần chạy thật 100 request).
- **Batch `id_list`:** dùng batch ~50 ID/request; luôn đặt `max_results` bằng kích thước batch, vì mặc định chỉ trả 10.
- **Endpoint OAI-PMH:** dùng `https://oaipmh.arxiv.org/oai`; endpoint cũ chậm hơn khoảng 3,7 lần trong các lần đo của nhóm (14,37 s/ID so với 3,84 s/ID ở phiên A).
- **ID không tồn tại / bị rút:** API bỏ qua entry, OAI trả `idDoesNotExist`; cần log và giữ record rỗng theo yêu cầu đề.
- **Định dạng ID:** từ 2015 là `yymm.NNNNN` (5 chữ số, padding 0, ví dụ `2506.00788`); ID cũ như `hep-th/9901001` chỉ xuất hiện ở phần reference.
- **Source:** nguồn LaTeX lấy qua `export.arxiv.org/src/{id}v{n}`, có thể là `.tar.gz` hoặc một `.gz` đơn chứa `.tex`; bài chỉ có PDF/Word không có source ⇒ giữ record rỗng.

## 6. Việc tiếp theo (phụ thuộc chuỗi P1-00 → P2-01 → P3-02a)

- [x] Điền RAM đỉnh khi đọc Kaggle (mục 2.3).
- [x] Chạy lại `discovery.py` bản mới (20/10/2026), có đủ môi trường, SHA-256 và kiểm tra tự động.
- [x] Chạy `discovery.py --mode full` toàn dải: 5.000/5.000 bài, 33 bài lệch `n_versions` (mục 4.1); số đo thời gian thật thay cho ước lượng 7–26 phút.
- [x] Bàn giao interface discovery cho crawler engine (P2-01): đầu vào `--id_start --id_end` (ID đầy đủ, dải có thể qua nhiều tháng), đầu ra `data/raw/discovery.jsonl`, mỗi bài một dòng `{arxiv_id, n_versions, v1_date, latest_version_date, version_dates, title, authors}`.
- [ ] Điền **phiên bản dataset** và **ngày cập nhật** của Kaggle snapshot (mục 7.2). Lần chạy 20/10 truyền nguyên văn `Version N` và `2026-MM-DD` nên kết quả chưa có thông tin này.
- [ ] Xác định hướng lệch của 33 ID (Kaggle thiếu version mới hay ngược lại) từ `discovery_full_result.json` và vá ngày version bằng OAI `GetRecord` cho các ID đó.
- [ ] Chạy lặp `--mode full` thêm 1–2 lần ở thời điểm khác để có mean ± std (đặt `--out` và `--report` khác nhau mỗi lần để không ghi đè); làm rõ hai điểm chưa giải thích: RAM đỉnh 3.264 MB so với 2.061 MB, và ≈ 230 s chưa rõ nguồn ở bước API (mục 2.1).


## 7. Thông tin để tái lập

> Các trường `⟨điền: …⟩` là thông tin chưa có trong kết quả chạy.

### 7.1 Môi trường đo

- Máy cá nhân, Windows 11 (`Windows-11-10.0.26200-SP0`), Python 3.14.0.
- Phiên A chạy lúc 2026-10-20 13:53:25 UTC. Commit git của repo lúc chạy: `88af6de` (lúc đó `discovery.py` chưa được commit, nên commit này không chứa script).
- Chạy thật toàn dải (`--mode full`): chạy lúc `⟨điền: environment.run_utc⟩`, commit git `⟨điền: environment.git_commit⟩`, Python và phiên bản thư viện `⟨điền: environment.libs⟩` (đều lấy từ `discovery_full_result.json`); cùng máy cá nhân Windows.
- Mẫu: 50 ID liên tiếp `2506.00788` → `2506.00837` cho API và Kaggle; 20 ID đầu của mẫu đó cho OAI (`--oai-n 20`). Kaggle quét cả dải `788 → 5787`.
- Tham số cố định: API batch 50 ID/request với `max_results=len(batch)`, sleep 3 s giữa hai request; OAI `GetRecord` với `metadataPrefix=arXivRaw`, sleep 3 s giữa hai ID, `Sickle(max_retries=3)`; Kaggle `pd.read_json(lines=True, chunksize=200_000, dtype=False)`. Mọi request gửi `User-Agent` mô tả dự án kèm email liên hệ.

### 7.2 Kaggle snapshot

Kết quả lệch API–Kaggle phụ thuộc trực tiếp vào độ cũ của snapshot: bài có version mới hơn thời điểm snapshot sẽ bị Kaggle thiếu `v(N+1)`.

| Mục | Giá trị |
|---|---|
| Dataset | `Cornell-University/arxiv` |
| File | `arxiv-metadata-oai-snapshot.json` (1,73 GB nén; **5,61 GB** sau khi giải nén), đặt ở `data/raw/` |
| Phiên bản dataset trên Kaggle | `⟨điền: Version N⟩` |
| Ngày cập nhật của phiên bản đó | `⟨điền: dd/mm/yyyy⟩` |
| Ngày tải về | `⟨điền: dd/mm/yyyy⟩` (thời điểm sửa file trên máy: 14/10/2026 16:55:21 UTC) |
| SHA-256 của file sau khi giải nén | `9ee5aafa99209d6dbf4cd8c138dc662f290a7b49065d7301d4a0152bc0701c75` (lần chạy thật cũng dùng `--hash-kaggle`; giá trị phải khớp `environment.kaggle_file.sha256` trong `discovery_full_result.json`) |

Cách lấy: số phiên bản và ngày cập nhật nằm ở trang dataset trên Kaggle (mục *Version* / *Updated*); chọn đúng phiên bản có hiệu lực vào ngày tải về. SHA-256 cho phép người khác kiểm tra họ đang dùng đúng file này. Thời điểm sửa file (mtime) chỉ cho biết lần ghi gần nhất trên máy (có thể là lúc giải nén hoặc sao chép), **không phải** ngày của snapshot.

### 7.3 Script đo và cách đo

- **Script:** `src/m1_crawling/discovery.py`. Lệnh chạy (từ thư mục gốc repo):

```
py src\m1_crawling\discovery.py --contact "<email>" ^
    --kaggle-json data\raw\arxiv-metadata-oai-snapshot.json ^
    --download-seconds 4980 --repeats 3 ^
    --kaggle-version "<Version N>" --kaggle-snapshot-date <YYYY-MM-DD> ^
    --hash-kaggle
```

  Đầu ra (ghi vào thư mục đang đứng khi chạy): `discovery_benchmark_result.json` (số liệu thô, gồm môi trường và cấu hình; email được che) và `discovery_results.md` (bảng dán vào báo cáo). Có thể chạy riêng từng phương pháp bằng `--methods api` / `oai` / `kaggle`.
- **Chạy thật toàn dải (nguồn của số chính thức về tổng thời gian discovery):**

```
py src\m1_crawling\discovery.py --mode full ^
    --id-start 2506.00788 --id-end 2506.05787 --contact "<email>" ^
    --kaggle-json data\raw\arxiv-metadata-oai-snapshot.json --hash-kaggle
```

  Các cờ nhận cả dạng gạch nối và gạch dưới (`--id_start`, `--id_end`, `--kaggle_json`); ID là ID đầy đủ `YYMM.NNNNN`, dải có thể qua nhiều tháng. Đầu ra: `data/raw/discovery.jsonl` (interface cho P2-01) và `discovery_full_result.json` (thời gian từng giai đoạn, log từng request, RAM nền/đỉnh, tốc độ, danh sách ID lệch/thiếu, SHA-256 của `discovery.jsonl`, môi trường; email được che). Quy trình: quét Kaggle một lần → gọi API theo batch 50 (tối đa 4 lần thử mỗi request với backoff; ID thiếu được hỏi lại 1 lần) → đối chiếu `n_versions` và ngày v1 → ghi file. Thời gian đo bằng `perf_counter()` cho cả tiến trình, **không gồm thời gian tải file Kaggle** (gộp được bằng `--download-seconds`). RAM đỉnh đọc ngay sau bước quét Kaggle (giai đoạn tốn RAM nhất) và cuối chương trình, bằng cùng cách đo ở dưới.
- **Thiết kế đo:** mỗi cặp (phương pháp, lần lặp) chạy trong **một tiến trình riêng** để RAM đỉnh không lẫn giữa các phương pháp. Lặp 3 lần, xen kẽ theo thứ tự api → oai → kaggle → api → … để giảm ảnh hưởng của mạng. Báo cáo mean, median, std, min, max.
- **Đo thời gian:** `time.perf_counter()` (thời gian thực, wall-clock) bao toàn bộ vòng lặp của worker. Với API và OAI, thời gian **đã gồm** sleep 3 s giữa các request; thời gian riêng từng request được ghi trong `req_seconds`. Với Kaggle, chỉ gồm đọc + lọc, **không gồm tải file** (nhập tay bằng `--download-seconds`).
- **Ước lượng 5.000 ID:** API = `ceil(5000/50)` request × (thời gian request + 3 s); OAI = 5.000 × (thời gian request + 3 s). Script xuất ba cột: giá trị trung bình, khoảng tin cậy 95% của trung bình (phân phối t, theo số mẫu request) và khoảng [min, max] (kịch bản cực đoan). Khi dưới 10 mẫu request, script in cảnh báo. Khoảng tin cậy chỉ phản ánh dao động trong một phiên đo (xem mục 2.4).
- **Kiểm tra tự động (phiên A đều đạt):** Kaggle tìm đủ 5.000/5.000 ID trong dải ở cả 3 lần; API và OAI tìm đủ ID của mẫu ở mọi lần lặp; số version khớp giữa các nguồn (0/50 lệch) và giữa các lần lặp của cùng một nguồn; RAM đỉnh 2064 MB ≤ ngưỡng 6.000 MB. Lần chạy thật toàn dải: Kaggle tìm đủ 5.000/5.000 ID, API trả đủ 5.000/5.000 bài, 0 ID lệch ngày v1, 33 ID lệch `n_versions` (API làm chuẩn), RAM đỉnh 3.264 MB ≤ 6.000 MB.
- **Cách đo RAM:** không lấy mẫu định kỳ. Cuối mỗi worker, script đọc RAM đỉnh của tiến trình: trên Windows là `psutil.Process().memory_info().peak_wset`, trên Linux/macOS là `resource.getrusage(...).ru_maxrss`. "RAM nền" là RSS (`psutil ... .rss`) đọc **sau khi import thư viện của phương pháp** và trước khi làm việc. Con số "2061,2 ± 4,0 MB" là mean ± std của RAM đỉnh trên **3 lần lặp** (đỉnh cao nhất 2064,1 MB), không phải trung bình các mẫu theo thời gian.

### 7.4 Thư viện và phiên bản

Ghi nhận từ môi trường chạy phiên A:

| Thư viện | Phiên bản | Dùng cho |
|---|---|---|
| Python | 3.14.0 | |
| pandas | 3.0.2 | đọc/lọc snapshot Kaggle |
| sickle | 0.7.0 | OAI-PMH `GetRecord` |
| psutil | 7.2.2 | đo RAM |
| requests | 2.33.1 | gọi arXiv API |
| feedparser | 6.0.14 | parse feed Atom của API |
| `xml.etree.ElementTree` | thư viện chuẩn của Python | đếm `<version>` trong `arXivRaw` |
| kaggle (CLI) | 2.2.4 | phiên bản cài trong môi trường; `discovery.py` không dùng để tải |

### 7.5 Hạn chế đã biết của phép đo

- **API: đã đo thật 100 request nhưng chỉ một lần chạy.** Có 5/100 request chậm bất thường (7,7–43,4 s) và ≈ 230 s của bước API chưa giải thích được (mục 2.1). Cần chạy lặp ở các thời điểm khác nhau để có mean ± std; ước lượng 7–26 phút từ 3 mẫu request đã bị thay bằng số đo thật.
- **OAI chỉ đo 20 ID mỗi lần lặp** (60 request mỗi phiên), nên cận trên cực đoan có thể bị kéo bởi vài request chậm.
- **Chênh lệch giữa các lần đo (mục 2.4)**, gồm cả chênh giữa ước lượng 399 s và số đo thật 784,2 s, cho thấy thời điểm đo ảnh hưởng mạnh tới API và OAI; mỗi phiên chỉ đo ở một thời điểm trong ngày nên chưa biết giờ cao điểm của server arXiv ảnh hưởng thế nào.
- **Cache hệ điều hành:** lần đọc Kaggle đầu của phiên A chậm hơn khoảng 10–14 s so với hai lần sau. Số 66,76 s là trung bình có gồm lần đọc "nguội"; đọc "nóng" ≈ 61–64 s.
- **Tải file Kaggle chỉ đo một lần** (4.980 s, mạng chậm); con số này phụ thuộc mạng từng người.
- **Phiên bản và ngày của Kaggle snapshot chưa ghi** (mục 7.2) nên chưa kiểm được giả thuyết 33 ID lệch là do snapshot cũ hơn API; phiên B không ghi ngày chạy và thông tin môi trường.
- **RAM đỉnh khác nhau giữa hai cách chạy:** 2.061,2 MB (benchmark, 3 lần lặp) và 3.264,0 MB (chạy thật, 1 lần) với cùng `chunksize` 200.000; chưa rõ nguyên nhân. Chỉ nên coi RAM đỉnh Kaggle nằm trong khoảng 2,0–3,3 GB, dưới ngưỡng 6.000 MB, cho tới khi chạy lặp.