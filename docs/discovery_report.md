# Khảo sát Discovery & Phương pháp Crawling (P1-00)

> **Dải ID:** 5 dải của 5 thành viên, tổng **25.000 bài** (file `ranges.txt`): `2410.13025` → `2410.18024`, `2411.03799` → `2411.08798`, `2502.18235` → `2503.01913` (qua 2 tháng), `2503.16914` → `2503.21913`, `2506.00788` → `2506.05787` (mỗi dải 5.000 bài). Giảng viên khuyến nghị gộp danh sách ID của cả nhóm và chạy chung ở cả 3 milestone, cuối M3 mới tách phần của từng người, nên discovery chạy cho cả 25.000 ID.
> **Mẫu thử (phiên A, benchmark, chỉ dải `2506.00788` → `2506.05787`):** API và Kaggle dùng 50 ID liên tiếp `2506.00788` → `2506.00837` (Kaggle quét cả dải 5.000 ID); OAI-PMH dùng 20 ID đầu của mẫu đó · **Môi trường đo:** máy cá nhân, Windows 11, Python 3.14.0
> **Số liệu chính thức:** (1) **Chạy thật toàn dải 25.000 ID** (`discovery.py --mode full --ranges ...`, 1 lần, 500 request API + quét Kaggle một lần cho cả 5 dải; chạy lúc **07:37:03 UTC ngày 25/10/2026**, tức 14:37 giờ Việt Nam; commit `5fa5015`) là nguồn của mọi con số "tổng thời gian discovery" và của số đo API toàn dải; (2) **phiên A** (`discovery.py` benchmark, 3 lần lặp, mỗi lần một tiến trình riêng, chạy lúc 13:53 UTC ngày 20/10/2026, tức 20:53 giờ Việt Nam, chỉ trên dải `2506`) là nguồn của số đo mẫu OAI-PMH, API và Kaggle. **Lần chạy toàn dải trước** (22/10/2026 04:53:25 UTC, chỉ dải `2506`, 5.000 ID, 100 request, commit `5c0bc13`) được giữ làm số đối chiếu độ nhạy theo thời điểm (mục 2.4), không phải số chính thức. **Phiên B** là một lần chạy trước đó bằng bản script cũ, không ghi ngày và không có file kết quả đi kèm nên **không kiểm chứng được**: chỉ giữ làm ghi chú ở mục 2.4, không dùng làm số chính thức. Các lần đo đơn lẻ ngày 14/10 và 16/10 chỉ đo một lần nên không dùng.
>
> Ghi chú đọc số: thời gian phụ thuộc máy, mạng và **thời điểm đo**. Số của API và của tổng discovery là **số đo thật** từ lần chạy toàn dải; số ước lượng của OAI (5.000 hoặc 25.000 ID) được ngoại suy từ số đo mẫu (theo số ID). Thời gian API và OAI **đã gồm** sleep 3 s giữa các request; thời gian Kaggle **không gồm** thời gian tải file.

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
- **Đo thật toàn dải 25.000 ID (`--mode full`, 1 lần, 25/10/2026 07:37 UTC):** rate limit tính theo **số request**, không theo số ID; batch 50 ⇒ **500 request**, mỗi request = thời gian xử lý + 3 s sleep bắt buộc.
  - **Tổng thời gian API: 2.525,95 s ≈ 42,1 phút** (wall-clock, gồm sleep, parse feed và ghi checkpoint sau mỗi batch). Nhận đủ 25.000/25.000 bài; 500/500 request thành công ngay lần thử đầu (0 lần thử lại).
  - **Thời gian mỗi request (50 ID, 500 request):** trung bình 1,29 s (KTC 95% ± 0,09 s), median 1,08 s, std 1,02 s, [0,82; 15,62] s. 496/500 request dưới 5 s; 4 request chậm (5,5; 10,4; 11,3 và 15,6 s) chiếm khoảng 43 s trong tổng 646,7 s thời gian request.
  - **Cấu thành:** 646,7 s (tổng thời gian request, theo log) + 1.497 s (499 lần sleep 3 s danh nghĩa) = 2.143,7 s; **còn ≈ 382 s (≈ 0,77 s mỗi vòng lặp) chưa giải thích được**. Phần này nằm ngoài thời gian request và sleep (parse feed, ghi + `fsync` checkpoint mỗi batch), chưa đo riêng từng phần. Báo cáo dùng số wall-clock 2.525,95 s vì đó là số đo thật.
  - **So với lần chạy toàn dải trước (22/10, 04:53 UTC, 5.000 ID):** 100 request, 784,2 s; request trung bình 2,57 s, median 1,35 s, [0,94; 43,40] s. Lần 25/10 nhanh hơn khoảng 2 lần ở trung bình, 1,25 lần ở median, và đuôi chậm ngắn hơn nhiều (15,6 s so với 43,4 s). Hai lần chạy ở hai thời điểm khác nhau nên đây là bằng chứng thêm cho độ nhạy theo thời điểm (mục 2.4).
  - **So với ước lượng từ mẫu:** phiên A ước lượng 500 × (0,99 + 3) ≈ 1.995 s, số đo thật 2.525,95 s **gấp khoảng 1,3 lần** (ở lần chạy 22/10 là khoảng 2 lần). Mẫu 3 request không bắt được đuôi chậm của phân bố.

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
- **Ngoại suy 25.000 ID:** 25.000 × 3,84 ≈ **96.000 s ≈ 26,7 giờ** (khoảng tin cậy 95% của trung bình: 92.115–99.810 s, tức 25,6–27,7 giờ; nhân 5 từ khoảng ở trên). Chỉ là ngoại suy từ phiên A, chưa chạy thật.

### 2.3 Kaggle dataset (`Cornell-University/arxiv`, đọc bằng pandas)

- **Cách hoạt động:** tải file `arxiv-metadata-oai-snapshot.json` (JSON Lines), đọc bằng `pd.read_json(lines=True, chunksize=...)`, lọc các `id` nằm trong một trong các dải đã cho bằng so sánh chuỗi ID `start <= id <= end` (ID cùng độ rộng `YYMM.NNNNN` nên so sánh chuỗi đúng thứ tự).
- **Dữ liệu có:** `id, title, authors, categories, doi, journal-ref, update_date` và `versions: [{version, created}]` ⇒ **có số version và ngày từng version** trong một lần đọc.
- **Ưu:** không tốn request nào tới arXiv, không dính giới hạn 3 s; lọc dải ID chỉ là thao tác pandas; một lần đọc là xong cả 5.000 bài.
- **Nhược:** là snapshot (cập nhật định kỳ), bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu; file lớn (1,73 GB nén, 5,61 GB sau khi giải nén), cần tài khoản/API key Kaggle; không có thông tin source `.tar.gz`.
- **Thời gian đo được:**
  - **Tải file:** **1 giờ 23 phút** (≈ 4.980 s) cho file 1,73 GB nén, tức tốc độ trung bình ≈ **350 kB/s** (1,73 GB / 4.980 s). Chi phí **một lần**, đo một lần duy nhất, phụ thuộc mạng.
  - **Đọc + lọc dải 5.000 ID (phiên A, 3 lần lặp):** **66,76 ± 7,58 s** (khoảng [61,44; 75,44] s, median 63,40 s). Tìm thấy **5.000/5.000** ID trong dải ở cả 3 lần, và 50/50 ID của mẫu.
  - Lần lặp đầu (75,44 s) chậm hơn rõ so với hai lần sau (61,44 s và 63,40 s), nhiều khả năng do cache hệ điều hành chưa chứa file 5,61 GB. Đọc "nóng" ≈ 61–64 s; lần đọc đầu có thể chậm hơn khoảng 10 s. Phiên B (64,44 ± 0,72 s) không có hiện tượng này.
  - **Tổng end-to-end:** ≈ 5.047 s ≈ **1 giờ 24 phút** (gồm cả tải).
  - **RAM đỉnh** khi đọc (chunksize `200_000`): **2061,2 ± 4,0 MB** (mean ± std của 3 lần lặp; lần cao nhất 2064,1 MB). RAM nền sau khi import pandas: 78,2 MB.
  - **Chạy thật toàn dải (`--mode full`, 25/10, 1 lần, cả 5 dải trong một lần quét):** quét + lọc **77,9 s** (≈ 72,0 MB/s), tìm thấy **25.000/25.000** ID trong 5 dải; RAM đỉnh **3.264,2 MB** (RAM nền 80,2 MB, chunksize `200_000`). Lần chạy 22/10 (1 dải, 5.000 ID) là 73,3 s và 3.264,0 MB, nên số bài lọc tăng 5 lần gần như không đổi RAM đỉnh. RAM đỉnh vẫn cao hơn khoảng 1,2 GB so với 2.061,2 MB của benchmark phiên A và **chưa giải thích được** chênh lệch này; chế độ full gọi `read_json(..., convert_dates=False)` còn benchmark thì không. Cả hai đều dưới ngưỡng 6.000 MB.
- **Đối chiếu giữa các nguồn trên mẫu:** 50 ID được đối chiếu số version, **0 ID lệch**, và 0 ID đổi version giữa các lần lặp của cùng một nguồn (API và Kaggle phủ 50 ID; OAI phủ 20 ID đầu). Đây chỉ là mẫu 1% của dải `2506`; kết quả đối chiếu toàn bộ 25.000 ID (98 ID lệch `n_versions`) nằm ở mục 4.1.

### 2.4 Độ nhạy theo thời điểm đo

Phần chạy cục bộ (đọc Kaggle) ổn định, còn phần phụ thuộc server arXiv (API, OAI) dao động mạnh theo thời điểm. **Phiên B chỉ là ghi chú** ("lần đo trước chậm hơn nhiều"): chạy bằng script cũ, không ghi ngày, không có file kết quả đi kèm nên không kiểm chứng được, và **không** dùng làm cận trên chính thức.

| Chỉ số | Phiên A (20/10, benchmark) | Phiên B (script cũ, không ghi ngày) | Chạy thật 22/10 (5.000 ID) | Chạy thật 25/10 (25.000 ID, **chính thức**) |
|---|---|---|---|---|
| API, thời gian 1 request (50 ID) | 0,99 ± 0,26 s (3 request) | 12,71 ± 10,67 s | TB 2,57 s, median 1,35 s, [0,94; 43,40] s (100 request) | TB 1,29 s, median 1,08 s, [0,82; 15,62] s (500 request) |
| OAI, thời gian 1 request | ≈ 0,84 s (3,84 s/ID gồm sleep) | ≈ 2,94 s (5,94 s/ID) | không chạy | không chạy |
| Kaggle, đọc + lọc | 66,76 ± 7,58 s (dải 5.000 ID) | 64,44 ± 0,72 s | 73,3 s | 77,9 s (5 dải, 25.000 ID) |
| Kaggle, RAM đỉnh | 2061,2 ± 4,0 MB | 2048,7 ± 18,9 MB | 3264,0 MB | 3264,2 MB |
| API, toàn dải | 399 s ≈ 6,6 phút (ước lượng, 5.000 ID) | 1.571 s ≈ 26,2 phút (ước lượng, 5.000 ID) | **784,2 s ≈ 13,1 phút (đo thật)** | **2.525,95 s ≈ 42,1 phút (đo thật)** |
| OAI, toàn dải (ước lượng) | 19.192 s ≈ 5,3 giờ (5.000 ID) | 29.686 s ≈ 8,2 giờ (5.000 ID) | không chạy | ≈ 96.000 s ≈ 26,7 giờ (ngoại suy từ phiên A, 25.000 ID) |
| Discovery kết hợp, chưa tính tải | ≈ 465 s ≈ 7,8 phút (ước lượng, 5.000 ID) | ≈ 1.635 s ≈ 27,3 phút (ước lượng, 5.000 ID) | **857,5 s ≈ 14,3 phút (đo thật)** | **2.604,2 s ≈ 43,4 phút (đo thật)** |
| Discovery kết hợp, tính cả tải | ≈ 5.445 s ≈ 1 giờ 31 (ước lượng, 5.000 ID) | ≈ 6.615 s ≈ 1 giờ 50 (ước lượng, 5.000 ID) | **≈ 5.837,5 s ≈ 1 giờ 37 (đo thật)** | **≈ 7.584,2 s ≈ 2 giờ 06 (đo thật)** |

Nhận xét:
- Thời gian đọc Kaggle ổn định giữa các lần đo (61–78 s); RAM đỉnh thì khác giữa benchmark và chạy thật (mục 2.3).
- Ước lượng từ 3 request thấp hơn số đo thật: 399 s so với 784,2 s ở 5.000 ID (khoảng 2 lần) và 1.995 s so với 2.525,95 s ở 25.000 ID (khoảng 1,3 lần), vì mẫu nhỏ không bắt được các request chậm.
- Độ lệch chuẩn *trong* một phiên (và khoảng tin cậy tính từ nó) nhỏ hơn nhiều so với chênh lệch *giữa* các lần đo, nên khoảng tin cậy của một phiên đơn lẻ đánh giá thấp độ bất định thật.
- **Kết luận chọn phương án không đổi:** OAI một mình (≈ 96.000 s ≈ 26,7 giờ cho 25.000 ID, ngoại suy phiên A) đắt hơn phương án kết hợp đo thật khoảng **37 lần** khi chưa tính tải (2.604,2 s) và **12,7 lần** khi tính cả tải lần đầu (7.584,2 s).
- **Con số đưa vào báo cáo M1:** discovery cho 25.000 ID ≈ **43,4 phút** (2.604,2 s, chưa tính tải) và ≈ **2 giờ 06 phút** (≈ 7.584,2 s) nếu tính cả tải lần đầu. Đây là kết quả của một lần chạy; hai lần chạy toàn dải ở hai thời điểm (22/10 và 25/10) cho thời gian mỗi request chênh nhau khoảng 2 lần, nên cần chạy lặp thêm ở thời điểm khác để có mean ± std của tổng thời gian.

## 3. Bảng so sánh

| Tiêu chí | arXiv API (`id_list`) | OAI-PMH (`sickle`) | Kaggle snapshot |
|---|---|---|---|
| Truy vấn theo dải ID | Có, trực tiếp | Không (`ListRecords`); 1 ID/request (`GetRecord`) | Có, lọc bằng pandas |
| Số version | Có, từ hậu tố `vN` của `<id>` | Có, đếm `<version>` | Có, `len(versions)` |
| Ngày từng version | Không, chỉ v1 và mới nhất | Có (kèm size) | Có (`created`) |
| Phụ thuộc rate limit 3 s | Có | Có | Không |
| Độ mới dữ liệu | Realtime | Realtime | Theo snapshot |
| Số request cho 5.000 bài (25.000 bài) | ~100 (batch 50); 500 cho 25.000 bài | ~5.000 (`GetRecord`); 25.000 cho 25.000 bài | 0 |
| Thời gian đo (phiên A, mean ± std, 3 lần) | 0,99 ± 0,26 s (50 ID, 1 request) | 73,78 ± 6,72 s (20 ID, 20 request) | 66,76 ± 7,58 s (đọc + lọc cả dải 5.000 ID) |
| RAM đỉnh khi đo | 46,5 MB | 46,6 MB | 2061,2 ± 4,0 MB (benchmark); 3264,2 MB (chạy thật 25/10) |
| Thời gian chuẩn bị / tải | 0 | 0 | 4.980 s (1 giờ 23 phút), một lần |
| Cho 5.000 ID (đo thật 22/10 hoặc ước lượng) | **784,2 s ≈ 13,1 phút (đo thật, 100 request)**; ước lượng phiên A từ mẫu: 399 s | ≈ 19.192 s (≈ 5,3 giờ), ước lượng phiên A | **73,3 s (đo thật, chưa tính tải)**; ≈ 5.053 s (tính cả tải) |
| Cho 25.000 ID (đo thật hoặc ước lượng) | **2.525,95 s ≈ 42,1 phút (đo thật, 500 request)** | ≈ 96.000 s (≈ 26,7 giờ), ngoại suy từ phiên A | **77,9 s (đo thật, chưa tính tải)**; ≈ 5.058 s (tính cả tải) |

> Chú thích: cột Kaggle phải đọc kèm dòng "thời gian chuẩn bị / tải". Chi phí tải (1 giờ 23 phút) là một lần và có thể chia sẻ cho cả nhóm qua Drive, nhưng vẫn là chi phí thật của phương pháp này. Số mẫu của ba cột khác nhau (50 ID / 20 ID / cả dải) nên chỉ so sánh được qua dòng "Cho 5.000 ID" và "Cho 25.000 ID".

## 4. Kết luận & lựa chọn

### 4.1 Phương án chọn

**Kết hợp hai nguồn trên toàn bộ 25.000 ID (5 dải): arXiv API là nguồn chuẩn cho "bài tồn tại" và "số version"; Kaggle snapshot cung cấp ngày từng version và metadata; OAI-PMH `GetRecord` chỉ vá các ca mà ngày version thiếu không suy ra được.** Khi hai nguồn mâu thuẫn, lấy API làm chuẩn.

Quy trình:
1. **Quét Kaggle (1 lần):** đọc snapshot bằng pandas theo chunk, lọc cả 5 dải trong **một lần quét** (so sánh chuỗi ID với từng dải). Thu được ID, số version và ngày từng version cho cả 25.000 bài (đo thật 77,9 s).
2. **Chạy arXiv API cho toàn bộ 25.000 ID** (500 request, batch 50, đo thật 2.525,95 s ≈ 42,1 phút). Dải trong một tháng thì sinh đủ ID; dải qua 2 tháng (`2502.18235` → `2503.01913`) lấy ID từ snapshot Kaggle. Sau **mỗi batch** script ghi nối tiếp vào `discovery_25k.jsonl` (Checkpoint/Resume): nếu một request lỗi sau 4 lần thử thì chạy lại cùng lệnh, ID đã có được bỏ qua.
3. **So `n_versions` giữa API và Kaggle, lấy API làm chuẩn:**
   - Khớp ⇒ dùng ngày version từ Kaggle.
   - API có `vN` mới hơn Kaggle (snapshot cũ) hoặc bài chỉ có ở API ⇒ bổ sung ngày các version còn thiếu. Nếu Kaggle chỉ thiếu **đúng version cuối** thì ngày của nó chính là `latest_version_date` của API (trên 24.902 bài khớp, `latest_version_date` luôn trùng ngày version cuối ở Kaggle), nên không cần request nào. Chỉ khi Kaggle thiếu **từ 2 version trở lên** (ngày các version giữa không có ở nguồn nào khác) mới gọi OAI `GetRecord` (`arXivRaw`).
   - Lý do chạy toàn bộ thay vì chỉ lấy mẫu: nếu snapshot thiếu `v(N+1)` thì crawler sẽ bỏ sót version, vi phạm yêu cầu "mọi version". Chi phí chạy hết đo thật ≈ 42 phút cho cả 25.000 ID (chạy một lần cho cả nhóm) nên không đáng để tiết kiệm.

**Tổng thời gian discovery (số liệu cho báo cáo M1), 25.000 ID, đo thật một lần chạy `--mode full` (25/10/2026):**

| Thành phần | Thời gian đo thật | Ghi chú |
|---|---|---|
| Quét Kaggle | 77,9 s | một lần quét cho cả 5 dải, ≈ 72,0 MB/s |
| arXiv API toàn dải | 2.525,95 s (≈ 42,1 phút) | 500 request, 0 lần thử lại, gồm sleep 3 s; thời gian request trung bình 1,29 s (mục 2.1) |
| Ghi `discovery_25k.jsonl` | 0,34 s | gộp, sắp xếp, ghi file cuối (checkpoint đã ghi dần sau mỗi batch, tính trong thời gian API) |
| **Cộng (chưa tính tải)** | **2.604,2 s ≈ 43,4 phút** | wall-clock end-to-end của một tiến trình |
| Tải file Kaggle | 4.980 s (1 giờ 23 phút) | một lần, chia sẻ cho nhóm |
| **Cộng (lần đầu, tính cả tải)** | **≈ 7.584,2 s ≈ 2 giờ 06 phút** | |
| Vá bằng OAI | ≈ 3,84 s/bài (phiên A) | chỉ 4 bài thiếu từ 2 version ≈ 15 s; nếu vá cả 98 bài lệch bằng OAI ≈ 376 s ≈ 6,3 phút |

Lần chạy toàn dải trước (22/10, 5.000 ID, một dải) cho 857,5 s ≈ 14,3 phút (API 784,2 s, quét Kaggle 73,3 s). Số đo trên thay cho khoảng ước lượng "8–27 phút" ở bản đầu (cận trên 27 phút dựa trên phiên B không kiểm chứng được nên đã bỏ). Đây là một lần chạy; thời gian API phụ thuộc thời điểm (mục 2.4) nên cần chạy lặp để có mean ± std.

**Số bài lệch API–Kaggle trên toàn bộ 25.000 ID: 98 (0,39%)** lệch `n_versions` (lấy API làm chuẩn); lệch ngày v1 (`published` của API so với `v1` của Kaggle): **0 ID**. Phân bố theo tiền tố ID: `2506` 34 bài, `2503` 30, `2411` 12, `2410` 11, `2502` 11. Mười ID lệch đầu danh sách: `2410.13168`, `2410.13278`, `2410.13366`, `2410.13735`, `2410.13925`, `2410.14056`, `2410.14904`, `2410.15740`, `2410.16457`, `2410.17192`; danh sách đầy đủ nằm trong `discovery_full_result_25k.json` (`check.n_versions_mismatch_api_vs_kaggle`). Kết quả nhất quán với 0/50 ở mẫu phiên A vì ID lệch nhỏ nhất của dải `2506` là `2506.00948`, nằm ngoài mẫu 50 ID đầu (`2506.00788` → `2506.00837`). Lần chạy 22/10 có 33 bài lệch trong dải `2506`; lần 25/10 có 34 bài, gồm 10 ID đầu danh sách cũ.

**Hướng lệch: 98/98 bài có Kaggle ít version hơn API** (94 bài ít hơn 1 version, 4 bài ít hơn 2 version), phù hợp với snapshot cũ hơn API. Ở cả 98 bài, các version Kaggle có là `v1…vk` liên tục và ngày `latest_version_date` của API không có trong Kaggle. Hệ quả cho việc vá:
- **94 bài thiếu đúng 1 version (version cuối):** ngày của version đó là `latest_version_date` của API, điền trực tiếp, **không cần request**.
- **4 bài thiếu 2 version** (`2502.20105`, `2503.18397`, `2503.21850`, `2506.02914`): thiếu cả version giữa, cần OAI `GetRecord`, 4 × 3,84 s ≈ 15 s.
- Hiện `version_dates` trong `discovery_25k.jsonl` chưa được vá (mục 6).

### 4.2 Ưu điểm của phương án này

| # | Ưu điểm | Giải thích |
|---|---|---|
| 1 | **Khớp bài toán "theo dải ID"** | Cả API và Kaggle đều lấy được đúng dải. OAI-PMH `ListRecords` chỉ lọc theo ngày/set. |
| 2 | **Đủ dữ liệu mà ít request** | Kaggle cho ngày từng version với 0 request. API chỉ cần 500 request cho 25.000 bài, thay vì 25.000 request của OAI `GetRecord`. |
| 3 | **Không bỏ sót version** | API là nguồn realtime làm chuẩn nên bắt được version mới hơn snapshot. Đây là điều kiện để đáp ứng "crawl mọi version". |
| 4 | **Tiết kiệm thời gian ở bước crawl thật** | Biết trước `n_versions` thì crawler tải đúng `v1..vN`, không phải dò `v(N+1)` để gặp 404. Mỗi lần dò tốn ≥ 3 s, nhân 25.000 bài ≈ 21 giờ lãng phí. |
| 5 | **Có kiểm chứng chéo** | Hai nguồn độc lập đối chiếu nhau trên toàn bộ dải, không chỉ trên mẫu. |
| 6 | **Nhanh hơn OAI-only** | ≈ 43,4 phút (chưa tính tải) hoặc ≈ 2 giờ 06 phút (tính cả tải lần đầu), so với ≈ 26,7 giờ (ngoại suy từ phiên A) nếu chỉ dùng OAI `GetRecord` cho 25.000 ID. |

### 4.3 Nhược điểm và cách giảm thiểu

| # | Nhược điểm | Cách giảm thiểu |
|---|---|---|
| 1 | **Tải file Kaggle rất lâu** (1 giờ 23 phút, tốc độ trung bình ≈ 350 kB/s), file 1,73 GB nén (5,61 GB giải nén), cần tài khoản/API key. | Chỉ một người tải một lần và chia sẻ qua Drive chung. Báo cáo ghi rõ chi phí này, không gộp vào thời gian đọc. Nếu chỉ tính một dải 5.000 ID của một người thì API một mình (≈ 13 phút, đo thật 22/10) rẻ hơn tổng end-to-end Kaggle (≈ 84 phút: 73,3 s quét + 4.980 s tải). Với cả nhóm (25.000 ID) API một mình vẫn rẻ hơn (≈ 42,1 phút so với ≈ 84 phút), và hai đường chỉ ngang nhau ở khoảng 50.000 ID (ngoại suy từ ≈ 0,10 s/ID của API). Kaggle có lợi nhờ ngày version và việc dùng lại cho nhiều thành viên. |
| 2 | **Snapshot có thể cũ**: bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu hoặc lệch. | Chạy API toàn dải, lấy API làm chuẩn, vá ngày version (điền từ `latest_version_date` khi thiếu 1 version, OAI `GetRecord` khi thiếu từ 2 version). Trên toàn bộ 25.000 ID đo được 98 bài (0,39%) lệch `n_versions` (mục 4.1). Ghi rõ phiên bản và ngày snapshot (mục 7.2). |
| 3 | **Vá bằng OAI tốn thời gian** (≈ 3,84 s/bài, phiên A). | Chỉ áp dụng cho bài thiếu từ 2 version trở lên (4/25.000 bài ≈ 15 s); bài thiếu 1 version điền từ `latest_version_date`; ước lượng chi phí theo tỉ lệ lệch ở mục 4.5. |
| 4 | **Thời gian phản hồi của API dao động mạnh theo thời điểm**: lần chạy 22/10 một request 50 ID mất từ 0,94 s đến 43,4 s (median 1,35 s; 5/100 request trên 5 s; có 1 lần `ConnectionResetError` được thử lại thành công); lần chạy 25/10 từ 0,82 s đến 15,6 s (median 1,08 s; 4/500 request trên 5 s; 0 lần thử lại). | Lập kế hoạch theo số đo thật ≈ 42 phút cho API (25.000 ID) và dự phòng gấp đôi; `discovery.py` tự thử lại tối đa 4 lần với backoff và có Checkpoint/Resume (ghi nối tiếp sau mỗi batch, chạy lại cùng lệnh để tiếp tục). |
| 5 | **Không có thông tin source** (`.tar.gz`/`.gz`). | Không thuộc phạm vi discovery. Source vẫn lấy từ `export.arxiv.org/src/{id}v{n}` ở bước crawl. |
| 6 | **Ngày ở dạng RFC 2822**, có thể lệch múi giờ giữa nguồn. | Chuẩn hóa sang ISO 8601 (UTC) bằng một hàm duy nhất, kiểm tra khớp trên mẫu. |
| 7 | **Phụ thuộc nguồn bên thứ ba** (Kaggle/Cornell). | Nếu Kaggle không dùng được, API + OAI `GetRecord` thay thế (chậm hơn, ≈ 5,3 giờ cho phần OAI). |

### 4.4 Vì sao không chọn một phương pháp đơn lẻ

- **Chỉ arXiv API:** nhanh (≈ 42 phút cho 25.000 ID, đo thật) và chuẩn về số version, nhưng thiếu ngày version trung gian. Đó là lý do cần nguồn thứ hai.
- **Chỉ OAI-PMH:** `arXivRaw` đầy đủ nhất và chỉ cần một nguồn (đơn giản hơn). Nhưng `ListRecords` không lọc được theo dải ID và buộc quét cả tháng (thu hoạch quy mô kho, ngoài phạm vi milestone); còn `GetRecord` đo được ≈ 3,84 s/ID (phiên A), ngoại suy ≈ 5,3 giờ cho 5.000 ID và ≈ 26,7 giờ cho 25.000 ID (nhanh hơn khoảng 3,7 lần so với endpoint cũ, ≈ 20 giờ cho 5.000 ID, nhưng vẫn gấp khoảng 37 lần phương án kết hợp đo thật khi chưa tính tải, và khoảng 12,7 lần khi tính cả tải lần đầu). Phù hợp làm phương án dự phòng và để vá lẻ.
- **Chỉ Kaggle:** đủ dữ liệu nhưng không đảm bảo độ mới; rủi ro bỏ sót version.

### 4.5 Khi nào đổi phương án

Chi phí vá bằng OAI tăng tuyến tính theo số bài cần vá (≈ 3,84 s/bài, phiên A). Ví dụ 5% (1.250 bài trong 25.000) ≈ 80 phút, 10% (2.500 bài) ≈ 160 phút; với 98 bài lệch đo được (0,39%) mà chỉ 4 bài thật sự cần OAI thì ≈ 15 s (nếu vá cả 98 bài bằng OAI: ≈ 376 s ≈ 6,3 phút). **Trường hợp xấu nhất là mọi bài đều lệch:** tổng ≈ 2.604,2 + 96.000 ≈ 98.600 s ≈ 27,4 giờ, chỉ hơn OAI `GetRecord` một mình khoảng 43 phút (phần API + quét Kaggle).

Về thời gian, phương án kết hợp vẫn rẻ hơn OAI một mình cho tới khi tỉ lệ lệch đạt khoảng **97%** (chưa tính tải Kaggle) hoặc khoảng **92%** (tính cả tải lần đầu); tính từ số đo thật và số đo OAI phiên A. Với tỉ lệ lệch đo được 0,39%, ngưỡng này còn rất xa. Vì vậy việc bỏ bước Kaggle khi tỉ lệ lệch cao (ví dụ trên 50%) là quyết định vì **đơn giản hóa quy trình**, không phải vì tiết kiệm thời gian. Ngưỡng ngoại suy từ số đo OAI mẫu, nên mang độ bất định của mẫu 20 ID.

## 5. Ghi chú kỹ thuật phát hiện được

- **Số version của một bài:** từ API lấy hậu tố `vN` trong `<id>` của entry; từ Kaggle `len(versions)`; từ OAI đếm các phần tử `<version>`.
- **Ngày từng version:** API chỉ có `published` (v1) và `updated` (mới nhất); Kaggle `versions[i].created` và OAI `arXivRaw` `<version><date>` đều có đủ. Định dạng là RFC 2822 (`Mon, 2 Jun 2025 12:34:56 GMT`), cần đổi sang **ISO 8601** khi ghi `metadata.json`.
- **Rate limit:** ≥ 3 giây giữa hai request tới arXiv, 1 kết nối tại một thời điểm, dùng `export.arxiv.org` (không dùng `arxiv.org`) và **User-Agent mô tả rõ** kèm email liên hệ. Semantic Scholar: delay 1 s + API key.
- **Rate limit tính theo request, không theo ID:** batch 30 hay 50 ID thì số request mới quyết định thời gian, nên ước lượng tổng thời gian dựa trên số request (`ceil(25000/50) = 500`). Tuy nhiên thời gian xử lý mỗi request dao động mạnh theo thời điểm (0,82 s đến 43,4 s qua hai lần chạy thật, 100 và 500 request).
- **Batch `id_list`:** dùng batch ~50 ID/request; luôn đặt `max_results` bằng kích thước batch, vì mặc định chỉ trả 10.
- **Endpoint OAI-PMH:** dùng `https://oaipmh.arxiv.org/oai`; endpoint cũ chậm hơn khoảng 3,7 lần trong các lần đo của nhóm (14,37 s/ID so với 3,84 s/ID ở phiên A).
- **ID không tồn tại / bị rút:** API bỏ qua entry, OAI trả `idDoesNotExist`; cần log và giữ record rỗng theo yêu cầu đề.
- **Định dạng ID:** từ 2015 là `yymm.NNNNN` (5 chữ số, padding 0, ví dụ `2506.00788`); ID cũ như `hep-th/9901001` chỉ xuất hiện ở phần reference.
- **Source:** nguồn LaTeX lấy qua `export.arxiv.org/src/{id}v{n}`, có thể là `.tar.gz` hoặc một `.gz` đơn chứa `.tex`; bài chỉ có PDF/Word không có source ⇒ giữ record rỗng.
- **Nhiều dải và checkpoint:** `discovery.py --ranges` nhận file nhiều dải (mỗi dòng một dải, không chồng nhau); Kaggle lọc tất cả dải trong một lần quét. Dải qua nhiều tháng lấy ID từ snapshot. Sau mỗi batch API ghi nối tiếp vào `--out`; chạy lại cùng lệnh bỏ qua ID đã có (`--fresh` để chạy lại từ đầu). `authors` ghi dạng danh sách chuỗi (tên có thể chứa dấu phẩy), `venue` lấy từ `arxiv_journal_ref` (có ở 4.119/25.000 bài, còn lại `null`).

## 6. Việc tiếp theo (phụ thuộc chuỗi P1-00 → P2-01 → P3-02a)

- [x] Điền RAM đỉnh khi đọc Kaggle (mục 2.3).
- [x] Chạy lại `discovery.py` bản mới (20/10/2026), có đủ môi trường, SHA-256 và kiểm tra tự động.
- [x] Chạy `discovery.py --mode full` toàn bộ 25.000 ID (5 dải) ngày 25/10/2026: 25.000/25.000 bài, 98 bài lệch `n_versions` (mục 4.1); số đo thời gian thật thay cho ước lượng. Lần chạy trước (22/10, 5.000 ID) cho 33 bài lệch trong dải `2506`.
- [x] Bàn giao interface discovery cho crawler engine (P2-01): đầu vào `--ranges ranges.txt` (hoặc `--id_start --id_end`; ID đầy đủ, dải có thể qua nhiều tháng), đầu ra `data/raw/discovery_25k.jsonl`, mỗi bài một dòng `{arxiv_id, n_versions, v1_date, latest_version_date, version_dates, title, authors (danh sách), venue}`.
- [x] Sửa theo review: `authors` là danh sách chuỗi, thêm `venue`, nhận nhiều dải ID, Checkpoint/Resume.
- [x] Điền **phiên bản dataset** và **ngày cập nhật** của Kaggle snapshot (mục 7.2). Lần chạy 20/10 truyền nguyên văn `Version N` và `2026-MM-DD` nên kết quả chưa có thông tin này.
- [x] Xác định hướng lệch của 98 ID: 98/98 bài có Kaggle ít version hơn API (94 bài thiếu 1 version, 4 bài thiếu 2 version), phù hợp với snapshot cũ hơn API (mục 4.1).
- [x] Vá `version_dates` trong `discovery_25k.jsonl`: 94 bài thiếu 1 version điền từ `latest_version_date` (không cần request); 4 bài (`2502.20105`, `2503.18397`, `2503.21850`, `2506.02914`) gọi OAI `GetRecord` (≈ 15 s).
- [ ] Chạy lặp `--mode full` thêm 1–2 lần ở thời điểm khác để có mean ± std của tổng thời gian (đặt `--out` và `--report` khác nhau mỗi lần để không ghi đè). Hai lần chạy toàn dải đã có (22/10 và 25/10, khác số ID) cho thấy thời gian mỗi request chênh khoảng 2 lần. Làm rõ hai điểm chưa giải thích: RAM đỉnh 3.264 MB so với 2.061 MB của benchmark, và ≈ 382 s chưa rõ nguồn ở bước API (mục 2.1).


## 7. Thông tin để tái lập

### 7.1 Môi trường đo

- Máy cá nhân, Windows 11 (`Windows-11-10.0.26200-SP0`), Python 3.14.0.
- Phiên A chạy lúc 2026-10-20 13:53:25 UTC. Commit git của repo lúc chạy: `88af6de` (lúc đó `discovery.py` chưa được commit, nên commit này không chứa script).
- Chạy thật toàn dải 25.000 ID (`--mode full`, nguồn của số chính thức): chạy lúc **2026-10-25 07:37:03 UTC**, commit git `5fa5015`, Python 3.14.0 và thư viện cùng phiên bản với phiên A (mục 7.4); cùng máy cá nhân Windows. Lần chạy toàn dải trước (5.000 ID, dải `2506`): 2026-10-22 04:53:25 UTC, commit `5c0bc13`.
- Mẫu (phiên A, benchmark, dải `2506`): 50 ID liên tiếp `2506.00788` → `2506.00837` cho API và Kaggle; 20 ID đầu của mẫu đó cho OAI (`--oai-n 20`). Kaggle quét cả dải `788 → 5787`.
- Tham số cố định: API batch 50 ID/request với `max_results=len(batch)`, sleep 3 s giữa hai request; OAI `GetRecord` với `metadataPrefix=arXivRaw`, sleep 3 s giữa hai ID, `Sickle(max_retries=3)`; Kaggle `pd.read_json(lines=True, chunksize=200_000, dtype=False)`. Mọi request gửi `User-Agent` mô tả dự án kèm email liên hệ.

### 7.2 Kaggle snapshot

Kết quả lệch API–Kaggle phụ thuộc trực tiếp vào độ cũ của snapshot: bài có version mới hơn thời điểm snapshot sẽ bị Kaggle thiếu `v(N+1)`.

| Mục | Giá trị |
| :--- | :--- |
| Dataset | Cornell-University/arxiv |
| File | `arxiv-metadata-oai-snapshot.json` (1,73 GB nén; 5,65 GB sau khi giải nén), đặt ở `data/raw/` |
| Phiên bản dataset trên Kaggle | **Version 307** |
| Ngày cập nhật của phiên bản đó | **11/10/2026** |
| Ngày tải về | **14/10/2026** (thời điểm sửa file trên máy: 14/10/2026 16:55:21 UTC) |
| SHA-256 của file sau khi giải nén | 9ee5aafa99209d6dbf4cd8c138dc662f290a7b49065d7301d4a0152bc0701c75 (tính ở phiên A với `--hash-kaggle`). Lần chạy toàn dải 25/10 không dùng `--hash-kaggle` (`environment.kaggle_file.sha256 = null`), nhưng file cùng tên, cùng thời điểm sửa (14/10/2026 16:55:21 UTC) và kích thước tương đương 5.65 GB; có thể tính lại SHA-256 để xác nhận. |

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
py src\m1_crawling\discovery.py --mode full --ranges data\raw\ranges.txt --total-ids 25000 ^
    --contact "kton61174@gmail.com" --kaggle-json data\raw\arxiv-metadata-oai-snapshot.json ^
    --out data\raw\discovery_25k.jsonl --report discovery_full_result_25k.json
```

  `ranges.txt` có mỗi dòng một dải `ID_dau ID_cuoi` (dòng trống và phần sau `#` bị bỏ qua; các dải không được chồng nhau): 5 dải của nhóm (đầu báo cáo). Có thể chạy một dải bằng `--id-start/--id-end` (ID đầy đủ `YYMM.NNNNN`; các cờ nhận cả dạng gạch nối và gạch dưới). Đầu ra: `data/raw/discovery_25k.jsonl` (interface cho P2-01; SHA-256 `4d8a684db4dac93e5564aceeb7e3e86a9053b38007230438d767f56d4326283e`) và `discovery_full_result_25k.json` (thời gian từng giai đoạn, log từng request, RAM nền/đỉnh, tốc độ, danh sách ID lệch/thiếu, môi trường; email được che). Quy trình: quét Kaggle một lần cho mọi dải → gọi API theo batch 50 (tối đa 4 lần thử mỗi request với backoff; ID thiếu được hỏi lại 1 lần), **ghi nối tiếp vào jsonl sau mỗi batch** (Checkpoint/Resume: lỗi giữa chừng thì chạy lại cùng lệnh, ID đã có bị bỏ qua; `--fresh` để chạy lại từ đầu; file cũ có `authors` dạng chuỗi hoặc thiếu `venue` bị từ chối) → đối chiếu `n_versions` và ngày v1 → ghi file cuối (sắp xếp, không trùng). Thời gian đo bằng `perf_counter()` cho cả tiến trình, **không gồm thời gian tải file Kaggle** (gộp được bằng `--download-seconds`). RAM đỉnh đọc ngay sau bước quét Kaggle (giai đoạn tốn RAM nhất) và cuối chương trình, bằng cùng cách đo ở dưới.
- **Thiết kế đo:** mỗi cặp (phương pháp, lần lặp) chạy trong **một tiến trình riêng** để RAM đỉnh không lẫn giữa các phương pháp. Lặp 3 lần, xen kẽ theo thứ tự api → oai → kaggle → api → … để giảm ảnh hưởng của mạng. Báo cáo mean, median, std, min, max.
- **Đo thời gian:** `time.perf_counter()` (thời gian thực, wall-clock) bao toàn bộ vòng lặp của worker. Với API và OAI, thời gian **đã gồm** sleep 3 s giữa các request; thời gian riêng từng request được ghi trong `req_seconds`. Với Kaggle, chỉ gồm đọc + lọc, **không gồm tải file** (nhập tay bằng `--download-seconds`).
- **Ước lượng 5.000 ID:** API = `ceil(5000/50)` request × (thời gian request + 3 s); OAI = 5.000 × (thời gian request + 3 s). Script xuất ba cột: giá trị trung bình, khoảng tin cậy 95% của trung bình (phân phối t, theo số mẫu request) và khoảng [min, max] (kịch bản cực đoan). Khi dưới 10 mẫu request, script in cảnh báo. Khoảng tin cậy chỉ phản ánh dao động trong một phiên đo (xem mục 2.4).
- **Kiểm tra tự động (phiên A đều đạt):** Kaggle tìm đủ 5.000/5.000 ID trong dải ở cả 3 lần; API và OAI tìm đủ ID của mẫu ở mọi lần lặp; số version khớp giữa các nguồn (0/50 lệch) và giữa các lần lặp của cùng một nguồn; RAM đỉnh 2064 MB ≤ ngưỡng 6.000 MB. Lần chạy thật toàn dải 25/10: Kaggle tìm đủ 25.000/25.000 ID, API trả đủ 25.000/25.000 bài, 0 ID lệch ngày v1, 98 ID lệch `n_versions` (API làm chuẩn), RAM đỉnh 3.264 MB ≤ 6.000 MB.
- **Cách đo RAM:** không lấy mẫu định kỳ. Cuối mỗi worker, script đọc RAM đỉnh của tiến trình: trên Windows là `psutil.Process().memory_info().peak_wset`, trên Linux/macOS là `resource.getrusage(...).ru_maxrss`. "RAM nền" là RSS (`psutil ... .rss`) đọc **sau khi import thư viện của phương pháp** và trước khi làm việc. Con số "2061,2 ± 4,0 MB" là mean ± std của RAM đỉnh trên **3 lần lặp** (đỉnh cao nhất 2064,1 MB), không phải trung bình các mẫu theo thời gian.

### 7.4 Thư viện và phiên bản

Ghi nhận từ môi trường chạy phiên A (lần chạy toàn dải 25/10 dùng cùng phiên bản):

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

- **API: đã đo thật hai lần chạy toàn dải (100 request ngày 22/10, 500 request ngày 25/10), mỗi lần ở một thời điểm.** Thời gian mỗi request chênh khoảng 2 lần giữa hai lần (TB 2,57 s so với 1,29 s); lần 22/10 có 5/100 request chậm bất thường (7,7–43,4 s), lần 25/10 có 4/500 (5,5–15,6 s). ≈ 382 s của bước API ngày 25/10 (≈ 230 s ở lần 22/10) chưa giải thích được (mục 2.1). Cần chạy lặp ở các thời điểm khác nhau để có mean ± std của tổng thời gian; ước lượng 7–26 phút từ 3 mẫu request đã bị thay bằng số đo thật.
- **OAI chỉ đo 20 ID mỗi lần lặp** (60 request mỗi phiên), nên cận trên cực đoan có thể bị kéo bởi vài request chậm.
- **Chênh lệch giữa các lần đo (mục 2.4)**, gồm cả chênh giữa ước lượng 399 s và số đo thật 784,2 s, cũng như giữa hai lần chạy thật 22/10 và 25/10, cho thấy thời điểm đo ảnh hưởng mạnh tới API và OAI; mỗi phiên chỉ đo ở một thời điểm trong ngày nên chưa biết giờ cao điểm của server arXiv ảnh hưởng thế nào.
- **Cache hệ điều hành:** lần đọc Kaggle đầu của phiên A chậm hơn khoảng 10–14 s so với hai lần sau. Số 66,76 s là trung bình có gồm lần đọc "nguội"; đọc "nóng" ≈ 61–64 s.
- **Tải file Kaggle chỉ đo một lần** (4.980 s, mạng chậm); con số này phụ thuộc mạng từng người.
- **Phiên bản và ngày của Kaggle snapshot chưa ghi** (mục 7.2) nên chưa kiểm trực tiếp được giả thuyết 98 ID lệch là do snapshot cũ hơn API (dữ liệu chỉ cho thấy Kaggle luôn ít version hơn API); phiên B không ghi ngày chạy và thông tin môi trường.
- **RAM đỉnh khác nhau giữa hai cách chạy:** 2.061,2 MB (benchmark, 3 lần lặp) và 3.264,0–3.264,2 MB (chạy thật, 2 lần) với cùng `chunksize` 200.000; chưa rõ nguyên nhân. Chỉ nên coi RAM đỉnh Kaggle nằm trong khoảng 2,0–3,3 GB, dưới ngưỡng 6.000 MB, cho tới khi chạy lặp.