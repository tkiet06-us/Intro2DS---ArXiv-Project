# Khảo sát Discovery & Phương pháp Crawling (P1-00)

> **Dải ID:** `2506.00788` → `2506.05787` (5.000 bài)
> **Mẫu thử:** API và Kaggle dùng 50 ID liên tiếp `2506.00788` → `2506.00837` (Kaggle quét cả dải 5.000 ID); OAI-PMH dùng 20 ID đầu của mẫu đó · **Môi trường đo:** máy cá nhân, Windows 11, Python 3.14.0
> **Số liệu chính thức (phiên A):** `discovery.py`, 3 lần lặp, mỗi lần một tiến trình riêng, chạy lúc 13:53 UTC ngày 20/10/2026 (20:53 giờ Việt Nam). **Phiên B** là một lần chạy 3 lần lặp trước đó bằng bản script cũ (không ghi ngày), dùng để đánh giá độ nhạy theo thời điểm đo (mục 2.4). Các lần đo đơn lẻ ngày 14/10 và 16/10 chỉ đo một lần nên không dùng làm số chính thức.
>
> Ghi chú đọc số: thời gian phụ thuộc máy, mạng và **thời điểm đo**. Số "ước lượng 5.000 ID" được ngoại suy từ số đo mẫu (API theo số request, OAI theo số ID). Thời gian API và OAI **đã gồm** sleep 3 s giữa các request; thời gian Kaggle **không gồm** thời gian tải file.

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
  - Cùng một request, **phiên B đo 12,71 ± 10,67 s** (gấp khoảng 13 lần). Thời gian phản hồi của server phụ thuộc mạnh vào thời điểm đo (mục 2.4).
- **Ước lượng 5.000 ID:** rate limit tính theo **số request**, không theo số ID. Batch 50 ⇒ **100 request**, mỗi request = thời gian xử lý + 3 s sleep bắt buộc.
  - Phiên A: 100 × (0,99 + 3) ≈ **399 s ≈ 6,6 phút**.
  - Phiên B: 100 × (12,71 + 3) ≈ **1.571 s ≈ 26,2 phút**.
  - **Nên lập kế hoạch theo khoảng ≈ 7–26 phút.** Trong phiên A, khoảng tin cậy 95% của trung bình là 334–463 s và khoảng [min, max] là 371–422 s, nhưng chỉ có 3 mẫu request và các lần lặp cách nhau vài phút, nên khoảng này chỉ phản ánh dao động *trong phiên*, không phản ánh dao động *giữa các thời điểm*.

### 2.2 OAI-PMH (thư viện `sickle`)

- **Endpoint:** `https://oaipmh.arxiv.org/oai`. Endpoint cũ `export.arxiv.org/oai2` đã được thay (trang hướng dẫn của arXiv ghi rõ base URL đã đổi; theo tài liệu môn, từ 03/2025).
- **Cách hoạt động:** hai verb liên quan:
  - `GetRecord` (identifier `oai:arXiv.org:<id>`) – lấy **một** bài.
  - `ListRecords` – thu hoạch hàng loạt, lọc theo **ngày (`from`/`until`) và `set`**, phân trang bằng `resumptionToken`. `from`/`until` lọc theo *datestamp* (lần sửa metadata gần nhất), **không phải ngày nộp bài**; arXiv ghi rõ OAI-PMH không hỗ trợ thu hoạch chọn lọc theo ngày nộp.
- **Format `arXivRaw`:** chứa danh sách `<version version="vN">` với **ngày và kích thước từng version** ⇒ nguồn chính xác nhất cho "số version" và "ngày từng version".
- **Ưu:** đủ ngày + size mọi version trong một nguồn duy nhất; là chuẩn thu hoạch metadata.
- **Nhược:** `ListRecords` **không lọc được theo dải ID**, chỉ theo ngày/set; muốn lấy dải 788 → 5787 của 2025-06 phải quét cả tháng rồi bỏ phần thừa (đây chính là thu hoạch quy mô kho mà đề không yêu cầu). `GetRecord` thì 1 request/ID, nên 5.000 ID là 5.000 request, bị chặn dưới bởi sleep 3 s. arXiv dùng `503 Retry-After` để điều tiết.
- **Thời gian đo được (phiên A, endpoint mới, 3 lần lặp, mỗi lần 20 `GetRecord`):** **73,78 ± 6,72 s** (khoảng [66,18; 78,95] s), tìm thấy 20/20 ID, đã gồm 19 lần sleep 3 s (57 s). Thời gian request thực trung bình ≈ **0,84 s**, nên chi phí mỗi ID ≈ **3,84 s** (gồm sleep). RAM đỉnh 46,6 ± 0,2 MB (RAM nền 41,9 MB).
- **So với endpoint cũ:** lần đo trước (endpoint `export.arxiv.org/oai2`) là 14,37 s/ID (431,14 s cho 30 ID). Endpoint mới nhanh hơn khoảng **3,7 lần** ở phiên A (3,84 s/ID) hoặc **2,4 lần** ở phiên B (5,94 s/ID).
- **Ước lượng 5.000 ID:**
  - Phiên A: 5.000 × 3,84 ≈ **19.192 s ≈ 5,3 giờ**. Khoảng tin cậy 95% của trung bình: 18.423–19.962 s (5,1–5,5 giờ); khoảng [min, max] (cực đoan): 16.571–27.032 s (4,6–7,5 giờ).
  - Phiên B: **29.686 s ≈ 8,2 giờ**.
  - Cận dưới do sleep: 5.000 × 3 s ≈ 4,2 giờ. Với endpoint cũ là ≈ 20 giờ.

### 2.3 Kaggle dataset (`Cornell-University/arxiv`, đọc bằng pandas)

- **Cách hoạt động:** tải file `arxiv-metadata-oai-snapshot.json` (JSON Lines), đọc bằng `pd.read_json(lines=True, chunksize=...)`, lọc `id` bắt đầu bằng `2506.` và phần số trong `[788, 5787]`.
- **Dữ liệu có:** `id, title, authors, categories, doi, journal-ref, update_date` và `versions: [{version, created}]` ⇒ **có số version và ngày từng version** trong một lần đọc.
- **Ưu:** không tốn request nào tới arXiv, không dính giới hạn 3 s; lọc dải ID chỉ là thao tác pandas; một lần đọc là xong cả 5.000 bài.
- **Nhược:** là snapshot (cập nhật định kỳ), bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu; file lớn (1,73 GB nén, 5,61 GB sau khi giải nén), cần tài khoản/API key Kaggle; không có thông tin source `.tar.gz`.
- **Thời gian đo được:**
  - **Tải file:** **1 giờ 23 phút** (≈ 4.980 s), mạng chậm (~51 kB/s lúc đo). Chi phí **một lần**, đo một lần duy nhất, phụ thuộc mạng.
  - **Đọc + lọc dải 5.000 ID (phiên A, 3 lần lặp):** **66,76 ± 7,58 s** (khoảng [61,44; 75,44] s, median 63,40 s). Tìm thấy **5.000/5.000** ID trong dải ở cả 3 lần, và 50/50 ID của mẫu.
  - Lần lặp đầu (75,44 s) chậm hơn rõ so với hai lần sau (61,44 s và 63,40 s), nhiều khả năng do cache hệ điều hành chưa chứa file 5,61 GB. Đọc "nóng" ≈ 61–64 s; lần đọc đầu có thể chậm hơn khoảng 10 s. Phiên B (64,44 ± 0,72 s) không có hiện tượng này.
  - **Tổng end-to-end:** ≈ 5.047 s ≈ **1 giờ 24 phút** (gồm cả tải).
  - **RAM đỉnh** khi đọc (chunksize `200_000`): **2061,2 ± 4,0 MB** (mean ± std của 3 lần lặp; lần cao nhất 2064,1 MB). RAM nền sau khi import pandas: 78,2 MB.
- **Đối chiếu giữa các nguồn trên mẫu:** 50 ID được đối chiếu số version, **0 ID lệch**, và 0 ID đổi version giữa các lần lặp của cùng một nguồn (API và Kaggle phủ 50 ID; OAI phủ 20 ID đầu). Đây mới chỉ là mẫu 1% dải; phần còn lại sẽ được kiểm khi chạy API toàn dải (mục 4.1).

### 2.4 Độ nhạy theo thời điểm đo

Hai phiên đo cách nhau về thời điểm và cho kết quả rất khác nhau ở phần phụ thuộc server:

| Chỉ số | Phiên A (20/10, chính thức) | Phiên B (bản script cũ, không ghi ngày) |
|---|---|---|
| API, thời gian 1 request (50 ID) | 0,99 ± 0,26 s | 12,71 ± 10,67 s |
| OAI, thời gian 1 request | ≈ 0,84 s (3,84 s/ID gồm sleep) | ≈ 2,94 s (5,94 s/ID) |
| Kaggle, đọc + lọc cả dải | 66,76 ± 7,58 s | 64,44 ± 0,72 s |
| Kaggle, RAM đỉnh | 2061,2 ± 4,0 MB | 2048,7 ± 18,9 MB |
| Ước lượng API, 5.000 ID | 399 s (≈ 6,6 phút) | 1.571 s (≈ 26,2 phút) |
| Ước lượng OAI, 5.000 ID | 19.192 s (≈ 5,3 giờ) | 29.686 s (≈ 8,2 giờ) |
| Discovery kết hợp, chưa tính tải | ≈ 465 s (≈ 7,8 phút) | ≈ 1.635 s (≈ 27,3 phút) |
| Discovery kết hợp, tính cả tải | ≈ 5.445 s (≈ 1 giờ 31) | ≈ 6.615 s (≈ 1 giờ 50) |

Nhận xét:
- Phần chạy cục bộ (đọc Kaggle, RAM) ổn định giữa hai phiên; phần phụ thuộc server arXiv (API, OAI) dao động từ 3 đến 13 lần.
- Độ lệch chuẩn *trong* một phiên (và khoảng tin cậy tính từ nó) nhỏ hơn nhiều so với chênh lệch *giữa* hai phiên, nên khoảng tin cậy của một phiên đơn lẻ đánh giá thấp độ bất định thật.
- **Kết luận chọn phương án không đổi ở cả hai phiên** (OAI một mình đắt hơn phương án kết hợp 18–41 lần khi chưa tính tải).
- **Con số nên đưa vào báo cáo M1:** discovery ≈ **8–27 phút** (chưa tính tải), ≈ **1,5–1,8 giờ** (tính cả tải lần đầu); lấy phiên A làm giá trị chính, phiên B làm cận trên.

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
| RAM đỉnh khi đo | 46,5 MB | 46,6 MB | 2061,2 ± 4,0 MB |
| Thời gian chuẩn bị / tải | 0 | 0 | 4.980 s (1 giờ 23 phút), một lần |
| Ước lượng cho 5.000 ID (phiên A; phiên B) | ≈ 399 s (≈ 6,6 phút); phiên B ≈ 26 phút | ≈ 19.192 s (≈ 5,3 giờ); phiên B ≈ 8,2 giờ | ≈ 67 s (không tính tải); ≈ 5.047 s (tính cả tải) |

> Chú thích: cột Kaggle phải đọc kèm dòng "thời gian chuẩn bị / tải". Chi phí tải (1 giờ 23 phút) là một lần và có thể chia sẻ cho cả nhóm qua Drive, nhưng vẫn là chi phí thật của phương pháp này. Số mẫu của ba cột khác nhau (50 ID / 20 ID / cả dải) nên chỉ so sánh được qua dòng "ước lượng cho 5.000 ID".

## 4. Kết luận & lựa chọn

### 4.1 Phương án chọn

**Kết hợp hai nguồn trên toàn bộ 5.000 ID: arXiv API là nguồn chuẩn cho "bài tồn tại" và "số version"; Kaggle snapshot cung cấp ngày từng version và metadata; OAI-PMH `GetRecord` chỉ vá các ca lệch.** Khi hai nguồn mâu thuẫn, lấy API làm chuẩn.

Quy trình:
1. **Quét Kaggle (1 lần):** đọc snapshot bằng pandas theo chunk, lọc `2506.00788 → 2506.05787`. Thu được ID, số version và ngày từng version cho cả 5.000 bài (≈ 1 phút).
2. **Chạy arXiv API cho toàn bộ 5.000 ID** (100 request, batch 50, ≈ 7–26 phút tùy tốc độ server): xác nhận bài tồn tại và lấy `vN` mới nhất.
3. **So `n_versions` giữa API và Kaggle, lấy API làm chuẩn:**
   - Khớp ⇒ dùng ngày version từ Kaggle.
   - API có `vN` mới hơn Kaggle (snapshot cũ) hoặc bài chỉ có ở API ⇒ gọi OAI `GetRecord` (`arXivRaw`) để lấy ngày các version còn thiếu.
   - Lý do chạy toàn bộ thay vì chỉ lấy mẫu: nếu snapshot thiếu `v(N+1)` thì crawler sẽ bỏ sót version, vi phạm yêu cầu "mọi version". Chi phí chạy hết chỉ vài phút đến khoảng nửa tiếng nên không đáng để tiết kiệm.

**Tổng thời gian discovery (số liệu cho báo cáo M1), 5.000 ID:**

| Thành phần | Phiên A (chính thức) | Phiên B | Ghi chú |
|---|---|---|---|
| Quét Kaggle | 66,76 s | 64,44 s | đo trực tiếp, trung bình 3 lần |
| arXiv API toàn dải | ≈ 399 s (≈ 6,6 phút) | ≈ 1.571 s (≈ 26,2 phút) | ước lượng từ 3 mẫu request |
| **Cộng (chưa tính tải)** | **≈ 465 s ≈ 7,8 phút** | **≈ 1.635 s ≈ 27,3 phút** | khoảng dùng cho báo cáo: **8–27 phút** |
| Tải file Kaggle | 4.980 s (1 giờ 23 phút) | 4.980 s | một lần, chia sẻ cho nhóm |
| **Cộng (lần đầu, tính cả tải)** | **≈ 5.445 s ≈ 1 giờ 31 phút** | **≈ 6.615 s ≈ 1 giờ 50 phút** | khoảng dùng cho báo cáo: **1,5–1,8 giờ** |
| Vá bằng OAI | ≈ 3,84 s/bài | ≈ 5,94 s/bài | ví dụ 5% (250 bài) ≈ 16 phút / ≈ 25 phút |

Số bài lệch API–Kaggle: **0/50** trên mẫu; tỉ lệ trên toàn dải sẽ được ghi sau khi chạy API đầy đủ.

### 4.2 Ưu điểm của phương án này

| # | Ưu điểm | Giải thích |
|---|---|---|
| 1 | **Khớp bài toán "theo dải ID"** | Cả API và Kaggle đều lấy được đúng dải. OAI-PMH `ListRecords` chỉ lọc theo ngày/set. |
| 2 | **Đủ dữ liệu mà ít request** | Kaggle cho ngày từng version với 0 request. API chỉ cần khoảng 100 request cho 5.000 bài, thay vì 5.000 request của OAI `GetRecord`. |
| 3 | **Không bỏ sót version** | API là nguồn realtime làm chuẩn nên bắt được version mới hơn snapshot. Đây là điều kiện để đáp ứng "crawl mọi version". |
| 4 | **Tiết kiệm thời gian ở bước crawl thật** | Biết trước `n_versions` thì crawler tải đúng `v1..vN`, không phải dò `v(N+1)` để gặp 404. Mỗi lần dò tốn ≥ 3 s, nhân 5.000 bài ≈ 4 giờ lãng phí. |
| 5 | **Có kiểm chứng chéo** | Hai nguồn độc lập đối chiếu nhau trên toàn bộ dải, không chỉ trên mẫu. |
| 6 | **Nhanh hơn OAI-only** | ≈ 8–27 phút (chưa tính tải) hoặc ≈ 1,5–1,8 giờ (tính cả tải lần đầu), so với ≈ 5,3–8,2 giờ nếu chỉ dùng OAI `GetRecord` cho cả dải. |

### 4.3 Nhược điểm và cách giảm thiểu

| # | Nhược điểm | Cách giảm thiểu |
|---|---|---|
| 1 | **Tải file Kaggle rất lâu** (1 giờ 23 phút ở mạng ~51 kB/s), file 1,73 GB nén (5,61 GB giải nén), cần tài khoản/API key. | Chỉ một người tải một lần và chia sẻ qua Drive chung. Báo cáo ghi rõ chi phí này, không gộp vào thời gian đọc. Nếu chỉ tính riêng dải 5.000 ID của một người thì API một mình (≈ 7–26 phút) rẻ hơn tổng end-to-end Kaggle (≈ 84 phút); Kaggle có lợi nhờ ngày version và việc dùng lại cho nhiều thành viên. |
| 2 | **Snapshot có thể cũ**: bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu hoặc lệch. | Chạy API toàn dải, lấy API làm chuẩn, vá ngày version bằng OAI `GetRecord`. Trên mẫu 50 ID không có ca lệch. Ghi rõ phiên bản và ngày snapshot (mục 7.2). |
| 3 | **Vá bằng OAI tốn thời gian** (≈ 3,84–5,94 s/bài). | Chỉ áp dụng cho bài lệch; ước lượng chi phí theo tỉ lệ lệch ở mục 4.5. |
| 4 | **Thời gian phản hồi của API dao động mạnh theo thời điểm**: một request 50 ID mất 0,71–1,22 s ở phiên A nhưng 0,81–21,41 s ở phiên B. | Lập kế hoạch theo khoảng ≈ 7–26 phút; crawler có Checkpoint/Resume. |
| 5 | **Không có thông tin source** (`.tar.gz`/`.gz`). | Không thuộc phạm vi discovery. Source vẫn lấy từ `export.arxiv.org/src/{id}v{n}` ở bước crawl. |
| 6 | **Ngày ở dạng RFC 2822**, có thể lệch múi giờ giữa nguồn. | Chuẩn hóa sang ISO 8601 (UTC) bằng một hàm duy nhất, kiểm tra khớp trên mẫu. |
| 7 | **Phụ thuộc nguồn bên thứ ba** (Kaggle/Cornell). | Nếu Kaggle không dùng được, API + OAI `GetRecord` thay thế (chậm hơn, ≈ 5–8 giờ cho phần OAI). |

### 4.4 Vì sao không chọn một phương pháp đơn lẻ

- **Chỉ arXiv API:** nhanh (≈ 7–26 phút) và chuẩn về số version, nhưng thiếu ngày version trung gian. Đó là lý do cần nguồn thứ hai.
- **Chỉ OAI-PMH:** `arXivRaw` đầy đủ nhất và chỉ cần một nguồn (đơn giản hơn). Nhưng `ListRecords` không lọc được theo dải ID và buộc quét cả tháng (thu hoạch quy mô kho, ngoài phạm vi milestone); còn `GetRecord` đo được ≈ 3,84–5,94 s/ID, ngoại suy ≈ 5,3–8,2 giờ cho 5.000 ID (nhanh hơn 2,4–3,7 lần so với ≈ 20 giờ ở endpoint cũ, nhưng vẫn gấp khoảng 18–41 lần phương án kết hợp khi chưa tính tải, và khoảng 3,5–4,5 lần khi tính cả tải lần đầu). Phù hợp làm phương án dự phòng và để vá lẻ.
- **Chỉ Kaggle:** đủ dữ liệu nhưng không đảm bảo độ mới; rủi ro bỏ sót version.

### 4.5 Khi nào đổi phương án

Chi phí vá bằng OAI tăng tuyến tính theo số bài lệch (≈ 3,84 s/bài ở phiên A, ≈ 5,94 s/bài ở phiên B). Ví dụ 5% (250 bài) ≈ 16–25 phút, 10% (500 bài) ≈ 32–50 phút. **Trường hợp xấu nhất là mọi bài đều lệch:** tổng ≈ 399 + 67 + 19.192 ≈ 19.660 s ≈ 5,5 giờ ở phiên A (≈ 8,7 giờ ở phiên B), chỉ hơn OAI `GetRecord` một mình khoảng 8–27 phút (phần API + quét Kaggle).

Về thời gian, phương án kết hợp vẫn rẻ hơn OAI một mình cho tới khi tỉ lệ lệch đạt khoảng **94–98%** (chưa tính tải Kaggle) hoặc khoảng **72–78%** (tính cả tải lần đầu), tùy phiên đo. Vì vậy việc bỏ bước Kaggle khi tỉ lệ lệch cao (ví dụ trên 50%) là quyết định vì **đơn giản hóa quy trình**, không phải vì tiết kiệm thời gian. Các ngưỡng này ngoại suy từ số đo mẫu và cần đối chiếu với tỉ lệ lệch đo được trên toàn dải.

## 5. Ghi chú kỹ thuật phát hiện được

- **Số version của một bài:** từ API lấy hậu tố `vN` trong `<id>` của entry; từ Kaggle `len(versions)`; từ OAI đếm các phần tử `<version>`.
- **Ngày từng version:** API chỉ có `published` (v1) và `updated` (mới nhất); Kaggle `versions[i].created` và OAI `arXivRaw` `<version><date>` đều có đủ. Định dạng là RFC 2822 (`Mon, 2 Jun 2025 12:34:56 GMT`), cần đổi sang **ISO 8601** khi ghi `metadata.json`.
- **Rate limit:** ≥ 3 giây giữa hai request tới arXiv, 1 kết nối tại một thời điểm, dùng `export.arxiv.org` (không dùng `arxiv.org`) và **User-Agent mô tả rõ** kèm email liên hệ. Semantic Scholar: delay 1 s + API key.
- **Rate limit tính theo request, không theo ID:** batch 30 hay 50 ID thì số request mới quyết định thời gian, nên ước lượng tổng thời gian dựa trên số request (`ceil(5000/50) = 100`). Tuy nhiên thời gian xử lý mỗi request dao động mạnh theo thời điểm (0,71 s đến 21,41 s qua hai phiên đo).
- **Batch `id_list`:** dùng batch ~50 ID/request; luôn đặt `max_results` bằng kích thước batch, vì mặc định chỉ trả 10.
- **Endpoint OAI-PMH:** dùng `https://oaipmh.arxiv.org/oai`; endpoint cũ chậm hơn khoảng 2,4–3,7 lần trong các lần đo của nhóm (14,37 s/ID so với 5,94 và 3,84 s/ID).
- **ID không tồn tại / bị rút:** API bỏ qua entry, OAI trả `idDoesNotExist`; cần log và giữ record rỗng theo yêu cầu đề.
- **Định dạng ID:** từ 2015 là `yymm.NNNNN` (5 chữ số, padding 0, ví dụ `2506.00788`); ID cũ như `hep-th/9901001` chỉ xuất hiện ở phần reference.
- **Source:** nguồn LaTeX lấy qua `export.arxiv.org/src/{id}v{n}`, có thể là `.tar.gz` hoặc một `.gz` đơn chứa `.tex`; bài chỉ có PDF/Word không có source ⇒ giữ record rỗng.

## 6. Việc tiếp theo (phụ thuộc chuỗi P1-00 → P2-01 → P3-02a)

- [x] Điền RAM đỉnh khi đọc Kaggle (mục 2.3).
- [x] Chạy lại `discovery.py` bản mới (20/10/2026), có đủ môi trường, SHA-256 và kiểm tra tự động.
- [ ] Điền **phiên bản dataset** và **ngày cập nhật** của Kaggle snapshot (mục 7.2). Lần chạy 20/10 truyền nguyên văn `Version N` và `2026-MM-DD` nên kết quả chưa có thông tin này.
- [ ] Bàn giao interface discovery cho crawler engine (P2-01): đầu vào `--id_start --id_end`, đầu ra danh sách `{id, n_versions, version_dates}`.
- [ ] Khi chạy crawler thật: chạy API toàn dải, so `n_versions` với Kaggle và ghi tỉ lệ lệch vào báo cáo M1; ghi lại thời gian API thật để thay cho ước lượng 7–26 phút.

## 7. Thông tin để tái lập

> Các trường `⟨điền: …⟩` là thông tin chưa có trong kết quả chạy.

### 7.1 Môi trường đo

- Máy cá nhân, Windows 11 (`Windows-11-10.0.26200-SP0`), Python 3.14.0.
- Phiên A chạy lúc 2026-10-20 13:53:25 UTC. Commit git của repo lúc chạy: `88af6de` (lúc đó `discovery.py` chưa được commit, nên commit này không chứa script).
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
| SHA-256 của file sau khi giải nén | `9ee5aafa99209d6dbf4cd8c138dc662f290a7b49065d7301d4a0152bc0701c75` |

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
- **Thiết kế đo:** mỗi cặp (phương pháp, lần lặp) chạy trong **một tiến trình riêng** để RAM đỉnh không lẫn giữa các phương pháp. Lặp 3 lần, xen kẽ theo thứ tự api → oai → kaggle → api → … để giảm ảnh hưởng của mạng. Báo cáo mean, median, std, min, max.
- **Đo thời gian:** `time.perf_counter()` (thời gian thực, wall-clock) bao toàn bộ vòng lặp của worker. Với API và OAI, thời gian **đã gồm** sleep 3 s giữa các request; thời gian riêng từng request được ghi trong `req_seconds`. Với Kaggle, chỉ gồm đọc + lọc, **không gồm tải file** (nhập tay bằng `--download-seconds`).
- **Ước lượng 5.000 ID:** API = `ceil(5000/50)` request × (thời gian request + 3 s); OAI = 5.000 × (thời gian request + 3 s). Script xuất ba cột: giá trị trung bình, khoảng tin cậy 95% của trung bình (phân phối t, theo số mẫu request) và khoảng [min, max] (kịch bản cực đoan). Khi dưới 10 mẫu request, script in cảnh báo. Khoảng tin cậy chỉ phản ánh dao động trong một phiên đo (xem mục 2.4).
- **Kiểm tra tự động (phiên A đều đạt):** Kaggle tìm đủ 5.000/5.000 ID trong dải ở cả 3 lần; API và OAI tìm đủ ID của mẫu ở mọi lần lặp; số version khớp giữa các nguồn (0/50 lệch) và giữa các lần lặp của cùng một nguồn; RAM đỉnh 2064 MB ≤ ngưỡng 6.000 MB.
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

- **API chỉ có 3 mẫu request** (1 request × 3 lần lặp mỗi phiên). Ước lượng 7–26 phút có độ bất định lớn; chặt hơn bằng cách tăng mẫu (ví dụ `--n 150` cho 3 request mỗi lần lặp, kèm sleep giữa các request) hoặc lấy số đo thật khi chạy API toàn dải ở bước crawl.
- **OAI chỉ đo 20 ID mỗi lần lặp** (60 request mỗi phiên), nên cận trên cực đoan có thể bị kéo bởi vài request chậm.
- **Chênh lệch giữa hai phiên (mục 2.4)** cho thấy thời điểm đo ảnh hưởng mạnh tới API và OAI; mỗi phiên chỉ đo ở một thời điểm trong ngày nên chưa biết giờ cao điểm của server arXiv ảnh hưởng thế nào.
- **Cache hệ điều hành:** lần đọc Kaggle đầu của phiên A chậm hơn khoảng 10–14 s so với hai lần sau. Số 66,76 s là trung bình có gồm lần đọc "nguội"; đọc "nóng" ≈ 61–64 s.
- **Tải file Kaggle chỉ đo một lần** (4.980 s, mạng chậm); con số này phụ thuộc mạng từng người.
- **Phiên bản và ngày của Kaggle snapshot chưa ghi** (mục 7.2) nên chưa thể nói trước tỉ lệ lệch API–Kaggle trên toàn dải; phiên B không ghi ngày chạy và thông tin môi trường.
