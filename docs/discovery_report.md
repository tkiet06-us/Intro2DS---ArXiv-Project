# Khảo sát Discovery & Phương pháp Crawling (P1-00)

> **Dải ID:** `2506.00788` → `2506.05787` (5.000 bài)
> **Mẫu thử:** 50 ID liên tiếp `2506.00788` → `2506.00837` (Kaggle: quét cả dải 5.000 ID) · **Môi trường đo:** máy cá nhân (Windows), Python 3.14
> **Ngày đo:** 14/10/2026 (lần 1) và 16/10/2026 (lần 2, số liệu chính thức, dùng endpoint OAI mới)
>
> Ghi chú đọc số: thời gian phụ thuộc máy và mạng. Số "ước lượng 5.000 ID" được ngoại suy từ số đo mẫu (API theo số request, OAI theo số ID).

## 1. Mục tiêu

Chọn cách "discovery" (xác định bài nào tồn tại trong dải ID được giao, có bao nhiêu version, ngày của từng version, metadata cơ bản) cho Milestone 1. Hệ quả: đầu vào cho crawler (P2-01) và đợt crawl thật (22/10 → 28/10).

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
- **Nhược:** giới hạn ≥ 3 giây/request, 1 kết nối; ID không tồn tại thì bị bỏ qua trong feed, cần so khớp với danh sách đã hỏi; thiếu ngày các version trung gian; thời gian phản hồi dao động.
- **Thời gian đo được:**
  - Lần 1: **2,74 s** cho 30 ID (1 request).
  - Lần 2: **10,99 s** cho 50 ID (1 request, 50/50 bài tìm thấy).
  - Hai lần cách nhau gần 4 lần, nên thời gian phản hồi của server **không ổn định**; dùng số lần 2 làm số lập kế hoạch (thận trọng).
- **Ước lượng 5.000 ID:** rate limit tính theo **số request**, không theo số ID. Batch 50 ⇒ **100 request**, mỗi request ≈ 10,99 s (xử lý) + 3 s (sleep bắt buộc) ⇒ **100 × 13,99 ≈ 1.399 s ≈ 23 phút**. Theo số đo lần 1 thì khoảng 574 s (≈ 9,6 phút), nên khoảng ước lượng là **10–23 phút**.

### 2.2 OAI-PMH (thư viện `sickle`)

- **Endpoint:** `https://oaipmh.arxiv.org/oai`. Endpoint cũ `export.arxiv.org/oai2` đã được thay (trang hướng dẫn của arXiv ghi rõ base URL đã đổi; theo tài liệu môn, từ 03/2025).
- **Cách hoạt động:** hai verb liên quan:
  - `GetRecord` (identifier `oai:arXiv.org:<id>`) – lấy **một** bài.
  - `ListRecords` – thu hoạch hàng loạt, lọc theo **ngày (`from`/`until`) và `set`**, phân trang bằng `resumptionToken`. `from`/`until` lọc theo *datestamp* (lần sửa metadata gần nhất), **không phải ngày nộp bài**; arXiv ghi rõ OAI-PMH không hỗ trợ thu hoạch chọn lọc theo ngày nộp.
- **Format `arXivRaw`:** chứa danh sách `<version version="vN">` với **ngày và kích thước từng version** ⇒ nguồn chính xác nhất cho "số version" và "ngày từng version".
- **Ưu:** đủ ngày + size mọi version trong một nguồn duy nhất; là chuẩn thu hoạch metadata.
- **Nhược:** `ListRecords` **không lọc được theo dải ID**, chỉ theo ngày/set; muốn lấy dải 788 → 5787 của 2025-06 phải quét cả tháng rồi bỏ phần thừa. `GetRecord` thì 1 request/ID, nên 5.000 ID là 5.000 request, bị chặn dưới bởi sleep 3 s. arXiv dùng `503 Retry-After` để điều tiết.
- **Thời gian đo được (endpoint mới):** `GetRecord` 50 ID = **178,7 s**, tức **3,57 s/ID** (đã gồm sleep 3 s; thời gian request thực ≈ (178,72 − 49×3)/50 ≈ 0,63 s). 50/50 bài tìm thấy.
- **So với endpoint cũ:** lần đo trước (endpoint `export.arxiv.org/oai2`) là 14,37 s/ID (431,14 s cho 30 ID). Endpoint mới nhanh hơn khoảng 4 lần, hiện gần như chỉ còn bị giới hạn bởi sleep 3 s.
- **Ước lượng 5.000 ID:** 3,57 × 5.000 ≈ **17.872 s ≈ 5,0 giờ** (cận dưới do sleep: 5.000 × 3 s ≈ 4,2 giờ). Với endpoint cũ là ≈ 20 giờ.

### 2.3 Kaggle dataset (`Cornell-University/arxiv`, đọc bằng pandas)

- **Cách hoạt động:** tải file `arxiv-metadata-oai-snapshot.json` (JSON Lines), đọc bằng `pd.read_json(lines=True, chunksize=...)`, lọc `id` bắt đầu bằng `2506.` và phần số trong `[788, 5787]`.
- **Dữ liệu có:** `id, title, authors, categories, doi, journal-ref, update_date` và `versions: [{version, created}]` ⇒ **có số version và ngày từng version** trong một lần đọc.
- **Ưu:** không tốn request nào tới arXiv, không dính giới hạn 3 s; lọc dải ID chỉ là thao tác pandas; một lần đọc là xong cả 5.000 bài.
- **Nhược:** là snapshot (cập nhật định kỳ), bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu; file lớn (1,73 GB nén), cần tài khoản/API key Kaggle; không có thông tin source `.tar.gz`.
- **Thời gian đo được:**
  - **Tải file:** **1 giờ 23 phút** (≈ 4.980 s), mạng chậm (~51 kB/s lúc đo). Chi phí **một lần**, phụ thuộc mạng.
  - **Đọc + lọc dải 5.000 ID:** **66,38 s** (lần 1: 57,44 s). Tìm thấy **5.000/5.000** ID trong dải, và 50/50 ID của mẫu.
  - **Tổng end-to-end:** ≈ 5.046 s ≈ **1 giờ 24 phút** (gồm cả tải).
  - **RAM đỉnh** khi đọc (chunksize `200_000`): **2062.5 MB** (trung bình 2048.7 ± 18.9 MB).
- **Đối chiếu với API trên mẫu 50 ID:** số version **khớp hoàn toàn** (không có ID lệch). Đây mới chỉ là mẫu 1% dải; phần còn lại sẽ được kiểm khi chạy API toàn dải (mục 4.1).

## 3. Bảng so sánh

| Tiêu chí | arXiv API (`id_list`) | OAI-PMH (`sickle`) | Kaggle snapshot |
|---|---|---|---|
| Truy vấn theo dải ID | Có, trực tiếp | Không (`ListRecords`); 1 ID/request (`GetRecord`) | Có, lọc bằng pandas |
| Số version | Có, từ hậu tố `vN` của `<id>` | Có, đếm `<version>` | Có, `len(versions)` |
| Ngày từng version | Không, chỉ v1 và mới nhất | Có (kèm size) | Có (`created`) |
| Phụ thuộc rate limit 3 s | Có | Có | Không |
| Độ mới dữ liệu | Realtime | Realtime | Theo snapshot |
| Số request cho 5.000 bài | ~100 (batch 50) | ~5.000 (`GetRecord`) | 0 |
| Thời gian đo (mẫu) | 10,99 s (50 ID, 1 request) | 178,7 s (50 ID, 50 request) | 66,38 s (đọc + lọc cả dải 5.000 ID) |
| Thời gian chuẩn bị / tải | 0 | 0 | 4.980 s (1 giờ 23 phút), một lần |
| Ước lượng cho 5.000 ID | ≈ 1.399 s (≈ 23 phút); 574 s theo lần đo 1 | ≈ 17.872 s (≈ 5,0 giờ) | 66,38 s (không tính tải); ≈ 5.046 s (tính cả tải) |

> Chú thích: cột Kaggle phải đọc kèm dòng "thời gian chuẩn bị / tải". Chi phí tải (1 giờ 23 phút) là một lần và có thể chia sẻ cho cả nhóm qua Drive, nhưng vẫn là chi phí thật của phương pháp này.

## 4. Kết luận & lựa chọn

### 4.1 Phương án chọn

**Kết hợp hai nguồn trên toàn bộ 5.000 ID: arXiv API là nguồn chuẩn cho "bài tồn tại" và "số version"; Kaggle snapshot cung cấp ngày từng version và metadata; OAI-PMH `GetRecord` chỉ vá các ca lệch.** Khi hai nguồn mâu thuẫn, lấy API làm chuẩn.

Quy trình:
1. **Quét Kaggle (1 lần):** đọc snapshot bằng pandas theo chunk, lọc `2506.00788 → 2506.05787`. Thu được ID, số version và ngày từng version cho cả 5.000 bài (66,38 s).
2. **Chạy arXiv API cho toàn bộ 5.000 ID** (100 request, batch 50, ≈ 23 phút): xác nhận bài tồn tại và lấy `vN` mới nhất.
3. **So `n_versions` giữa API và Kaggle, lấy API làm chuẩn:**
   - Khớp ⇒ dùng ngày version từ Kaggle.
   - API có `vN` mới hơn Kaggle (snapshot cũ) hoặc bài chỉ có ở API ⇒ gọi OAI `GetRecord` (`arXivRaw`) để lấy ngày các version còn thiếu.
   - Lý do chạy toàn bộ thay vì chỉ lấy mẫu: nếu snapshot thiếu `v(N+1)` thì crawler sẽ bỏ sót version, vi phạm yêu cầu "mọi version". Chi phí chạy hết chỉ khoảng 23 phút nên không đáng để tiết kiệm.

**Tổng thời gian discovery (số liệu cho báo cáo M1), 5.000 ID:**

| Thành phần | Thời gian | Ghi chú |
|---|---|---|
| Quét Kaggle | 66,38 s | đo trực tiếp |
| arXiv API toàn dải | ≈ 1.399 s (≈ 23 phút) | ước lượng từ số đo mẫu |
| **Cộng (chưa tính tải)** | **≈ 1.466 s ≈ 24,4 phút** | |
| Tải file Kaggle | 4.980 s (1 giờ 23 phút) | một lần, chia sẻ cho nhóm |
| **Cộng (lần đầu, tính cả tải)** | **≈ 6.446 s ≈ 1 giờ 47 phút** | |
| Vá bằng OAI | ≈ 3,57 s/bài | ví dụ 5% (250 bài) ≈ 15 phút |

Số bài lệch API–Kaggle: **0/50** trên mẫu; tỉ lệ trên toàn dải sẽ được ghi sau khi chạy API đầy đủ.

### 4.2 Ưu điểm của phương án này

| # | Ưu điểm | Giải thích |
|---|---|---|
| 1 | **Khớp bài toán "theo dải ID"** | Cả API và Kaggle đều lấy được đúng dải. OAI-PMH `ListRecords` chỉ lọc theo ngày/set. |
| 2 | **Đủ dữ liệu mà ít request** | Kaggle cho ngày từng version với 0 request. API chỉ cần khoảng 100 request cho 5.000 bài, thay vì 5.000 request của OAI `GetRecord`. |
| 3 | **Không bỏ sót version** | API là nguồn realtime làm chuẩn nên bắt được version mới hơn snapshot. Đây là điều kiện để đáp ứng "crawl mọi version". |
| 4 | **Tiết kiệm thời gian ở bước crawl thật** | Biết trước `n_versions` thì crawler tải đúng `v1..vN`, không phải dò `v(N+1)` để gặp 404. Mỗi lần dò tốn ≥ 3 s, nhân 5.000 bài ≈ 4 giờ lãng phí. |
| 5 | **Có kiểm chứng chéo** | Hai nguồn độc lập đối chiếu nhau trên toàn bộ dải, không chỉ trên mẫu. |
| 6 | **Nhanh hơn OAI-only** | ≈ 24 phút (chưa tính tải) hoặc ≈ 1 giờ 47 phút (tính cả tải lần đầu), so với ≈ 5 giờ nếu chỉ dùng OAI `GetRecord` cho cả dải. |

### 4.3 Nhược điểm và cách giảm thiểu

| # | Nhược điểm | Cách giảm thiểu |
|---|---|---|
| 1 | **Tải file Kaggle rất lâu** (1 giờ 23 phút ở mạng ~51 kB/s), file 1,73 GB nén, cần tài khoản/API key. | Chỉ một người tải một lần và chia sẻ qua Drive chung. Báo cáo ghi rõ chi phí này, không gộp vào 66,38 s. Nếu chỉ tính riêng dải 5.000 ID của một người thì API một mình (≈ 23 phút) rẻ hơn tổng end-to-end Kaggle (≈ 84 phút); Kaggle có lợi nhờ ngày version và việc dùng lại cho nhiều thành viên. |
| 2 | **Snapshot có thể cũ**: bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu hoặc lệch. | Chạy API toàn dải, lấy API làm chuẩn, vá ngày version bằng OAI `GetRecord`. Trên mẫu 50 ID không có ca lệch. |
| 3 | **Vá bằng OAI tốn thời gian** (≈ 3,57 s/bài). | Chỉ áp dụng cho bài lệch; ước lượng chi phí theo tỉ lệ lệch ở mục 4.5. |
| 4 | **Thời gian phản hồi của API dao động** (2,74 s và 10,99 s ở hai lần đo). | Lập kế hoạch theo số thận trọng (≈ 23 phút), crawler có Checkpoint/Resume. |
| 5 | **Không có thông tin source** (`.tar.gz`/`.gz`). | Không thuộc phạm vi discovery. Source vẫn lấy từ `export.arxiv.org/src/{id}v{n}` ở bước crawl. |
| 6 | **Ngày ở dạng RFC 2822**, có thể lệch múi giờ giữa nguồn. | Chuẩn hóa sang ISO 8601 (UTC) bằng một hàm duy nhất, kiểm tra khớp trên mẫu. |
| 7 | **Phụ thuộc nguồn bên thứ ba** (Kaggle/Cornell). | Nếu Kaggle không dùng được, API + OAI `GetRecord` thay thế (chậm hơn, ≈ 5 giờ cho phần OAI). |

### 4.4 Vì sao không chọn một phương pháp đơn lẻ

- **Chỉ arXiv API:** nhanh (≈ 10–23 phút) và chuẩn về số version, nhưng thiếu ngày version trung gian. Đó là lý do cần nguồn thứ hai.
- **Chỉ OAI-PMH:** `arXivRaw` đầy đủ nhất và chỉ cần một nguồn (đơn giản hơn). Nhưng `ListRecords` không lọc được theo dải ID và buộc quét cả tháng; còn `GetRecord` đo được ≈ 3,57 s/ID, ngoại suy ≈ 5 giờ cho 5.000 ID (nhanh hơn nhiều so với 20 giờ ở endpoint cũ, nhưng vẫn gấp khoảng 12 lần phương án kết hợp khi chưa tính tải, và khoảng 2,8 lần khi tính cả tải lần đầu). Phù hợp làm phương án dự phòng và để vá lẻ.
- **Chỉ Kaggle:** đủ dữ liệu nhưng không đảm bảo độ mới; rủi ro bỏ sót version.

### 4.5 Khi nào đổi phương án

Chi phí vá bằng OAI tăng tuyến tính theo số bài lệch (≈ 3,57 s/bài). Ví dụ 10% (500 bài) ≈ 30 phút. **Trường hợp xấu nhất là mọi bài đều lệch, khi đó tương đương OAI `GetRecord` toàn dải ≈ 5 giờ.** Vì vậy phương án này luôn bị chặn trên ở mức khoảng 5 giờ + 23 phút, và không có kịch bản nào tệ hơn việc dùng OAI một mình. Nếu tỉ lệ lệch rất cao (trên khoảng 50%), nên bỏ bước Kaggle và dùng API + OAI `GetRecord` trực tiếp cho đơn giản. Ngưỡng này là ước lượng, cần đối chiếu với tỉ lệ lệch đo được trên toàn dải.

## 5. Ghi chú kỹ thuật phát hiện được

- **Số version của một bài:** từ API lấy hậu tố `vN` trong `<id>` của entry; từ Kaggle `len(versions)`; từ OAI đếm các phần tử `<version>`.
- **Ngày từng version:** API chỉ có `published` (v1) và `updated` (mới nhất); Kaggle `versions[i].created` và OAI `arXivRaw` `<version><date>` đều có đủ. Định dạng là RFC 2822 (`Mon, 2 Jun 2025 12:34:56 GMT`), cần đổi sang **ISO 8601** khi ghi `metadata.json`.
- **Rate limit:** ≥ 3 giây giữa hai request tới arXiv, 1 kết nối tại một thời điểm, dùng `export.arxiv.org` (không dùng `arxiv.org`) và **User-Agent mô tả rõ** kèm email liên hệ. Semantic Scholar: delay 1 s + API key.
- **Rate limit tính theo request, không theo ID:** batch 30 hay 50 ID thì số request mới quyết định thời gian, nên ước lượng tổng thời gian dựa trên số request (`ceil(5000/50) = 100`). Tuy nhiên thời gian xử lý mỗi request có dao động lớn (2,74 s đến 10,99 s).
- **Batch `id_list`:** dùng batch ~50 ID/request; luôn đặt `max_results` bằng kích thước batch, vì mặc định chỉ trả 10.
- **Endpoint OAI-PMH:** dùng `https://oaipmh.arxiv.org/oai`; endpoint cũ chậm hơn khoảng 4 lần trong lần đo của nhóm (14,37 s/ID so với 3,57 s/ID).
- **ID không tồn tại / bị rút:** API bỏ qua entry, OAI trả `idDoesNotExist`; cần log và giữ record rỗng theo yêu cầu đề.
- **Định dạng ID:** từ 2015 là `yymm.NNNNN` (5 chữ số, padding 0, ví dụ `2506.00788`); ID cũ như `hep-th/9901001` chỉ xuất hiện ở phần reference.
- **Source:** nguồn LaTeX lấy qua `export.arxiv.org/src/{id}v{n}`, có thể là `.tar.gz` hoặc một `.gz` đơn chứa `.tex`; bài chỉ có PDF/Word không có source ⇒ giữ record rỗng.

## 6. Việc tiếp theo (phụ thuộc chuỗi P1-00 → P2-01 → P3-02a)

- [x] Điền RAM đỉnh khi đọc Kaggle (mục 2.3).
- [ ] Bàn giao interface discovery cho crawler engine (P2-01): đầu vào `--id_start --id_end`, đầu ra danh sách `{id, n_versions, version_dates}`.
- [ ] Khi chạy crawler thật: chạy API toàn dải, so `n_versions` với Kaggle và ghi tỉ lệ lệch vào báo cáo M1.