# Khảo sát Discovery & Phương pháp Crawling (P1-00)

> **Dải ID:** `2506.00788` → `2506.05787` (5.000 bài)
> **Mẫu thử:** `⟨N⟩` ID liên tiếp bắt đầu từ `2506.00788` · **Ngày đo:** `14/10/2026`



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
- **Không có:** ngày của các version trung gian (v2 nếu có v3). Muốn đủ phải lấy thêm từ nguồn khác (OAI `arXivRaw` hoặc trang abs).
- **Ưu:** truy vấn thẳng theo ID nên khớp hoàn toàn với việc được giao theo dải ID; dữ liệu luôn mới; 1 request chứa được cả chục đến vài chục ID.
- **Nhược:** bị giới hạn tốc độ (≥ 3 giây/request, 1 kết nối); ID không tồn tại thì bị bỏ qua trong feed, cần so khớp với danh sách đã hỏi; thiếu ngày các version trung gian.
- **Thời gian đo được:** `⟨x⟩` s cho `⟨N⟩` ID (`⟨r⟩` request, batch = 50). Ước lượng 5.000 ID: `⟨5000/50 × (3 s + latency)⟩` ≈ `⟨y⟩` s.

### 2.2 OAI-PMH (thư viện `sickle`)

- **Cách hoạt động:** endpoint `http://export.arxiv.org/oai2`. Hai verb liên quan:
  - `GetRecord` (identifier `oai:arXiv.org:<id>`) – lấy **một** bài.
  - `ListRecords` – thu hoạch hàng loạt, lọc theo **ngày (`from`/`until`) và `set`**, phân trang bằng `resumptionToken`.
- **Format `arXivRaw`:** chứa danh sách `<version version="vN">` với **ngày và kích thước từng version** ⇒ đây là nguồn chính xác nhất cho "số version" và "ngày từng version".
- **Ưu:** đủ ngày + size mọi version; chuẩn thu hoạch metadata, bền cho việc quét lớn.
- **Nhược:** `ListRecords` **không lọc được theo dải ID**, chỉ theo ngày/set. Muốn lấy dải 788 → 5787 của 2025-06 phải quét cả tháng (hàng chục nghìn bài) rồi lọc bỏ phần thừa. `GetRecord` thì 1 request/ID ⇒ 5.000 request × ≥ 3 s. Có thể gặp `503 Retry-After`.
- **Thời gian đo được:** `GetRecord` `⟨N⟩` ID = `⟨x⟩` s; `ListRecords` 1 trang (~1000 record) = `⟨y⟩` s ⇒ quét cả tháng ≈ `⟨z⟩` s (ước lượng).

### 2.3 Kaggle dataset (`Cornell-University/arxiv`, đọc bằng pandas)

- **Cách hoạt động:** tải file `arxiv-metadata-oai-snapshot.json` (JSON Lines, vài GB), đọc bằng `pd.read_json(lines=True, chunksize=...)`, lọc `id` bắt đầu bằng `2506.` và phần số trong `[788, 5787]`.
- **Dữ liệu có:** `id, title, authors, categories, doi, journal-ref, update_date` và `versions: [{version, created}]` ⇒ **có số version và ngày từng version** trong một lần đọc.
- **Ưu:** không tốn request nào tới arXiv, không dính giới hạn 3 s; lọc dải ID chỉ là thao tác pandas; một lần đọc là xong toàn bộ 5.000 bài.
- **Nhược:** là snapshot (cập nhật định kỳ), bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu; cần tải file lớn và API key Kaggle; không có thông tin về *source* (file `.tar.gz`) – vẫn phải crawl `export.arxiv.org/src/...`.
- **Thời gian đo được:** tải `⟨a⟩` s (`⟨GB⟩` GB); đọc + lọc dải ID `⟨b⟩` s; RAM đỉnh `⟨c⟩` MB (chunksize `200_000`).

## 3. Bảng so sánh

| Tiêu chí | arXiv API (`id_list`) | OAI-PMH (`sickle`) | Kaggle snapshot |
|---|---|---|---|
| Truy vấn theo dải ID |  trực tiếp |  (`ListRecords`) /  1 ID/req (`GetRecord`) |  lọc bằng pandas |
| Số version |  từ hậu tố `vN` của `<id>` |  đếm `<version>` |  `len(versions)` |
| Ngày từng version |  chỉ v1 và mới nhất |  (kèm size) |  (`created`) |
| Phụ thuộc rate limit 3 s | Có | Có | Không |
Độ mới dữ liệu | Realtime | Realtime | Theo snapshot |
| Số request cho 5.000 bài | ~100 | ~5.000 (GetRecord) | 0 |
| Thời gian đo (30 ID) | 2.74 s | 431.14 s | 57.44 s (quét luôn 5000 bài) |
| Thời gian ước lượng (5.000 ID) | 456.67 s | 71856.67 s (~20 tiếng) | 57.44 s |

## 4. Kết luận & lựa chọn

### 4.1 Phương án chọn

**Kaggle snapshot làm nguồn discovery chính, arXiv API `id_list` để đối chiếu, OAI-PMH `GetRecord` chỉ để vá các ca thiếu.** (Cần xác nhận lại bằng số đo ở mục 2–3.)

Quy trình ba bước:
1. **Quét Kaggle (1 lần):** đọc snapshot bằng pandas theo chunk, lọc `2506.00788 → 2506.05787`. Thu được danh sách ID tồn tại, số version và ngày từng version.
2. **Đối chiếu bằng arXiv API:** hỏi `id_list` theo lô 50 cho các ID có trong dải nhưng *không* thấy trong snapshot, và kiểm tra mẫu ngẫu nhiên xem số version hai nguồn có khớp không.
3. **Vá bằng OAI `GetRecord` (`arXivRaw`):** chỉ cho số ít bài mà API/Kaggle lệch nhau hoặc thiếu ngày version.

### 4.2 Ưu điểm của phương án này

| # | Ưu điểm | Giải thích |
|---|---|---|
| 1 | **Khớp đúng bài toán "theo dải ID"** | Đề giao dải ID liên tiếp. Với Kaggle, lọc dải ID chỉ là một điều kiện pandas. OAI-PMH thì chỉ lọc được theo ngày/set. |
| 2 | **Không tốn request tới arXiv** | Không bị giới hạn ≥ 3 s/request. Toàn bộ 5.000 bài được xác định trong một lần đọc file. |
| 3 | **Có sẵn số version và ngày từng version** | Đúng thứ cần cho `metadata.json` (ngày nộp v1 và các lần revised, đổi sang ISO). API chỉ cho v1 và bản mới nhất. |
| 4 | **Tiết kiệm thời gian ở bước crawl thật** | Biết trước `n_versions` thì crawler tải đúng `v1..vN`, không phải thử `v(N+1)` để dò 404. Mỗi lần dò tốn ≥ 3 s, nhân 5.000 bài ≈ 4 giờ lãng phí. |
| 5 | **Chạy lại được, ổn định** | File cố định nên kết quả discovery lặp lại được (dễ cho Checkpoint/Resume và báo cáo), không phụ thuộc mạng hay trạng thái server. |
| 6 | **Có lớp kiểm chứng** | API là nguồn realtime độc lập, dùng để phát hiện snapshot cũ hoặc thiếu. |

### 4.3 Nhược điểm và cách giảm thiểu

| # | Nhược điểm | Cách giảm thiểu |
|---|---|---|
| 1 | **Snapshot có thể cũ**: bài mới hoặc version mới sau thời điểm snapshot sẽ thiếu hoặc lệch. | Kiểm tra `update_date` lớn nhất của snapshot. So số version Kaggle với API trên mẫu và với mọi ID thiếu. Ghi tỉ lệ lệch vào báo cáo. |
| 2 | **File rất lớn** (vài GB), cần tài khoản và API key Kaggle, tải lâu. | Chỉ tải một lần, lưu Drive chung của nhóm. Đọc theo chunk để RAM thấp, ghi riêng số liệu RAM đỉnh và thời gian đọc. |
| 3 | **Không có thông tin source** (`.tar.gz`/`.gz`). | Đây không phải việc của discovery. Source vẫn lấy từ `export.arxiv.org/src/{id}v{n}` ở bước crawl. |
| 4 | **Số đo "thời gian discovery" không phản ánh chi phí mạng** vì phần lớn là đọc file local. | Ghi rõ trong báo cáo: thời gian tải riêng, thời gian đọc và lọc riêng, cộng thêm thời gian đối chiếu API. Không giấu thời gian tải. |
| 5 | **Ngày `created` ở dạng RFC 2822**, và có thể lệch múi giờ so với ngày API. | Chuẩn hóa sang ISO 8601 (UTC) ở một hàm duy nhất, và kiểm tra khớp với API trên mẫu. |
| 6 | **Phụ thuộc nguồn bên thứ ba** (Kaggle/Cornell). | Có phương án dự phòng bên dưới nên không bị chặn nếu Kaggle lỗi. |

### 4.4 Vì sao không chọn hai phương pháp còn lại làm chính

- **arXiv API làm chính:** dùng được và khớp theo ID, nhưng thiếu ngày version trung gian, và 5.000 ID cần ~100 request × ≥ 3 s. Phần ngày version vẫn phải lấy từ nguồn khác. Đây là phương án dự phòng tốt nhất.
- **OAI-PMH làm chính:** `arXivRaw` đầy đủ nhất nhưng sai "hình dạng" với bài toán. `ListRecords` buộc quét cả tháng 2025-06 rồi bỏ phần thừa, `GetRecord` thì 5.000 request × ≥ 3 s, có thể gặp `503 Retry-After`. Chỉ hợp để vá lẻ tẻ.

### 4.5 Khi nào đổi phương án

Nếu số đo cho thấy snapshot **thiếu hoặc lệch version nhiều** trong dải (`⟨x⟩%` bài), đổi sang: **arXiv API làm chính + OAI `GetRecord` bổ sung ngày version**, chấp nhận thời gian discovery dài hơn (ước lượng `⟨y⟩` s) để đổi lấy dữ liệu mới.

**Tổng thời gian discovery (số liệu cho báo cáo M1):** `⟨...⟩` s cho 5.000 ID theo phương án đã chọn (đo thật trên dải đầy đủ, tách riêng: tải / đọc-lọc / đối chiếu API).

## 5. Ghi chú kỹ thuật phát hiện được

- **Số version của một bài:** từ API lấy hậu tố `vN` trong `<id>` của entry; từ Kaggle `len(versions)`; từ OAI đếm các phần tử `<version>`.
- **Ngày từng version:** API chỉ có `published` (v1) và `updated` (mới nhất); Kaggle `versions[i].created` và OAI `arXivRaw` `<version><date>` đều có đủ. Định dạng là RFC 2822 (`Mon, 2 Jun 2025 12:34:56 GMT`), cần đổi sang **ISO 8601** khi ghi `metadata.json`.
- **Rate limit:** ≥ 3 giây giữa hai request tới arXiv, 1 kết nối tại một thời điểm, dùng `export.arxiv.org` (không dùng `arxiv.org`) và **User-Agent mô tả rõ** kèm email liên hệ. Semantic Scholar: delay 1 s + API key.
- **Batch `id_list`:** dùng batch ~50 ID/request để giảm số request; luôn đặt `max_results` bằng kích thước batch, vì mặc định chỉ trả 10.
- **ID không tồn tại / bị rút:** API bỏ qua entry, OAI trả `idDoesNotExist`; cần log và giữ record rỗng theo yêu cầu đề.
- **Định dạng ID:** từ 2015 là `yymm.NNNNN` (5 chữ số, padding 0, ví dụ `2506.00788`); ID cũ như `hep-th/9901001` chỉ xuất hiện ở phần reference.
- **Source:** nguồn LaTeX lấy qua `export.arxiv.org/src/{id}v{n}`, có thể là `.tar.gz` hoặc một `.gz` đơn chứa `.tex`; bài chỉ có PDF/Word không có source ⇒ giữ record rỗng.
