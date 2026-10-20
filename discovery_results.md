### Ket qua do (3 lan lap, mau 50 ID API/Kaggle, 20 ID OAI)

| Phuong phap | Thoi gian mau, s (mean ± std) | Median | [min, max] | RAM dinh, MB (mean ± std) | RAM nen, MB | So request | Tim thay/mau | Lan OK |
|---|---|---|---|---|---|---|---|---|
| arXiv API | 0.99 ± 0.26 | 1.02 | [0.71, 1.22] | 46.5 ± 0.2 | 40.8 | 1 | 50/50 | 3/3 |
| OAI-PMH GetRecord | 73.78 ± 6.72 | 76.20 | [66.18, 78.95] | 46.6 ± 0.2 | 41.9 | 20 | 20/20 | 3/3 |
| Kaggle (quet ca dai) | 66.76 ± 7.58 | 63.40 | [61.44, 75.44] | 2061.2 ± 4.0 | 78.2 | 0 | 50/50 | 3/3 |

### Uoc luong cho 5000 ID

| Phuong phap | So request | Trung binh | KTC 95% cua trung binh | [min, max] (cuc doan) | So mau |
|---|---|---|---|---|---|
| arXiv API | 100 | 399 s (6.6 phut) | 334 - 463 s | 371 - 422 s | 3 request |
| OAI-PMH GetRecord | 5000 | 19192 s (319.9 phut) | 18423 - 19962 s | 16571 - 27032 s | 60 request |
| Kaggle (quet ca dai) | 0 | 67 s (1.1 phut) (+ tai 4980s = 5047s end-to-end) | 48 - 86 s | 61 - 75 s | 3 lan do |

### Doi chieu so version

So ID duoc doi chieu giua >= 2 nguon: 50; so ID lech: 0 

So ID doi version giua cac lan lap cua cung 1 nguon: 0 

### Kiem tra quy tac quyet dinh

- Phuong an ket hop (API toan dai + quet Kaggle), chua tinh tai: 465 s (7.8 phut); tinh ca tai lan dau: 5445 s.
- OAI GetRecord toan dai: 19192 s (5.3 gio) => gap 41.3 lan (chua tinh tai), 3.5 lan (tinh ca tai).
- Diem hoa von ve thoi gian cua phuong an ket hop so voi OAI don le: ti le lech ≈ 98% (chua tinh tai), ≈ 72% (tinh ca tai).
- Kaggle tim thay [5000, 5000, 5000] bai trong dai (ky vong 5000) => DAT.
- RAM dinh Kaggle (max cac lan) = 2064 MB; nguong rang buoc = 6000 MB (~50% RAM Colab CPU) => DAT (khong phai rang buoc).
- arXiv API tim thay du 50 ID moi lan => DAT.
- OAI-PMH GetRecord tim thay du 20 ID moi lan => DAT.
- Nhat quan version giua cac nguon: DAT.

### Thong tin tai lap

- Chay luc (UTC): 2026-10-20 13:53:25; Python 3.14.0; Windows-11-10.0.26200-SP0; git: 88af6de
- Thu vien: pandas 3.0.2, sickle 0.7.0, psutil 7.2.2, requests 2.33.1, feedparser 6.0.14, kaggle 2.2.4
- Lenh: python discovery.py --contact '<email>' --kaggle-json 'data\raw\arxiv-metadata-oai-snapshot.json' --download-seconds 4980 --repeats 3 --kaggle-version 'Version N' --kaggle-snapshot-date 2026-MM-DD --hash-kaggle
- File Kaggle: arxiv-metadata-oai-snapshot.json, 5.61 GB, mtime (UTC) 2026-10-14 16:55:21, dataset version: Version N, ngay snapshot: 2026-MM-DD, SHA-256: 9ee5aafa99209d6dbf4cd8c138dc662f290a7b49065d7301d4a0152bc0701c75

### Canh bao

- Qua it mau request de uoc luong dang tin: arXiv API (< 10 request). Khoang [min, max] va KTC rat rong; can them --n hoac --repeats.
