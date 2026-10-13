# P1-05 – ML Design & Predictions Spec (Owner: Kiệt)
- Bài toán: Ranking m×n cặp (bibitem × bài arXiv), nhãn nhị phân Yes/No
- Output `predictions.parquet` (9 cột): arxiv_id, partition, citation_key, groundtruth_arxiv_id, cand_1..cand_5
- Metric: MRR@5 trên tập Test
- Chia tập: Test = 1 tay + 1 tự động; Valid = 1 tay + 1 tự động; Train = còn lại
## Features
| Feature | Lý do (dựa trên phân tích dữ liệu) |
|---|---|
| Levenshtein | |
| Jaccard n-gram | |
| TF-IDF cosine | |
## Label spec




**Đây chỉ là sample mẫu, người viết phải chỉnh sửa lại**