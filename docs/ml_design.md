# Reference Matching: Thiết kế ML, định dạng Label/Feature và schema `predictions`

> Task **P1-05**: Thiết kế Kiến trúc ML & Predictions Spec (Milestone 3).
> Tài liệu này là hợp đồng chung cho `src/features.py`, `src/matching_model.py`, `src/train.py`, `src/evaluate.py`, `src/export_parquet.py`. Mọi thay đổi định dạng phải sửa ở đây trước.

---

## 1. Bài toán

Với mỗi bài báo (publication), ta có:

- **`bibtex`** (Milestone 2): `m` BibTeX entry đã chuẩn hóa của bài.
- **`references`** (Milestone 1): `n` reference đã crawl từ Semantic Scholar **có arXiv ID** của cùng bài (bảng đã loại sẵn các ref không có arXiv ID).

**Mục tiêu:** với mỗi BibTeX entry, xếp hạng các reference có arXiv ID của **cùng publication** và trả về tối đa **5** ứng viên theo thứ tự điểm giảm dần. Đáp án đúng là một reference trong pool hoặc không có reference nào phù hợp. Pool có thể rỗng; khi đó danh sách ứng viên rỗng và cả năm cột `cand_k` là null.

Ràng buộc từ đề:

- Việc ghép chỉ diễn ra **bên trong một bài**: entry của bài A chỉ được ghép với reference của bài A.
- Mỗi entry có **tối đa một** đáp án đúng.
- Metric: **MRR@5** trên tập Test.

```
bibtex (m entry) ─┐
                  ├─► cặp (entry i, ref j) ─► features ─► model ─► điểm s(i,j) ─► top-5 mỗi entry ─► predictions
references (n ref)┘
```

---

## 2. Khung hóa: Ranking trên ma trận m×n

### 2.1 Ma trận ứng viên

Mỗi bài có một ma trận nhị phân `Y ∈ {0,1}^{m×n}`:

- Hàng `i` = BibTeX entry `i`, cột `j` = reference `j`.
- `Y[i,j] = 1` nếu entry `i` ứng với reference `j`, ngược lại bằng 0.

**Điền theo hàng (row-wise):** mỗi entry đã gán nhãn có **đúng một ô 1** (nếu có đáp án) hoặc **cả hàng toàn 0** (nếu không có reference nào khớp). Entry chưa gán nhãn **không có hàng nào** trong tập huấn luyện/đánh giá.

### 2.2 Cách mô hình hóa (đề xuất)

| Thành phần | Lựa chọn |
|---|---|
| Đơn vị mẫu | Một cặp `(entry i, ref j)` cùng bài |
| Nhóm xếp hạng | Tất cả cặp của cùng một entry (`group = (arxiv_id, citation_key)`) |
| Cách học | **Pointwise**: phân loại nhị phân cho từng cặp, rồi sắp xếp ref theo xác suất. Baseline ML: Logistic Regression. Mô hình chính: Gradient Boosting (LightGBM/XGBoost) |
| Naive baseline | **Rule-based, không huấn luyện**: xếp ref theo `title_lev_sim` (F01) giảm dần, lấy 5 ref đầu. Tie-break cố định theo `ref_arxiv_id` tăng dần; cặp có F01 = NaN (thiếu tiêu đề) xếp sau cùng, cũng theo tie-break đó. Là mốc dưới: mô hình ML chỉ có giá trị nếu thắng baseline này, đặc biệt trên bài **manual** (xem mục 7 và 3.5) |
| Mô hình thay thế | LambdaRank/LambdaMART với `group` theo entry (nếu pointwise cho MRR thấp), có thể so sánh với mô hình trên trong tập dữ liệu `Test`|
| Sinh ứng viên | Lấy toàn bộ `n` ref của bài, không tạo cặp xuyên publication. Không blocking trong baseline; chỉ cân nhắc nếu pool thực tế quá lớn và phải báo cáo ảnh hưởng đến recall |
| Mất cân bằng lớp | Mỗi hàng có tối đa 1 dương trên `n` ô, nên dùng `class_weight`/`scale_pos_weight` hoặc lấy mẫu âm có kiểm soát khi train. **Không** lấy mẫu khi valid/test |
| Nguồn nhãn khi train | Không coi mọi nhãn như nhau: loại hàng `unresolved` của bài auto, giảm trọng số nhãn auto, che feature trùng luật. Chi tiết ở mục 3.5 |
| Hậu xử lý | Sắp giảm dần theo điểm, lấy 5 ref đầu, bỏ trùng; nếu `n < 5` thì các `cand_k` còn lại để `null` |

### 2.3 Chia tập (theo **bài**, không theo entry)

| Tập | Thành phần |
|---|---|
| Test | 1 bài gán tay + 1 bài gán tự động |
| Valid | 1 bài gán tay + 1 bài gán tự động |
| Train | Tất cả bài gán nhãn còn lại (cả tay lẫn tự động) |

Chia theo bài để tránh rò rỉ: entry của cùng một bài chia sẻ reference pool nên không được nằm ở hai tập khác nhau.

> **Lưu ý thống kê:** Test chỉ có 2 bài nên MRR rất dao động. Có thể dùng thêm đánh giá nhóm theo publication (ví dụ leave-one-publication-out) trên Train để chọn mô hình; báo cáo riêng kết quả validation/cross-validation và kết quả Test chính thức. Không dùng nhãn Test để chọn mô hình hay ngưỡng.

---

## 3. Định dạng nhãn (Label format)

### 3.1 Yêu cầu số lượng (từ đề)

| Nguồn | Yêu cầu |
|---|---|
| Gán tay | ≥ **5 bài**, mỗi bài ≥ **20 entry** được gán, tổng ≥ **100 entry** |
| Tự động (regex/string similarity) | Gán nhãn cho ≥ **10%** số bài **ngoài** nhóm gán tay; matcher tự động gán **mọi** entry của các bài đó |

### 3.2 Lưu trữ

Mỗi bài một file: `data/labels/<yymm-id>.json` (ghi bằng `write_json` của `utils_io.py`, tên folder dùng `to_folder_name`). Dữ liệu nhãn nằm ở Drive chung/HF private, **không commit** vào repo. Có thể dùng thư mục `data` với .`gitkeep` để làm file nhớ tạm (nhớ bổ sung `.gitignore`).

```json
{
  "arxiv_id": "2310.12345",
  "label_source": "manual",
  "labels": [
    {
      "citation_key": "vaswani2017",
      "groundtruth_arxiv_id": "1706.03762",
      "label_status": "matched",
      "annotator": "Kiệt",
      "rule": null,
      "note": ""
    },
    {
      "citation_key": "some_book_2005",
      "groundtruth_arxiv_id": null,
      "label_status": "no_match_confirmed",
      "annotator": "Kiệt",
      "rule": null,
      "note": "sách, không có trên arXiv"
    }
  ]
}
```

| Trường | Ý nghĩa |
|---|---|
| `arxiv_id` | ID chuẩn (dạng `2310.12345` hoặc `hep-th/9901001`, không hậu tố version) |
| `label_source` | `"manual"` hoặc `"auto"` (cấp bài) |
| `citation_key` | Khóa trong bảng `bibtex` sau dedup |
| `groundtruth_arxiv_id` | `ref_arxiv_id` đúng, hoặc `null` nếu không có nhãn khớp (xem `label_status` để biết `null` nghĩa là gì) |
| `label_status` | `"matched"`: có đáp án. `"no_match_confirmed"`: **chỉ bài manual**, người gán đã xác nhận pool không có đáp án. `"unresolved"`: **chỉ bài auto**, luật không tìm ra, chưa chắc là no-match thật. Hai trạng thái null có cùng `groundtruth_arxiv_id: null` nhưng cách dùng khi train khác nhau (mục 3.5) |
| `annotator` | Người gán (chỉ với nhãn tay) |
| `rule` | Tên luật đã khớp, ví dụ `"arxiv_in_url"`, `"title_exact"`, `"title_sim>=0.92"` (chỉ với nhãn tự động) |
| `note` | Ghi chú tùy chọn |

Ví dụ record của bài `auto` (khớp bằng luật, và một entry luật không tìm ra):

```json
{ "citation_key": "bahdanau2014", "groundtruth_arxiv_id": "1409.0473", "label_status": "matched",    "annotator": null, "rule": "title_exact", "note": "" },
{ "citation_key": "lecun_abbrev", "groundtruth_arxiv_id": null,         "label_status": "unresolved", "annotator": null, "rule": null,          "note": "" }
```

File `data/labels/split.json` ánh xạ `arxiv_id → {"partition": "train|valid|test", "label_source": "manual|auto"}`. Đây là **nguồn sự thật duy nhất** cho việc chia tập.

### 3.3 Quy tắc kiểm tra nhãn (script `check_labels` phải chạy qua trước khi train)

1. `citation_key` tồn tại trong bảng `bibtex` của bài đó, và không trùng trong file nhãn.
2. `groundtruth_arxiv_id` khác `null` thì phải nằm trong tập `ref_arxiv_id` của **chính bài đó** (nếu đáp án đúng không có trong pool thì ghi `null`).
3. `groundtruth_arxiv_id` đúng dạng ID chuẩn (dùng `validate_arxiv_id`).
4. Bài `auto` phải có một label record cho **mọi** entry BibTeX sau dedup, kể cả khi không tìm được reference (record đó có `groundtruth_arxiv_id: null` và `label_status: "unresolved"`); bài `manual` có ít nhất 20 entry đã gán. Entry bị bỏ sót khỏi file label là **chưa gán nhãn**, không được hiểu là nhãn null.
5. Mỗi bài xuất hiện đúng một lần trong `split.json`; đúng 1 manual + 1 auto ở Test, 1 manual + 1 auto ở Valid.
6. `label_status` nhất quán: `groundtruth_arxiv_id` khác `null` ⇔ `matched`; `null` ⇒ `no_match_confirmed` (bài manual) hoặc `unresolved` (bài auto), không đảo ngược. Record auto `matched` phải có `rule`; bài manual không có `unresolved`.

### 3.4 Ma trận m×n sinh ra khi chạy, không lưu

```python
# với mỗi entry đã gán nhãn của bài P và mỗi ref j trong pool(P):
if label_status[i] == "unresolved":      # auto, luật không tìm ra
    label(i, j) = MASK                    # "chưa biết": KHÔNG phải mẫu âm, loại khỏi loss khi train
else:
    label(i, j) = 1 if ref_arxiv_id[j] == groundtruth_arxiv_id[i] else 0
    # matched            -> một ô 1, các ô còn lại là âm đáng tin
    # no_match_confirmed -> cả hàng 0 (âm đáng tin; có dùng khi train hay không: xem mục 3.5)
# Khi ĐÁNH GIÁ (mục 7): mọi labeled entry, kể cả "unresolved", vẫn nằm trong |Q|; null → điểm 0.
```

### 3.5 Nhãn tự động: rủi ro và chính sách huấn luyện

Nhãn auto do luật sinh ra (`arxiv_in_url`, `title_exact`, `title_sim>=0.92`…) nên không phải ground truth. Đưa toàn bộ vào train như nhãn thật có ba rủi ro:

1. **`null` của bài auto là "luật không tìm ra", không phải "không khớp".** Luật có precision cao nhưng recall thấp: entry có tiêu đề viết tắt, bị cắt hoặc sai chính tả sẽ rơi vào `null` dù đáp án nằm trong pool. Nếu coi cả hàng là mẫu âm cứng, mô hình học điều sai "tiêu đề hơi lệch hoặc viết tắt ⇒ không khớp", đúng là các ca khó mà MRR cần xếp đúng. Đây là nhiễu một phía (false negative), giống bài toán positive-unlabeled.
2. **Học lại luật thay vì quy luật tổng quát.** Mẫu dương auto chỉ gồm các ca luật bắt được (lệch về ca dễ), và feature gần như trùng định nghĩa luật (`arxiv_in_url` ↔ F09; `title_sim>=0.92` ↔ F01–F04, F10–F11). Mô hình có thể đạt điểm cao trên Train/Valid auto mà không học thêm gì so với luật.
3. **Mất cân đối khối lượng.** Theo kế hoạch 25.000 bài, ≥10% (≈2.500 bài) là auto, trong khi nhãn manual chỉ ≥100 entry. Không có trọng số thì nhãn auto áp đảo tín hiệu thật.

Một điểm làm giải pháp đơn giản hơn: hệ thống luôn trả top-5 và hàng không có đáp án có điểm 0 **bất kể thứ tự**, nên các hàng đó không đóng góp gì cho MRR. Loại chúng khỏi loss không làm mất khả năng tối ưu MRR (LambdaRank cũng cho gradient 0 với group không có mẫu dương). Mô hình không cần học "no-match", chỉ cần xếp đáp án lên cao ở các hàng có đáp án.

**Chính sách mặc định khi train**

| Loại hàng | Dùng khi train? | Trọng số mẫu |
|---|---|---|
| manual · `matched` | Có | 1.0 |
| manual · `no_match_confirmed` | Không (mặc định); thử thêm làm hàng âm trong ablation | — |
| auto · `matched` | Có; các ô còn lại trong hàng là âm | `w_auto` < 1 (thử 0.1 / 0.3 / 0.5, chọn trên Valid) |
| auto · `unresolved` | **Không**: mask khỏi loss, không tính là âm | — |

**Che feature trùng luật.** Với hàng auto khớp bởi luật R, đặt NaN (chỉ khi train) cho feature trùng định nghĩa R, ví dụ `arxiv_in_url` → F09, để mô hình không học "F09 = 1 ⇒ nhãn" từ chính luật sinh nhãn. Với luật theo tiêu đề thì không che hết được F01–F04/F10–F11 vì sẽ mất gần hết tín hiệu; ở đó dựa vào `w_auto` thấp và kiểm chứng trên bài manual.

**Đo độ tin cậy của luật.** Chạy auto-labeler trên các bài manual thuộc Train/Valid (không dùng Test) và so với nhãn tay: precision/recall từng `rule`, và tỉ lệ entry `unresolved` mà thực ra có đáp án (false-null rate). Số liệu này quyết định giữ hay bỏ luật nào, chọn `w_auto`, và là một kết quả đáng đưa lên poster.

**Chọn chính sách bằng ablation** (theo MRR@5 trên Valid, ưu tiên bài manual, kèm leave-one-publication-out trên các bài manual của Train; không dùng Test):

| Cấu hình | Dữ liệu train |
|---|---|
| A | chỉ manual `matched` |
| B | A + auto `matched` (trọng số `w_auto`, che feature trùng luật) |
| C | B + manual `no_match_confirmed` làm hàng âm |
| D (đối chứng, không dùng) | A + toàn bộ auto, `null` làm mẫu âm: chỉ để chứng minh rủi ro 1 bằng số liệu |

**Khi đánh giá.** Ở Test/Valid, `null` của bài auto vẫn tính điểm 0 theo đề (mục 7), nhưng có thể là false-null, còn nhãn dương lại do chính luật sinh ra. MRR trên bài auto vì vậy phản ánh mức đồng thuận với luật hơn là độ đúng thật. Báo cáo riêng manual/auto và rút kết luận chính từ bài manual. Self-training/pseudo-label trên các hàng `unresolved` không nằm trong mặc định vì dễ củng cố chính lỗi của luật; chỉ thử sau khi A–C chạy ổn.

---

## 4. Tiền xử lý (Data Cleaning)

Áp dụng giống hệt cho entry và reference, trong một hàm dùng chung để hai bên không lệch:

| Bước | Mô tả |
|---|---|
| Bỏ lệnh LaTeX trong trường BibTeX | Xóa `\emph{}`, `{}`, `\'e`, `$...$`… (giữ chữ) |
| Chuẩn hóa Unicode | NFKD, bỏ dấu (ví dụ `é → e`) |
| Hạ chữ thường, bỏ dấu câu, gộp khoảng trắng | |
| Stop-words | Chỉ bỏ ở **tiêu đề**, không bỏ ở tên tác giả |
| Chuẩn hóa tên tác giả | Định nghĩa rõ parser theo từng author: dạng BibTeX `Family, Given` tách tại dấu phẩy; dạng `Given Family` lấy token cuối làm họ và token đầu làm initial. Chuẩn hóa dấu chấm/khoảng trắng; với tên nhiều thành phần hoặc hậu tố (Jr., von, họ kép), giữ fallback tokenized form thay vì đoán sai. So sánh Jaccard trên dạng chuẩn hóa, không gộp hai tác giả khác nhau chỉ vì initial trùng |
| Cụm `et al.` | Chạy **trước** khi đếm hoặc tách tác giả. Nhận diện ở cuối danh sách (sau khi hạ chữ thường, bỏ dấu câu và `~`): `et al`, `others` (kể cả `and others`). Loại phần tử đó khỏi danh sách và đặt `entry_has_et_al = 1`; danh sách còn lại mới dùng cho F05–F07, F13. Không để `al`/`others` lọt vào tập họ như một tác giả. Nếu danh sách rỗng sau khi loại thì coi `authors` là thiếu. Reference từ Semantic Scholar kỳ vọng có danh sách đầy đủ nên không có cờ tương ứng; kiểm tra bằng thống kê độ dài danh sách |
| Năm | Entry lấy `year` (int hoặc null); reference lấy năm đầu tiên hợp lệ trong `ref_submission_date` ISO hoặc null. Không parse được/thiếu thì feature `year_diff = NaN` kèm missing flag |

**Xử lý thiếu dữ liệu:** entry sinh từ `\bibitem` bằng regex có thể thiếu `title` hoặc `authors`; reference cũng có thể thiếu metadata. Feature so sánh hai phía trả `NaN` nếu thiếu đầu vào cần thiết và có cờ thiếu riêng cho entry/reference. Với feature có mẫu số 0 (ví dụ title rỗng sau cleanup), trả `NaN`, không chia cho 0. Mô hình cây có thể nhận NaN tùy thư viện/cấu hình; pipeline phải kiểm tra khả năng này. Logistic Regression cần impute giá trị trung tính **chỉ fit imputer trên Train**, giữ cờ missing, và áp dụng cùng imputer cho Valid/Test.

---

## 5. Feature Engineering

Mỗi feature phải có: **công thức**, **ý tưởng**, **minh họa bằng phân tích dữ liệu** (ví dụ phân phối giá trị giữa cặp dương và cặp âm) để đưa lên poster và README.

### 5.1 Danh sách feature đề xuất

| ID | Tên | Đầu vào | Công thức / ý tưởng | Miền |
|---|---|---|---|---|
| F01 | `title_lev_sim` | title | `1 − Levenshtein(a,b)/max(len)` trên tiêu đề đã làm sạch. Bắt lỗi chính tả, khác biệt nhỏ. Cài bằng `rapidfuzz` (mục 5.1.1) | [0,1] |
| F02 | `title_jaccard_char3` | title | Jaccard của tập 3-gram ký tự. Bền với sai khác thứ tự và dấu câu | [0,1] |
| F03 | `title_tfidf_cos` | title | Cosine TF-IDF (từ). Giảm trọng số các từ chung như "learning", "neural". **Chỉ fit trên Train** | [0,1] |
| F04 | `title_containment` | title | `\|tok(entry) ∩ tok(ref)\| / \|tok(entry)\|`. Bắt tiêu đề bị cắt cụt hoặc thiếu phụ đề | [0,1] |
| F05 | `author_jaccard` | authors | Jaccard tập họ tác giả. Bị kéo thấp khi entry viết `et al.` (entry chỉ liệt kê vài người, ref có đủ), nên đọc cùng F13, F14 | [0,1] |
| F06 | `first_author_match` | authors | 1 nếu họ tác giả đầu trùng | {0,1} |
| F07 | `n_authors_diff` | authors | `abs(len(entry_authors) − len(ref_authors))`. Phân biệt bài cùng tiêu đề nhưng khác nhóm tác giả; NaN nếu thiếu danh sách một bên **hoặc** `entry_has_et_al = 1` (số tác giả entry chỉ là cận dưới) | ℕ₀ |
| F08 | `year_diff_abs` | year, `ref_submission_date` | `\|year_ref − year_entry\|`. Năm xuất bản có thể lệch năm nộp arXiv ≥ 1, nên **không** dùng làm bộ lọc cứng. Có dấu: xem F15 | ℕ₀ |
| F09 | `arxiv_id_in_entry` | mọi trường BibTeX | 1 nếu ID của candidate xuất hiện sau chuẩn hóa trong trường entry; 0 nếu trường được kiểm tra đầy đủ nhưng không chứa ID; NaN nếu không có trường nào để kiểm tra. Chỉ kiểm tra ID ứng viên hiện tại, không dùng ground truth | {0,1}; NaN theo quy tắc ở cột công thức |
| F10 | `title_sim_rank_in_pool` | F01 | Hạng giảm dần của candidate trong pool ref của entry (**theo hàng**); quy tắc tie cố định (dense rank). Nếu title thiếu/không có điểm thì NaN | ℕ⁺ |
| F11 | `title_sim_margin` | F01 | `F01(candidate) − max(F01(other candidates))`; nếu pool chỉ có 1 ref, margin = F01(candidate); nếu không tính được thì NaN | ℝ |
| F12 | missing flags | title/authors/year | Cờ thiếu độc lập cho trường entry và reference dùng trong feature, ví dụ `entry_title_missing`, `ref_title_missing`, `entry_authors_missing`, `ref_authors_missing`, `entry_year_missing`, `ref_year_missing` | {0,1} |
| F13 | `author_containment` | authors | `\|A_entry ∩ A_ref\| / \|A_entry\|` trên tập họ chuẩn hóa, với `A_entry` đã loại `et al.` (mục 4). Chỉ hỏi "các tác giả entry liệt kê có nằm trong danh sách ref không", nên không bị phạt khi entry rút gọn bằng `et al.` như Jaccard. NaN nếu thiếu danh sách một bên hoặc `A_entry` rỗng | [0,1] |
| F14 | `entry_has_et_al` | authors | 1 nếu danh sách tác giả entry bị rút gọn bằng `et al.`/`others` (cờ đặt ở mục 4); 0 nếu không; NaN nếu entry thiếu tác giả. Cho mô hình biết khi nào F05, F07 kém tin cậy | {0,1} |
| F15 | `year_diff_signed` | year, `ref_submission_date` | `year_ref − year_entry`. Năm nộp arXiv thường ≤ năm xuất bản chính thức nên giá trị tập trung ở ≤ 0; dấu mang thông tin (xem 5.1.1). NaN nếu thiếu một bên | ℤ |
| F16 | `title_sim_col_rank_in_paper` | F01 | Hạng **theo cột**: với ref `j` cố định, xếp hạng giảm dần F01 của **mọi entry trong cùng bài**; trả hạng của entry hiện tại (dense rank, tie-break như F10). Hạng 1 nghĩa là entry này là entry khớp nhất với ref `j` trong cả bài. NaN nếu F01 = NaN | ℕ⁺ |

### 5.1.1 Ghi chú triển khai cho bộ feature

- **Tốc độ (F01 và các feature dựng từ F01).** Dùng `rapidfuzz` (lõi C++), không tự viết Levenshtein bằng Python thuần. Công thức F01 giữ nguyên: `rapidfuzz.distance.Levenshtein.normalized_similarity(a, b)` bằng `1 − Levenshtein(a,b)/max(len(a),len(b))`. Với cả pool của một bài, dùng `rapidfuzz.process.cdist(entry_titles, ref_titles, scorer=Levenshtein.normalized_similarity, workers=-1)` để lấy luôn ma trận F01 kích thước `m×n`; F10, F11, F16 đều suy ra từ ma trận này (hạng theo hàng và theo cột) nên không phải tính lại. Cần giữ hai điểm: (1) title rỗng sau cleanup phải gán NaN **sau** khi gọi, vì `rapidfuzz` trả 1.0 cho hai chuỗi rỗng, trái quy tắc "mẫu số 0 → NaN" ở mục 4 (kiểm tra lại khi viết test); (2) thêm test đối chiếu với cài đặt tham chiếu trên vài cặp cố định, và ghim version `rapidfuzz` trong `requirements.txt` (task P1-01).
- **F16 là feature mềm, không phải ràng buộc.** F10 (theo hàng) trả lời "ref `j` có phải ref khớp nhất với entry `i` không"; F16 (theo cột) trả lời "entry `i` có phải entry khớp nhất với ref `j` không". Cặp hạng 1 theo cả hai chiều thường là cặp đúng; cây quyết định tự học tương tác F10×F16 nên không cần thêm cờ. Ràng buộc ở mục 1 chỉ nói mỗi entry có tối đa một đáp án, không nói mỗi ref chỉ được một entry trích (entry trùng sót lại sau dedup vẫn có thể cùng trỏ một ref), vì vậy **không** ép phân công 1–1 kiểu Hungarian khi xếp hạng. F16 tính trên **toàn bộ `m` entry sau dedup** của bài, không chỉ entry đã gán nhãn; nếu không, bài manual (chỉ gán ≥20 entry) và bài auto (gán mọi entry) sẽ có phân phối F16 khác nhau.
- **`et al.` (F13, F14).** Entry sinh từ `\bibitem` bằng regex hay chỉ có `Vaswani et al.`; nếu để nguyên, F05 bị kéo thấp và F07 đếm sai số tác giả dù đúng bài. Giải pháp: tách `et al.` ở mục 4, dùng cờ F14 và phép đo chứa trùng F13. F06 (họ tác giả đầu) không bị ảnh hưởng. Cần thống nhất với owner task P2-04 để extractor không đưa `et al.` vào `authors` như một tác giả (hoặc để lại nguyên và để bước ở mục 4 xử lý, miễn là một nơi làm).
- **Năm có dấu (F15).** Giữ cả F08 (trị tuyệt đối) và F15: mô hình tuyến tính (Logistic Regression) không tự suy ra "gần 0 là tốt" từ giá trị có dấu. Quy luật "arXiv ≤ năm xuất bản" không tuyệt đối (tác giả có thể đăng arXiv sau khi xuất bản, entry có thể ghi chính năm arXiv) nên không dùng làm luật cứng; kiểm chứng bằng histogram F15 của cặp dương ở phần phân tích dữ liệu (mục 5.3).

### 5.2 Feature tùy chọn dùng cây phân cấp M2

Đề cho phép dùng hierarchy của Milestone 2. Ví dụ ngữ cảnh trích dẫn: câu chứa `\cite{key}`, section chứa nó (Introduction/Related Work/Appendix), số lần entry được trích. Chỉ làm sau khi bộ feature cơ bản chạy ổn.

**Thứ tự xử lý feature tương đối:** tính trên đầy đủ candidate pool của từng entry sau cleaning, trước negative sampling. Không dùng nhãn để tạo feature. Rank/margin (F10, F11) phải tham chiếu cùng full pool ref ở Train/Valid/Test; F16 phải tham chiếu cùng toàn bộ entry của bài; nếu không, ghi rõ sai khác vì sampling làm đổi phân phối.

### 5.3 Mẫu tài liệu hóa cho mỗi feature

```
Feature: <tên>
Công thức: ...
Ý tưởng: vì sao feature này giúp phân biệt cặp dương/âm
Phân tích: <biểu đồ/bảng: phân phối theo nhãn, tỉ lệ thiếu dữ liệu>
Xử lý thiếu: ...
```

---

## 6. Schema bảng `predictions` (9 cột)

Một dòng cho mỗi BibTeX entry của các bài thuộc Train/Valid/Test. Bài không dùng cho matching **không** có dòng.

| Cột | Kiểu Parquet | Null? | Nội dung |
|---|---|---|---|
| `arxiv_id` | string | **không** | Bài chứa entry (ID chuẩn) |
| `partition` | dictionary(int8, string) | không | `train` \| `valid` \| `test` |
| `citation_key` | string | **không** | Khóa BibTeX của entry |
| `groundtruth_arxiv_id` | string | có | `ref_arxiv_id` đúng khi đã biết; `null` nếu không có/không gán |
| `cand_1` … `cand_5` | string | có | Top-5 ứng viên, `cand_1` tốt nhất |

```python
pa.schema(
    [
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("partition", pa.dictionary(pa.int8(), pa.string()), nullable=False),
        pa.field("citation_key", pa.string(), nullable=False),
        pa.field("groundtruth_arxiv_id", pa.string(), nullable=True),
    ]
    + [pa.field(f"cand_{k}", pa.string(), nullable=True) for k in range(1, 6)]
)
```

### 6.1 Ràng buộc nội dung

1. `(arxiv_id, citation_key)` là **duy nhất**.
2. `groundtruth_arxiv_id` khác null chỉ khi entry đã gán nhãn và có đáp án; khi đó phải đúng dạng ID chuẩn và nằm trong pool `references` của chính bài. Mỗi `cand_k` khác null phải đúng dạng ID chuẩn và nằm trong pool đó.
3. Các `cand_k` khác `null` **không trùng nhau** và **không có khoảng trống** (`cand_3` có giá trị thì `cand_2` cũng phải có).
4. `cand_k` là null khi pool có ít hơn `k` ref; không điền placeholder. Pool rỗng thì cả năm cột null.
5. `partition` của mọi entry cùng bài là như nhau và khớp `split.json`.
6. Nullability trong file Parquet phải khớp bảng trên: `arxiv_id`, `partition`, `citation_key` là `nullable=False` (không dòng nào null); `groundtruth_arxiv_id`, `cand_1…cand_5` là `nullable=True`. Test schema đọc lại file đã ghi và so `schema.field(name).nullable`.

### 6.2 Ví dụ

| arxiv_id | partition | citation_key | groundtruth_arxiv_id | cand_1 | cand_2 | cand_3 | cand_4 | cand_5 |
|---|---|---|---|---|---|---|---|---|
| 2310.12345 | test | vaswani2017 | 1706.03762 | 1706.03762 | 1810.04805 | 1409.0473 | 1512.03385 | 1412.6980 |
| 2310.12345 | test | some_book_2005 | *null* | 1810.04805 | 1706.03762 | 1409.0473 | 1512.03385 | 1412.6980 |

---

## 7. Đánh giá: MRR@5

$$\text{MRR@5} = \frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\text{rank}_i}$$

- `|Q|` = số BibTeX entry **đã gán nhãn** trong hai publication Test, theo đề. Bao gồm entry có một đáp án và entry được xác nhận không có reference phù hợp.
- `rank_i` = vị trí từ 1 đến 5 của đáp án đúng trong candidates. Entry được gán là không có đáp án (`groundtruth_arxiv_id = null`) có điểm **0**; đáp án ngoài top-5 cũng có điểm **0**. Không bỏ các entry no-match khỏi mẫu số.
- Entry **chưa gán nhãn** không thuộc `Q`, dù bảng `predictions` vẫn có dòng cho entry đó; không đưa entry này vào tính MRR.

**Ví dụ** (4 entry):

| Entry | Đáp án ở vị trí | Điểm |
|---|---|---|
| A | cand_2 | 0.5 |
| B | cand_1 | 1.0 |
| C | ngoài top-5 | 0 |
| D | được xác nhận không có đáp án (`null`) | 0 |

MRR@5 = (0.5 + 1 + 0 + 0) / 4 = **0.375**. D vẫn ở mẫu số vì là entry đã gán nhãn, dù không có ground truth.

**Báo cáo:** MRR@5 chính thức trên toàn bộ labeled Test entries; thêm MRR riêng cho publication manual và auto. Trường hợp nhóm con không có labeled entries thì ghi N/A, không chia cho 0.

Mọi bảng kết quả (Valid và Test) gồm cả **Naive baseline (F01)**, Logistic Regression và mô hình chính, cùng cách tính MRR@5. MRR trên bài auto phản ánh mức đồng thuận với luật gán nhãn (nhãn `null` có thể là false-null; nhãn dương do luật sinh, mục 3.5) nên không dùng làm bằng chứng chính cho khả năng tổng quát hóa; kết luận chính dựa trên bài manual. Naive baseline cũng đặc biệt mạnh trên bài auto vì luật `title_sim>=0.92` gần như trùng với F01.

> Lưu ý (theo file đồ án): nhóm dùng tri thức ngoài (LLM, mô hình pretrained) sẽ được **chấm riêng** với các nhóm còn lại. Nếu dùng embedding pretrained thì phải ghi rõ trong báo cáo.

---

## 8. Cấu trúc code và đầu ra

| File | Vai trò |
|---|---|
| `src/features.py` | Làm sạch (mục 4, gồm tách `et al.`) và tính feature (mục 5) cho cặp `(entry, ref)`; ma trận F01 bằng `rapidfuzz.process.cdist` |
| `src/matching_model.py` | Định nghĩa mô hình, hàm `score(pairs)` và `top5(entry)`; thêm `naive_top5(entry)` (Naive baseline F01, không huấn luyện) |
| `src/train.py` | Đọc `split.json` + nhãn, sinh ma trận (mask hàng `unresolved`), áp chính sách mục 3.5 (trọng số `w_auto`, che feature trùng luật), huấn luyện, chạy ablation A–D, chọn tham số trên Valid |
| `src/evaluate.py` | Tính MRR@5, xuất `predictions.parquet` (9 cột) |
| `data/labels/` | Nhãn (mục 3), **không commit** |
| `tests/` | Test cho `check_labels` (gồm `label_status`), ràng buộc 6.1 (gồm nullability), MRR trên ví dụ mục 7, F01 `rapidfuzz` so với cài đặt tham chiếu, và rank theo hàng/cột (F10, F16) trên ma trận mẫu |

---

## 9. Quyết định đã chốt và điểm tích hợp cần kiểm tra

1. **MRR và no-match:** đề đã chốt: mọi labeled entry nằm trong `|Q|`, kể cả hàng toàn 0/no-match; điểm no-match là 0. Không cần hỏi giảng viên hay báo hai công thức.
2. **Số dòng predictions:** xuất một dòng cho **mọi BibTeX entry** của mỗi publication thuộc Train/Valid/Test. `groundtruth_arxiv_id = null` trong bảng công bố áp dụng cả cho entry no-match và entry chưa gán nhãn, nên bảng này một mình không đủ để tái tạo tập đánh giá. Giữ label files ngoài dataset công bố; metric chỉ join đúng tập entry có record label. Không thêm cột vào schema predictions.
3. **Tên và số bảng công bố:** đề chính thức yêu cầu đúng 6 bảng: `papers`, `references`, `elements`, `hierarchy`, `bibtex`, `predictions`. Schema hiện tại trong `src/schemas.py` đặt tên `paper_references`, `element_parents`, `bibtex_entries` và thêm `sources` (7 bảng); đây là schema làm việc nội bộ, **không trùng trực tiếp** với schema publish theo đề. Trước P4-02, cần một mapping/export layer tường minh: `paper_references → references`, `element_parents → hierarchy`, `bibtex_entries → bibtex`; `sources` không nằm trong sáu bảng publish. Kiểm tra từng cột/type/nullability với các statement M1/M2; không đổi tên hay xóa bảng nội bộ chỉ vì tên publish khác.
4. **Dạng ranking và model:** pointwise binary classifier là lựa chọn triển khai ban đầu, không phải yêu cầu bắt buộc của đề. Dùng Naive baseline (F01, mục 2.2) làm mốc dưới, Logistic Regression làm baseline ML, và chọn một mô hình chính theo kết quả validation theo publication. Chỉ thử LambdaRank nếu số query/group và thư viện cho phép; ghi rõ mỗi group là một entry và candidate order. Chọn model/threshold chỉ bằng Train/Valid, không bằng Test.
5. **Source text:** schema `bibtex` publish không có raw `\\bibitem`. Có thể giữ raw citation trong artifact làm việc ngoài published table để cải thiện parsing/features, nhưng ghi rõ provenance/cleanup và không phụ thuộc vào file mà nhóm không thể tái tạo/lưu.
6. **Ground truth trong predictions:** exporter đọc label files để điền groundtruth cho labeled row (gồm null cho no-match), đồng thời để null cho unlabeled row. Đây là annotation phục vụ lưu kết quả, không được đưa vào input features hoặc scoring model.
7. **Nhãn auto khi train:** `null` của bài auto là `unresolved`, không phải no-match; không dùng làm mẫu âm. Chính sách đầy đủ ở mục 3.5. Label record vẫn có đủ mọi entry của bài auto (để evaluator tính `|Q|` đúng theo đề), chỉ thêm `label_status`.
8. **Feature mới (F13–F16)** được thêm vào cuối bảng để không đổi ID F01–F12 đã dùng trong tài liệu và test. F08 đổi tên thành `year_diff_abs` (chưa có code phụ thuộc, task P2-06 chưa bắt đầu).
9. **Phụ thuộc:** thêm `rapidfuzz` (ghim version) vào `requirements.txt` (task P1-01); cập nhật `src/schemas.py` (task P1-02) cho `partition` `nullable=False`; script auto-label của task P3-07 phải ghi `rule` và `label_status` theo mục 3.

### 9.1 Contract giữa nhãn, candidates và predictions

| Trường hợp | Có label record? | `label_status` | `groundtruth_arxiv_id` trong label nội bộ | `groundtruth_arxiv_id` trong predictions | Đưa vào MRR? | Dùng để train? |
|---|---:|---|---|---|---:|---|
| Đã gán, có đáp án | Có | `matched` | Candidate ID | Candidate ID | Có | Có (auto: trọng số `w_auto`) |
| Manual, no-match | Có | `no_match_confirmed` | `null` | `null` | Có, điểm 0 | Mặc định không; ablation C |
| Auto, luật không tìm ra | Có | `unresolved` | `null` | `null` | Có, điểm 0 | **Không** (mask) |
| Chưa gán | Không | Không có record | Không có record | `null` | Không | Không |

Do ba trường hợp cuối đều là null trong Parquet, evaluator phải lấy danh sách labeled `(arxiv_id, citation_key)` từ label files trước khi tính metric.
