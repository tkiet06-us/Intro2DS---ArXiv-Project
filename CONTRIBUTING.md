# Quy tắc đóng góp

## Git
- Nhánh: `feature/<TASK-ID>-<mô-tả>` (vd `feature/P2-01-crawler`), tạo từ `main`.
- Không push trực tiếp vào `main`; mọi thay đổi qua Pull Request.
- Mỗi PR phải có **đủ 2 Reviewers** (Reviewer 1 & 2 theo bảng phân công) Approve mới được merge.
- Commit message: `P2-01: thêm retry cho crawler` (viết bằng tiếng Anh cũng được).
- Issue: Thảo luận các vấn đề gặp phải/Trợ giúp.

## Dữ liệu
- Repo KHÔNG chứa data thật (25.000 bài, .tex/.bib, ground_truth.json) → lưu trên Drive chung / HF Private.
- Chỉ commit mock nhỏ trong `tests/mock_data/`.
- Không commit API key; dùng `.env`.

## Liêm chính học thuật
- Tham khảo code/tài liệu ngoài phải ghi citation trong code (docstring/comment) và báo cáo.
- Gian lận / chép không dẫn nguồn = 0 điểm toàn milestone.
