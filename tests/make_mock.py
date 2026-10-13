"""P1-02 (Owner: Bảo) – Sinh dữ liệu giả lập cực nhẹ vào tests/mock_data/."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "mock_data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paper = {"arxiv_id": "2506.00001", "title": "Mock Paper", "authors": ["A. Author"],
             "submission_date": "2025-06-01T00:00:00Z"}
    (OUT / "mock_paper.json").write_text(json.dumps(paper, indent=2))
    print("Mock data created at", OUT)


if __name__ == "__main__":
    main()
