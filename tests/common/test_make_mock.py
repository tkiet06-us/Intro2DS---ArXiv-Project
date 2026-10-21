"""Kiểm tra mock generator sinh đủ 6 bảng đúng schema và đủ các tình huống khó."""

import pyarrow.parquet as pq

from src.common.utils_io import read_json
from src.schemas import PARQUET_SCHEMAS
from tests.make_mock import PARQUET_FILE_NAME, generate

# Sáu bảng của dataset cuối theo đề M1/M2/M3.
SPEC_TABLES = {"papers", "references", "elements", "hierarchy", "bibtex", "predictions"}


def read_table(root, name):
    return pq.read_table(root / "parquet" / name / PARQUET_FILE_NAME)


def test_schema_defines_exactly_the_six_spec_tables():
    assert set(PARQUET_SCHEMAS) == SPEC_TABLES


def test_generate_writes_all_tables_with_expected_schema(tmp_path):
    generate(tmp_path)

    for name, schema in PARQUET_SCHEMAS.items():
        table = read_table(tmp_path, name)
        assert table.schema.names == schema.names, name
        assert [f.nullable for f in table.schema] == [f.nullable for f in schema], name
        assert table.num_rows > 0, name


def test_fulltext_dedup_shares_elements_across_versions(tmp_path):
    generate(tmp_path)

    data = read_json(tmp_path / "2310-12345" / "hierarchy.json")
    v1, v2 = data["hierarchy"]["1"], data["hierarchy"]["2"]

    # Câu và công thức giống hệt dùng chung một id; câu bị sửa có id riêng.
    assert "2310-12345#s2" in v1 and "2310-12345#s2b" in v2
    assert "2310-12345#s1" in v1 and "2310-12345#s1" in v2
    # Cùng một element nhưng cha khác nhau theo version (root1 / root2).
    assert v1["2310-12345#h1"] == "2310-12345#root1"
    assert v2["2310-12345#h1"] == "2310-12345#root2"
    # root1, root2, h1, s1, f1, s2, s2b
    assert len(data["elements"]) == 7
    # Nút gốc không có cha nên không có dòng nào trong hierarchy.
    assert "2310-12345#root1" not in v1 and "2310-12345#root2" not in v2


def test_element_ids_start_with_publication_key(tmp_path):
    generate(tmp_path)

    elements = read_table(tmp_path, "elements").to_pylist()

    for row in elements:
        prefix = row["element_id"].split("#")[0]
        assert prefix in {"2310-12345", "2401-00001", "hep-th_9901001"}


def test_edge_cases_are_present(tmp_path):
    generate(tmp_path)

    assert (tmp_path / "hep-th_9901001" / "v1" / "intro.tex").exists()  # ID cũ + \input
    pdf_only = tmp_path / "2402-99999"
    assert (pdf_only / "metadata.json").exists()  # record rỗng vẫn có metadata
    assert (pdf_only / "v1").is_dir()  # version rỗng được ghi nhận, không bị bỏ qua
    assert not list((pdf_only / "v1").iterdir())  # nhưng không có .tex/.bib
    refs = read_json(tmp_path / "2401-00001" / "references.json")
    assert refs["1301-3781"]["ref_title"] is None  # ref thiếu tiêu đề


def test_predictions_only_use_references_of_the_same_paper(tmp_path):
    generate(tmp_path)

    pool: dict[str, set[str]] = {}
    for row in read_table(tmp_path, "references").to_pylist():
        pool.setdefault(row["arxiv_id"], set()).add(row["ref_arxiv_id"])

    for row in read_table(tmp_path, "predictions").to_pylist():
        ids = {row["groundtruth_arxiv_id"], *(row[f"cand_{k}"] for k in range(1, 6))}
        ids.discard(None)
        # Đề M3: một cặp chỉ tồn tại bên trong một bài, nên đáp án và ứng viên phải nằm
        # trong tập reference của chính bài đó.
        assert ids <= pool[row["arxiv_id"]], row


def test_every_bibtex_entry_of_a_matched_paper_has_a_prediction(tmp_path):
    generate(tmp_path)

    bib = {(r["arxiv_id"], r["citation_key"]) for r in read_table(tmp_path, "bibtex").to_pylist()}
    pred = {
        (r["arxiv_id"], r["citation_key"])
        for r in read_table(tmp_path, "predictions").to_pylist()
    }

    # Đề M3: predictions có một dòng cho mỗi BibTeX entry của mỗi bài tham gia matching.
    assert pred == bib