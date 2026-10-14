"""Kiểm tra mock generator sinh đủ 7 bảng đúng schema và đủ các tình huống khó."""

import pyarrow.parquet as pq

from src.common.utils_io import read_json
from src.schemas import PARQUET_SCHEMAS
from tests.make_mock import PARQUET_FILE_NAME, generate


def test_generate_writes_all_seven_tables_with_expected_schema(tmp_path):
    generate(tmp_path)

    assert len(PARQUET_SCHEMAS) == 7
    for name, schema in PARQUET_SCHEMAS.items():
        table = pq.read_table(tmp_path / "parquet" / name / PARQUET_FILE_NAME)
        assert table.schema.names == schema.names, name
        assert [f.nullable for f in table.schema] == [f.nullable for f in schema], name
        assert table.num_rows > 0, name


def test_fulltext_dedup_shares_elements_across_versions(tmp_path):
    generate(tmp_path)

    hierarchy = read_json(tmp_path / "2310-12345" / "hierarchy.json")

    # Câu và công thức giống hệt dùng chung một id; câu bị sửa có id riêng.
    assert "2310.12345#s2" in hierarchy["hierarchy"]["1"]
    assert "2310.12345#s2b" in hierarchy["hierarchy"]["2"]
    assert "2310.12345#s1" in hierarchy["hierarchy"]["1"]
    assert "2310.12345#s1" in hierarchy["hierarchy"]["2"]
    assert len(hierarchy["elements"]) == 6  # doc, sec1, s1, f1, s2, s2b
    assert "2310.12345#doc" not in hierarchy["hierarchy"]["1"]  # gốc không có cha


def test_edge_cases_are_present(tmp_path):
    generate(tmp_path)

    assert (tmp_path / "hep-th_9901001" / "v1" / "intro.tex").exists()  # ID cũ + \input
    pdf_only = tmp_path / "2402-99999"
    assert (pdf_only / "metadata.json").exists()  # record rỗng vẫn có metadata
    assert not (pdf_only / "v1").exists()  # nhưng không có mã nguồn
    refs = read_json(tmp_path / "2401-00001" / "references.json")
    assert refs["1301-3781"]["ref_title"] is None  # ref thiếu tiêu đề