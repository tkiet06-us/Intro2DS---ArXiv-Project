"""Kiểm tra schema Pydantic khớp schema Parquet và các ràng buộc dữ liệu."""

import pytest
from pydantic import ValidationError

from src.schemas import MODELS, PARQUET_SCHEMAS, Paper, Prediction, validate_arxiv_id

CAND_FIELDS = ["groundtruth_arxiv_id", "cand_1", "cand_2", "cand_3", "cand_4", "cand_5"]


def make_prediction(**overrides) -> Prediction:
    """Tạo Prediction hợp lệ tối thiểu, ghi đè các trường cần thử."""
    return Prediction(
        **{"arxiv_id": "2310.12345", "partition": "train", "citation_key": "foo", **overrides}
    )


def test_model_fields_match_parquet_columns():
    # Bắt lỗi "Pydantic một đằng, Parquet một nẻo": cùng tên và cùng THỨ TỰ cột.
    for name, model in MODELS.items():
        assert list(model.model_fields) == PARQUET_SCHEMAS[name].names, name


def test_non_null_parquet_columns_are_required_in_model():
    # Cột non-null trong Parquet mà model cho phép None/thiếu thì lúc ghi file mới nổ lỗi.
    for name, model in MODELS.items():
        for column in PARQUET_SCHEMAS[name]:
            if not column.nullable:
                assert model.model_fields[column.name].is_required(), (name, column.name)


def test_authors_must_be_list():
    # Chuỗi "Alice, Bob" không được lọt vào cột list<string>.
    with pytest.raises(ValidationError):
        Paper(arxiv_id="2310.12345", authors="Alice, Bob")


def test_partition_rejects_unknown_value():
    with pytest.raises(ValidationError):
        Prediction(arxiv_id="2310.12345", partition="dev", citation_key="foo")


def test_extra_field_is_forbidden():
    with pytest.raises(ValidationError):
        Paper(arxiv_id="2310.12345", typo_field="x")


@pytest.mark.parametrize(
    "arxiv_id", ["2310.12345", "1501.0001", "hep-th/9901001", "math.GT/0309136"]
)
def test_valid_arxiv_ids_accepted(arxiv_id):
    assert validate_arxiv_id(arxiv_id) == arxiv_id


@pytest.mark.parametrize(
    "arxiv_id", ["2310-12345", "2310.12345v2", "hep-th_9901001", "", "2310.12345\n"]
)
def test_invalid_arxiv_ids_rejected(arxiv_id):
    # Dạng folder, có hậu tố version, rỗng, hoặc có ký tự xuống dòng đều phải bị chặn.
    with pytest.raises(ValueError):
        validate_arxiv_id(arxiv_id)


def test_model_validates_arxiv_id():
    with pytest.raises(ValidationError):
        Paper(arxiv_id="2310-12345")


# ---- Prediction: các trường dùng để tính MRR phải đúng dạng ID chuẩn ----


@pytest.mark.parametrize("field", CAND_FIELDS)
@pytest.mark.parametrize("bad", ["1810-04805", "1810.04805v2", " 1810.04805"])
def test_prediction_rejects_malformed_candidate_ids(field, bad):
    # Dạng folder (key của references.json quên đổi), có version, có khoảng trắng.
    overrides = {"cand_1": "1706.03762"} if field != "cand_1" else {}
    with pytest.raises(ValidationError):
        make_prediction(**{**overrides, field: bad})


def test_prediction_accepts_old_style_ids():
    prediction = make_prediction(
        groundtruth_arxiv_id="hep-th/9503124", cand_1="hep-th/9503124", cand_2="1810.04805"
    )

    assert prediction.cand_1 == "hep-th/9503124"


def test_prediction_accepts_empty_and_partial_candidates():
    assert make_prediction().cand_1 is None  # chưa có dự đoán nào
    assert make_prediction(cand_1="1706.03762", cand_2="1810.04805").cand_3 is None


def test_prediction_accepts_full_top5_and_unknown_groundtruth():
    ids = ["1706.03762", "1810.04805", "1301.3781", "1409.0473", "1512.03385"]

    prediction = make_prediction(**{f"cand_{rank}": ids[rank - 1] for rank in range(1, 6)})

    assert prediction.groundtruth_arxiv_id is None
    assert prediction.cand_5 == "1512.03385"


def test_prediction_rejects_null_gap_in_candidates():
    # cand_2 null nhưng cand_3 có giá trị: danh sách xếp hạng bị "lỗ hổng".
    with pytest.raises(ValidationError, match="liền mạch"):
        make_prediction(cand_1="1706.03762", cand_3="1810.04805")


def test_prediction_rejects_duplicate_candidates():
    with pytest.raises(ValidationError, match="trùng"):
        make_prediction(cand_1="1706.03762", cand_2="1706.03762")