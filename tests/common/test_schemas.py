"""Kiểm tra schema Pydantic khớp schema Parquet và các ràng buộc dữ liệu."""

import pytest
from pydantic import ValidationError

from src.schemas import MODELS, PARQUET_SCHEMAS, Paper, Prediction, validate_arxiv_id


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


@pytest.mark.parametrize("arxiv_id", ["2310.12345", "1501.0001", "hep-th/9901001", "math.GT/0309136"])
def test_valid_arxiv_ids_accepted(arxiv_id):
    assert validate_arxiv_id(arxiv_id) == arxiv_id


@pytest.mark.parametrize("arxiv_id", ["2310-12345", "2310.12345v2", "hep-th_9901001", ""])
def test_invalid_arxiv_ids_rejected(arxiv_id):
    # Dạng folder, có hậu tố version, hoặc rỗng đều phải bị chặn.
    with pytest.raises(ValueError):
        validate_arxiv_id(arxiv_id)


def test_model_validates_arxiv_id():
    with pytest.raises(ValidationError):
        Paper(arxiv_id="2310-12345")