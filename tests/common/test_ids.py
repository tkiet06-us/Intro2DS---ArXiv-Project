"""Kiểm tra kiểm tra-ID và đổi qua lại giữa arxiv_id với tên folder."""

import pytest

from src.common.ids import from_folder_name, to_folder_name, validate_arxiv_id

GOOD_IDS = ["2310.12345", "1501.0001", "hep-th/9901001", "math.GT/0309136"]


@pytest.mark.parametrize("arxiv_id", GOOD_IDS)
def test_round_trip(arxiv_id):
    assert from_folder_name(to_folder_name(arxiv_id)) == arxiv_id


@pytest.mark.parametrize(
    ("arxiv_id", "folder"),
    [("2310.12345", "2310-12345"), ("hep-th/9901001", "hep-th_9901001")],
)
def test_folder_name_format(arxiv_id, folder):
    assert to_folder_name(arxiv_id) == folder


@pytest.mark.parametrize(
    "bad", ["2310-12345", "2310.12345v2", "", "../../etc/passwd", "2310.12345\n"]
)
def test_to_folder_name_rejects_bad_id(bad):
    # Chống lỗi lặng lẽ: dạng folder, có version, rỗng, path traversal, ký tự xuống dòng.
    with pytest.raises(ValueError):
        to_folder_name(bad)


@pytest.mark.parametrize("bad", ["2310.12345", "2310-12345v2", "", "a_b_c"])
def test_from_folder_name_rejects_bad_folder(bad):
    with pytest.raises(ValueError):
        from_folder_name(bad)


def test_validate_rejects_trailing_newline():
    with pytest.raises(ValueError):
        validate_arxiv_id("2310.12345\n")