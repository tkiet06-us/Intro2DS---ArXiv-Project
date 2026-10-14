"""Kiểm tra hàm đổi tên folder và ghi JSON an toàn."""

import pytest

from src.common.utils_io import from_folder_name, read_json, to_folder_name, write_json


@pytest.mark.parametrize(
    ("arxiv_id", "folder"),
    [("2310.12345", "2310-12345"), ("hep-th/9901001", "hep-th_9901001")],
)
def test_folder_name_round_trip(arxiv_id, folder):
    assert to_folder_name(arxiv_id) == folder
    assert from_folder_name(folder) == arxiv_id


def test_write_json_keeps_unicode_and_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "sub" / "data.json"

    write_json(target, {"author": "Nguyễn Văn A"})

    assert read_json(target) == {"author": "Nguyễn Văn A"}
    assert "Nguyễn" in target.read_text(encoding="utf-8")  # không bị escape thành \uXXXX
    assert not list(target.parent.glob("*.tmp"))