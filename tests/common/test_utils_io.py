"""Kiểm tra ghi JSON an toàn."""

from src.common.utils_io import read_json, write_json


def test_write_json_keeps_unicode_and_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "sub" / "data.json"

    write_json(target, {"author": "Nguyễn Văn A"})

    assert read_json(target) == {"author": "Nguyễn Văn A"}
    assert "Nguyễn" in target.read_text(encoding="utf-8")  # không bị escape thành \uXXXX
    assert not list(target.parent.glob("*.tmp"))