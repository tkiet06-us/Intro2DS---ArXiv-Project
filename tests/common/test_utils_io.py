"""Kiểm tra ghi JSON an toàn."""

import pytest

from src.common.utils_io import read_json, write_json


def test_write_json_keeps_unicode_and_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "sub" / "data.json"

    write_json(target, {"author": "Nguyễn Văn A"})

    assert read_json(target) == {"author": "Nguyễn Văn A"}
    assert "Nguyễn" in target.read_text(encoding="utf-8")  # không bị escape thành \uXXXX
    assert not list(target.parent.glob("*.tmp"))


def test_write_json_overwrites_existing_file(tmp_path):
    target = tmp_path / "data.json"
    write_json(target, {"v": 1})

    write_json(target, {"v": 2})

    assert read_json(target) == {"v": 2}


def test_write_json_rejects_nan_and_keeps_old_file(tmp_path):
    target = tmp_path / "data.json"
    write_json(target, {"year": 2020})

    with pytest.raises(ValueError):
        write_json(target, {"year": float("nan")})

    # File đích vẫn là bản cũ nguyên vẹn và không bỏ lại file tạm.
    assert read_json(target) == {"year": 2020}
    assert not list(tmp_path.glob("*.tmp"))


def test_write_json_cleans_tmp_file_when_serialization_fails(tmp_path):
    target = tmp_path / "data.json"

    with pytest.raises(TypeError):
        write_json(target, {"bad": object()})  # object() không tuần tự hóa được

    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_json_accepts_str_path(tmp_path):
    write_json(str(tmp_path / "data.json"), {"ok": True})

    assert read_json(tmp_path / "data.json") == {"ok": True}