"""Kiểm thử CLI runner ``src/main.py`` (P3-01): mock data, không gọi mạng.

Bao phủ: kiểm tra tham số, điều phối stage (``m1``/``m2``/``all``), ánh xạ ``--no_resume``,
mã thoát, import lười M2 khi chỉ chạy M1, và smoke ``--help`` qua subprocess.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src import main as runner
from src.m1_crawling import crawler, run_m1
from src.m2_parsing import run_m2

REPO_ROOT = Path(__file__).resolve().parents[1]
ID_START = "2506.00788"
ID_END = "2506.05787"


# ---------------------------------------------------------------------------
# Tiện ích dùng chung
# ---------------------------------------------------------------------------


def _fail_if_called(*args: object, **kwargs: object) -> int:
    """Hàm giả dùng để khẳng định một stage KHÔNG được gọi."""
    raise AssertionError("stage nay khong duoc goi")


def _kaggle(tmp_path: Path) -> Path:
    """Tạo file snapshot Kaggle rỗng để M1 vượt qua kiểm tra tham số."""
    path = tmp_path / "kaggle.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _config(tmp_path: Path, limit: int = 5) -> Path:
    """Ghi file cấu hình tối thiểu với ``pipeline.all_stage_limit`` cho trước."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"pipeline": {"all_stage_limit": limit}}), encoding="utf-8")
    return path


def _base_argv(tmp_path: Path, stage: str, limit: int = 5) -> list[str]:
    """Dựng argv hợp lệ tối thiểu cho runner (contact + kaggle + config)."""
    return [
        "--stage", stage,
        "--id_start", ID_START,
        "--id_end", ID_END,
        "--contact", "test@example.com",
        "--kaggle_json", str(_kaggle(tmp_path)),
        "--config", str(_config(tmp_path, limit)),
    ]


# ---------------------------------------------------------------------------
# Kiểm tra tham số / mã thoát
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["m1", "m2", "all"])
def test_missing_id_range_exits_bad_input(stage: str) -> None:
    """Thiếu ``--id_start``/``--id_end`` -> exit 1."""
    assert runner.main(["--stage", stage]) == runner.EXIT_BAD_INPUT


def test_invalid_stage_exits_bad_input() -> None:
    """``--stage`` ngoài {m1, m2, all} -> exit 1."""
    argv = ["--stage", "m3", "--id_start", ID_START, "--id_end", ID_END]
    assert runner.main(argv) == runner.EXIT_BAD_INPUT


def test_invalid_id_format_exits_bad_input() -> None:
    """ID sai định dạng -> exit 1."""
    argv = ["--stage", "m2", "--id_start", "khong-phai-id", "--id_end", ID_END]
    assert runner.main(argv) == runner.EXIT_BAD_INPUT


def test_inverted_range_exits_bad_input() -> None:
    """``id_start > id_end`` -> exit 1."""
    argv = ["--stage", "m2", "--id_start", ID_END, "--id_end", ID_START]
    assert runner.main(argv) == runner.EXIT_BAD_INPUT


def test_non_positive_limit_exits_bad_input() -> None:
    """``--limit <= 0`` -> exit 1."""
    argv = ["--stage", "m2", "--id_start", ID_START, "--id_end", ID_END, "--limit", "0"]
    assert runner.main(argv) == runner.EXIT_BAD_INPUT


@pytest.mark.parametrize(
    ("code", "expected"),
    [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (130, 130)],
)
def test_normalize_maps_to_runner_contract(code: int, expected: int) -> None:
    """Mã thoát của stage được chuẩn hóa về 0/1/130."""
    assert runner._normalize(code) == expected


def test_all_stage_limit_defaults_when_config_missing(tmp_path: Path) -> None:
    """Thiếu file cấu hình -> dùng giới hạn mặc định."""
    missing = tmp_path / "khong-ton-tai.yaml"
    assert runner._all_stage_limit(missing) == runner.DEFAULT_ALL_STAGE_LIMIT


def test_all_stage_limit_falls_back_on_bad_value(tmp_path: Path) -> None:
    """Giá trị ``all_stage_limit`` không hợp lệ -> dùng mặc định."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"pipeline": {"all_stage_limit": "abc"}}), encoding="utf-8")
    assert runner._all_stage_limit(path) == runner.DEFAULT_ALL_STAGE_LIMIT


# ---------------------------------------------------------------------------
# Điều phối stage
# ---------------------------------------------------------------------------


def test_stage_m1_dispatches_only_m1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--stage m1`` chỉ gọi M1, không đụng tới M2."""
    captured: dict[str, object] = {}

    def fake_pipeline(args: object, contact: str, *extra: object) -> int:
        captured["args"] = args
        captured["contact"] = contact
        return run_m1.EXIT_OK

    monkeypatch.setattr(run_m1, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(run_m2, "run_stage", _fail_if_called)

    assert runner.main(_base_argv(tmp_path, "m1")) == runner.EXIT_OK
    assert captured["args"].name == runner.DEFAULT_NAME  # type: ignore[attr-defined]
    assert captured["contact"] == "test@example.com"


def test_stage_m2_dispatches_only_m2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--stage m2`` chỉ gọi M2, không đụng tới M1."""
    captured: dict[str, object] = {}

    def fake_stage(args: object) -> int:
        captured["args"] = args
        return run_m2.EXIT_OK

    monkeypatch.setattr(run_m2, "run_stage", fake_stage)
    monkeypatch.setattr(run_m1, "run_pipeline", _fail_if_called)

    assert runner.main(_base_argv(tmp_path, "m2")) == runner.EXIT_OK
    assert captured["args"].id_start == ID_START  # type: ignore[attr-defined]


def test_stage_m2_accepts_range_across_months(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dải qua 2 tháng (Nhi: 2502.18235 -> 2503.01913) phải hợp lệ."""
    captured: dict[str, object] = {}

    def fake_stage(args: object) -> int:
        captured["args"] = args
        return run_m2.EXIT_OK

    monkeypatch.setattr(run_m2, "run_stage", fake_stage)
    argv = [
        "--stage", "m2",
        "--id_start", "2502.18235",
        "--id_end", "2503.01913",
        "--config", str(_config(tmp_path)),
    ]
    assert runner.main(argv) == runner.EXIT_OK
    assert captured["args"].id_start == "2502.18235"  # type: ignore[attr-defined]


def test_stage_all_runs_m1_then_m2_with_limit_and_sample_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``--stage all`` chạy M1 rồi M2, giới hạn N bài, tên ``_sample``, có warning."""
    order: list[str] = []
    seen: dict[str, object] = {}

    def fake_pipeline(args: object, contact: str, *extra: object) -> int:
        order.append("m1")
        seen["m1_name"] = args.name
        seen["m1_limit"] = args.limit
        return run_m1.EXIT_OK

    def fake_stage(args: object) -> int:
        order.append("m2")
        seen["m2_name"] = args.name
        seen["m2_limit"] = args.limit
        return run_m2.EXIT_OK

    monkeypatch.setattr(run_m1, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(run_m2, "run_stage", fake_stage)

    with caplog.at_level(logging.WARNING, logger="src.main"):
        code = runner.main(_base_argv(tmp_path, "all", limit=3))

    assert code == runner.EXIT_OK
    assert order == ["m1", "m2"]
    # Tên có hậu tố ``_sample`` -> không ghi đè summary benchmark của chế độ thật.
    assert seen["m1_name"] == f"{runner.DEFAULT_NAME}_sample"
    assert seen["m2_name"] == f"{runner.DEFAULT_NAME}_sample"
    assert seen["m1_limit"] == 3
    assert seen["m2_limit"] == 3
    assert any("chay thu" in record.getMessage() for record in caplog.records)


def test_custom_name_propagates_with_sample_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--name kien`` -> ``kien_sample`` ở chế độ ``all``."""
    seen: dict[str, object] = {}

    def fake_pipeline(args: object, contact: str, *extra: object) -> int:
        seen["m1_name"] = args.name
        return run_m1.EXIT_OK

    def fake_stage(args: object) -> int:
        seen["m2_name"] = args.name
        return run_m2.EXIT_OK

    monkeypatch.setattr(run_m1, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(run_m2, "run_stage", fake_stage)

    argv = _base_argv(tmp_path, "all", limit=2) + ["--name", "kien"]
    assert runner.main(argv) == runner.EXIT_OK
    assert seen["m1_name"] == "kien_sample"
    assert seen["m2_name"] == "kien_sample"


@pytest.mark.parametrize("no_resume", [False, True])
def test_resume_flags_map_to_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_resume: bool
) -> None:
    """``--no_resume`` -> ``fresh_*`` cho M1 và ``no_resume`` cho M2 (mặc định thì ngược lại)."""
    seen: dict[str, object] = {}

    def fake_pipeline(args: object, contact: str, *extra: object) -> int:
        seen["m1"] = args
        return run_m1.EXIT_OK

    def fake_stage(args: object) -> int:
        seen["m2"] = args
        return run_m2.EXIT_OK

    monkeypatch.setattr(run_m1, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(run_m2, "run_stage", fake_stage)

    argv = _base_argv(tmp_path, "all", limit=1)
    if no_resume:
        argv.append("--no_resume")
    assert runner.main(argv) == runner.EXIT_OK

    m1_args = seen["m1"]
    m2_args = seen["m2"]
    assert m1_args.fresh_discovery is no_resume  # type: ignore[attr-defined]
    assert m1_args.fresh_crawl is no_resume  # type: ignore[attr-defined]
    assert m1_args.fresh_s2 is no_resume  # type: ignore[attr-defined]
    assert m2_args.no_resume is no_resume  # type: ignore[attr-defined]


def test_no_worker_propagates_to_m2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--no_worker`` được chuyển tiếp xuống M2."""
    seen: dict[str, object] = {}

    def fake_stage(args: object) -> int:
        seen["m2"] = args
        return run_m2.EXIT_OK

    monkeypatch.setattr(run_m2, "run_stage", fake_stage)

    argv = _base_argv(tmp_path, "m2") + ["--no_worker"]
    assert runner.main(argv) == runner.EXIT_OK
    assert seen["m2"].no_worker is True  # type: ignore[attr-defined]


def test_all_stops_when_m1_fails_hard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """M1 lỗi nặng (mã >= 2) -> dừng, không chạy M2, exit 1."""
    monkeypatch.setattr(run_m1, "run_pipeline", lambda *a, **k: run_m1.EXIT_DISCOVERY)
    monkeypatch.setattr(run_m2, "run_stage", _fail_if_called)

    assert runner.main(_base_argv(tmp_path, "all", limit=2)) == runner.EXIT_BAD_INPUT


def test_all_continues_after_m1_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """M1 xong một phần (mã 1) vẫn chạy tiếp M2."""
    calls: list[str] = []

    def fake_stage(args: object) -> int:
        calls.append("m2")
        return run_m2.EXIT_OK

    monkeypatch.setattr(run_m1, "run_pipeline", lambda *a, **k: run_m1.EXIT_PARTIAL)
    monkeypatch.setattr(run_m2, "run_stage", fake_stage)

    assert runner.main(_base_argv(tmp_path, "all", limit=2)) == runner.EXIT_BAD_INPUT
    assert calls == ["m2"]


def test_all_returns_130_when_m2_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """M2 bị ngắt -> runner trả 130."""
    monkeypatch.setattr(run_m1, "run_pipeline", lambda *a, **k: run_m1.EXIT_OK)
    monkeypatch.setattr(run_m2, "run_stage", lambda args: run_m2.EXIT_INTERRUPTED)

    assert runner.main(_base_argv(tmp_path, "all", limit=2)) == runner.EXIT_INTERRUPTED


# ---------------------------------------------------------------------------
# Xử lý lỗi
# ---------------------------------------------------------------------------


def test_stage_exception_returns_bad_input_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Stage ném lỗi không mong đợi -> exit 1 và có log."""
    monkeypatch.setattr(run_m1, "run_pipeline", _fail_if_called)

    with caplog.at_level(logging.ERROR, logger="src.main"):
        code = runner.main(_base_argv(tmp_path, "m1"))

    assert code == runner.EXIT_BAD_INPUT
    assert any("khong mong doi" in record.getMessage() for record in caplog.records)


def test_keyboard_interrupt_returns_130(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ctrl+C -> exit 130."""
    monkeypatch.setattr(run_m1, "run_pipeline", _raise_keyboard_interrupt)

    assert runner.main(_base_argv(tmp_path, "m1")) == runner.EXIT_INTERRUPTED


def test_m1_missing_contact_returns_bad_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Thiếu ``--contact`` và biến môi trường -> M1 báo lỗi, runner trả 1."""
    monkeypatch.delenv(crawler.CONTACT_ENV, raising=False)
    monkeypatch.setattr(run_m1, "run_pipeline", _fail_if_called)

    argv = [
        "--stage", "m1",
        "--id_start", ID_START,
        "--id_end", ID_END,
        "--kaggle_json", str(_kaggle(tmp_path)),
        "--config", str(_config(tmp_path)),
    ]
    assert runner.main(argv) == runner.EXIT_BAD_INPUT


def _raise_keyboard_interrupt(*args: object, **kwargs: object) -> int:
    """Hàm giả mô phỏng người dùng nhấn Ctrl+C giữa chừng."""
    raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Import lười + smoke
# ---------------------------------------------------------------------------


def test_stage_m1_does_not_import_m2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Chạy ``--stage m1`` không import ``src.m2_parsing.run_m2`` (giữ số đo RAM sạch)."""
    removed = sys.modules.pop("src.m2_parsing.run_m2", None)
    monkeypatch.setattr(run_m1, "run_pipeline", lambda *a, **k: run_m1.EXIT_OK)
    try:
        assert runner.main(_base_argv(tmp_path, "m1")) == runner.EXIT_OK
        assert "src.m2_parsing.run_m2" not in sys.modules
    finally:
        if removed is not None:
            sys.modules["src.m2_parsing.run_m2"] = removed


def test_help_via_subprocess_exits_zero() -> None:
    """Smoke: ``python -m src.main --help`` thoát 0."""
    result = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--stage" in result.stdout


def test_skip_discovery_explicit_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--skip_discovery tường minh được chuyển xuống M1 và không cần kaggle_json."""
    seen: dict[str, object] = {}

    def fake_pipeline(args: object, contact: str, *extra: object) -> int:
        seen["m1"] = args
        return run_m1.EXIT_OK

    monkeypatch.setattr(run_m1, "run_pipeline", fake_pipeline)

    argv = [
        "--stage", "m1",
        "--id_start", ID_START,
        "--id_end", ID_END,
        "--contact", "test@example.com",
        "--config", str(_config(tmp_path)),
        "--skip_discovery",
    ]
    assert runner.main(argv) == runner.EXIT_OK
    assert seen["m1"].skip_discovery is True  # type: ignore[attr-defined]


def test_auto_detect_existing_discovery_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tự động nhận diện file discovery_<name>.jsonl đã có -> skip_discovery=True."""
    seen: dict[str, object] = {}

    def fake_pipeline(args: object, contact: str, *extra: object) -> int:
        seen["m1"] = args
        return run_m1.EXIT_OK

    monkeypatch.setattr(run_m1, "run_pipeline", fake_pipeline)

    data_root = tmp_path / "raw"
    data_root.mkdir()
    discovery_file = data_root / "discovery_kien.jsonl"
    discovery_file.write_text('{"arxiv_id": "2506.00788"}\n', encoding="utf-8")

    argv = [
        "--stage", "m1",
        "--name", "kien",
        "--id_start", ID_START,
        "--id_end", ID_END,
        "--contact", "test@example.com",
        "--config", str(_config(tmp_path)),
        "--data_root", str(data_root),
    ]
    assert runner.main(argv) == runner.EXIT_OK
    assert seen["m1"].skip_discovery is True  # type: ignore[attr-defined]


def test_auto_detect_shared_25k_discovery_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tự động nhận diện discovery_25k.jsonl -> copy sang discovery_<name>.jsonl & skip_discovery=True."""
    seen: dict[str, object] = {}

    def fake_pipeline(args: object, contact: str, *extra: object) -> int:
        seen["m1"] = args
        return run_m1.EXIT_OK

    monkeypatch.setattr(run_m1, "run_pipeline", fake_pipeline)

    data_root = tmp_path / "raw"
    data_root.mkdir()
    shared_25k = data_root / "discovery_25k.jsonl"
    shared_25k.write_text('{"arxiv_id": "2506.00788"}\n', encoding="utf-8")

    argv = [
        "--stage", "m1",
        "--name", "bao",
        "--id_start", ID_START,
        "--id_end", ID_END,
        "--contact", "test@example.com",
        "--config", str(_config(tmp_path)),
        "--data_root", str(data_root),
    ]
    assert runner.main(argv) == runner.EXIT_OK
    assert seen["m1"].skip_discovery is True  # type: ignore[attr-defined]
    assert (data_root / "discovery_bao.jsonl").exists()
