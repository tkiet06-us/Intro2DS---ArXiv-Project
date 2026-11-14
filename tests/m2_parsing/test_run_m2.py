"""Kiểm thử cho stage M2 (``src/m2_parsing/run_m2.py``): mock data, không gọi mạng.

Bao phủ: xử lý một bài, dải ID (cùng tháng & qua tháng), checkpoint/resume, ``--no_resume``,
bài lỗi/rỗng/lớn, thứ tự Cleanup -> Dedup, worker timeout + dựng lại, và summary_stats_m2.
"""

import logging
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.common.utils_io import append_jsonl, iter_jsonl, read_json
from src.m2_parsing import bibtex_extractor, dedup, latex_cleanup, run_m2, tree_builder

MOCK = Path(__file__).resolve().parents[1] / "mock_data"

# Ba bài ID mới (bài thứ tư hep-th/9901001 tách riêng vì ID cũ đứng khác chỗ khi sắp xếp).
NEW_ID_PAPERS = ["2310-12345", "2401-00001", "2402-99999"]


# ---------------------------------------------------------------------------
# Tiện ích dùng chung
# ---------------------------------------------------------------------------


def _copy_papers(root: Path, names: list[str]) -> Path:
    """Sao chép các bài mock vào ``root/papers_t`` và trả về thư mục đó."""
    papers = root / "papers_t"
    papers.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copytree(MOCK / name, papers / name)
    return papers


def _write_config(tmp_path: Path, m2: dict | None = None, pipeline: dict | None = None) -> Path:
    """Ghi file cấu hình tối thiểu (mặc định M2 vẫn lấy từ ``M2Settings``)."""
    path = tmp_path / "config.yaml"
    data = {"pipeline": pipeline or {"checkpoint_every": 1}, "m2": m2 or {}}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _args(tmp_path: Path, config: Path, **overrides) -> SimpleNamespace:
    """Dựng Namespace CLI giống ``run_m2.build_parser`` nhưng trỏ vào ``tmp_path``."""
    values = {
        "name": "t",
        "id_start": "2310.00001",
        "id_end": "2402.99999",
        "config": config,
        "data_root": tmp_path / "raw",
        "processed_root": tmp_path / "processed",
        "reports": tmp_path / "reports",
        "limit": None,
        "checkpoint_every": 1,
        "no_resume": False,
        "retry_failed": False,
        "no_worker": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _settings(**overrides) -> run_m2.M2Settings:
    """Cấu hình M2 chạy ngay trong tiến trình chính (trừ khi test cần worker thật)."""
    values = {"use_worker": False}
    values.update(overrides)
    return run_m2.M2Settings(**values)


def _spy(name: str, func, order: list[str]):
    """Bọc một hàm để ghi lại thứ tự gọi, vẫn gọi hàm gốc."""

    def wrapper(*args, **kwargs):
        order.append(name)
        return func(*args, **kwargs)

    return wrapper


def _hanging_worker(conn, _settings_dict) -> None:
    """Worker giả cho test timeout: ``arxiv_id`` chứa ``hang`` thì treo, còn lại trả ngay.

    Ở cấp module để tiến trình con (spawn) pickle được theo tên.
    """
    while True:
        task = conn.recv()
        if task is None:
            return
        _paper_dir, arxiv_id, _out_dir = task
        if "hang" in arxiv_id:
            time.sleep(30)
        conn.send({"arxiv_id": arxiv_id, "status": run_m2.STATUS_OK})


def _crashing_worker(conn, _settings_dict) -> None:
    """Worker giả cho test crash: ``arxiv_id`` chứa ``die`` (hoặc bài mock 2506.00099) thì ``os._exit(1)``.

    Ở cấp module để tiến trình con (spawn) pickle được theo tên.
    """
    while True:
        task = conn.recv()
        if task is None:
            return
        _paper_dir, arxiv_id, _out_dir = task
        if "die" in arxiv_id or arxiv_id == "2506.00099":
            os._exit(1)
        conn.send({"arxiv_id": arxiv_id, "status": run_m2.STATUS_OK})


# ---------------------------------------------------------------------------
# Liệt kê version & dải ID
# ---------------------------------------------------------------------------


def test_list_version_dirs_sorted(tmp_path):
    paper = tmp_path / "2310-12345"
    for name in ("v2", "v10", "v1", "notes"):
        (paper / name).mkdir(parents=True)
    assert [number for number, _ in run_m2.list_version_dirs(paper)] == [1, 2, 10]
    assert run_m2.list_version_dirs(tmp_path / "missing") == []


def test_list_version_dirs_case_and_leading_zero(tmp_path):
    paper = tmp_path / "p"
    for name in ("v1", "V2", "v01", "v0"):
        (paper / name).mkdir(parents=True)
    assert [number for number, _ in run_m2.list_version_dirs(paper)] == [1]


def test_id_range_key_orders_by_month_and_number():
    assert run_m2.id_range_key("2506.00788") == ("2506", 788)
    assert run_m2.id_range_key("hep-th/9901001") == ("hep-th", 9901001)
    assert run_m2.id_range_key("2506.00787") < run_m2.id_range_key("2506.00788")
    # Dải của Nhi qua hai tháng phải so đúng thứ tự (không so chuỗi thô).
    assert run_m2.id_range_key("2502.18235") < run_m2.id_range_key("2503.01913")
    with pytest.raises(ValueError):
        run_m2.id_range_key("khong-phai-id")


def test_list_papers_in_range_same_month(tmp_path):
    root = tmp_path / "papers_t"
    for name in ("2506-00787", "2506-00788", "2506-02000", "2506-05787", "2506-05788",
                 "2505-99999", "not_an_id"):
        (root / name).mkdir(parents=True)
    found = run_m2.list_papers_in_range(root, "2506.00788", "2506.05787")
    assert [arxiv_id for arxiv_id, _ in found] == ["2506.00788", "2506.02000", "2506.05787"]


def test_list_papers_in_range_cross_month(tmp_path):
    root = tmp_path / "papers_t"
    for name in ("2502-18234", "2502-18235", "2502-18236", "2503-01912", "2503-01913",
                 "2503-01914"):
        (root / name).mkdir(parents=True)
    found = run_m2.list_papers_in_range(root, "2502.18235", "2503.01913")
    assert [arxiv_id for arxiv_id, _ in found] == [
        "2502.18235", "2502.18236", "2503.01912", "2503.01913",
    ]


def test_list_papers_in_range_limit_and_missing_root(tmp_path):
    root = tmp_path / "papers_t"
    for name in ("2506-00788", "2506-02000", "2506-05787"):
        (root / name).mkdir(parents=True)
    found = run_m2.list_papers_in_range(root, "2506.00788", "2506.05787", limit=2)
    assert [arxiv_id for arxiv_id, _ in found] == ["2506.00788", "2506.02000"]
    assert run_m2.list_papers_in_range(tmp_path / "khong-co", "2506.00788", "2506.05787") == []


def test_list_papers_in_range_mixed_prefix_types_rejected(tmp_path):
    """Từ chối dải trộn giữa ID cũ (archive/YYMMNNN) và ID mới (YYMM.NNNNN)."""
    root = tmp_path / "papers_t"
    for name in ("hep-th_9901001", "2506-00788"):
        (root / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="khong cung dang ID"):
        run_m2.list_papers_in_range(root, "hep-th/9901001", "2506.00788")


def test_list_papers_in_range_mixed_folders_filtered(tmp_path):
    """Khi thư mục chứa lẫn ID cũ và mới, chỉ chọn bài đúng dạng ID và nằm trong dải."""
    root = tmp_path / "papers_t"
    for name in ("hep-th_9901001", "2506-00788", "2506-00789"):
        (root / name).mkdir(parents=True)
    found = run_m2.list_papers_in_range(root, "2506.00788", "2506.00789")
    assert [arxiv_id for arxiv_id, _ in found] == ["2506.00788", "2506.00789"]


def test_validate_id_range_rejects_mixed_or_different_archives():
    with pytest.raises(ValueError, match="khong cung dang ID"):
        run_m2.validate_id_range("hep-th/9901001", "2506.00788")
    with pytest.raises(ValueError, match="khac archive"):
        run_m2.validate_id_range("hep-th/9901001", "math.GT/0309136")
    with pytest.raises(ValueError, match="phai <="):
        run_m2.validate_id_range("2506.00788", "2506.00787")
    # Cùng dạng và đúng thứ tự -> không raise
    run_m2.validate_id_range("hep-th/9901001", "hep-th/9901050")
    run_m2.validate_id_range("2506.00788", "2506.05787")


def test_validate_rejects_bad_range_and_limit():
    parser = run_m2.build_parser()
    assert run_m2._validate(parser.parse_args(["--id_start", "2506.00788", "--id_end", "2506.00787"]))
    assert run_m2._validate(parser.parse_args(["--id_start", "bad", "--id_end", "2506.00787"]))
    assert run_m2._validate(
        parser.parse_args(["--id_start", "hep-th/9901001", "--id_end", "2506.00788"])
    )
    assert run_m2._validate(
        parser.parse_args(["--id_start", "2506.00788", "--id_end", "2506.05787", "--limit", "0"])
    )
    ok = parser.parse_args(["--id_start", "2502.18235", "--id_end", "2503.01913"])
    assert run_m2._validate(ok) is None


# ---------------------------------------------------------------------------
# Xử lý một bài
# ---------------------------------------------------------------------------


def test_process_paper_writes_hierarchy_and_bib(tmp_path):
    papers = _copy_papers(tmp_path / "raw", ["2310-12345"])
    out = tmp_path / "processed"
    result = run_m2.process_paper_m2(papers / "2310-12345", "2310.12345", _settings(), out)
    assert result.status == run_m2.STATUS_OK
    assert (result.n_versions, result.n_versions_done) == (2, 2)
    folder = out / "2310-12345"
    payload = read_json(folder / "hierarchy.json")
    assert payload["elements"] and payload["hierarchy"]
    assert (folder / "refs.bib").exists()
    assert result.nodes_by_kind["document"] == 2


def test_process_paper_empty_record_when_no_version(tmp_path):
    papers = _copy_papers(tmp_path / "raw", ["2402-99999"])
    result = run_m2.process_paper_m2(
        papers / "2402-99999", "2402.99999", _settings(), tmp_path / "out"
    )
    assert result.status == run_m2.STATUS_EMPTY
    assert result.n_versions == 0


def test_process_paper_old_id_follows_input(tmp_path):
    papers = _copy_papers(tmp_path / "raw", ["hep-th_9901001"])
    out = tmp_path / "out"
    result = run_m2.process_paper_m2(papers / "hep-th_9901001", "hep-th/9901001", _settings(), out)
    assert result.status == run_m2.STATUS_OK
    payload = read_json(out / "hep-th_9901001" / "hierarchy.json")
    assert any("Strings are fun" in text for text in payload["elements"].values())


def test_process_paper_without_metadata_still_ok(tmp_path):
    papers = _copy_papers(tmp_path / "raw", ["2310-12345"])
    (papers / "2310-12345" / "metadata.json").unlink()
    result = run_m2.process_paper_m2(
        papers / "2310-12345", "2310.12345", _settings(), tmp_path / "out"
    )
    assert result.status == run_m2.STATUS_OK


def test_process_paper_include_cycle_terminates(tmp_path):
    paper = tmp_path / "raw" / "papers_t" / "2506-00001"
    (paper / "v1").mkdir(parents=True)
    (paper / "v1" / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nCYCLE_MARK\n\\input{main}\n\\end{document}\n",
        encoding="utf-8",
    )
    result = run_m2.process_paper_m2(paper, "2506.00001", _settings(), tmp_path / "out")
    assert result.status == run_m2.STATUS_OK


def test_process_paper_degraded_when_large(tmp_path):
    papers = _copy_papers(tmp_path / "raw", ["2310-12345"])
    settings = _settings(max_tex_bytes=10, max_tex_bytes_skip=10_000_000)
    result = run_m2.process_paper_m2(papers / "2310-12345", "2310.12345", settings, tmp_path / "out")
    assert result.status == run_m2.STATUS_OK
    assert result.degraded is True
    assert result.n_duplicate_nodes_merged == 0  # bỏ dedup_fulltext ở chế độ nhẹ


def test_process_paper_skipped_when_too_large(tmp_path):
    papers = _copy_papers(tmp_path / "raw", ["2310-12345"])
    settings = _settings(max_tex_bytes=5, max_tex_bytes_skip=10)
    result = run_m2.process_paper_m2(papers / "2310-12345", "2310.12345", settings, tmp_path / "out")
    assert result.status == run_m2.STATUS_SKIPPED_LARGE


def test_process_paper_failed_does_not_write_outputs(tmp_path, monkeypatch):
    papers = _copy_papers(tmp_path / "raw", ["2310-12345"])
    paper = papers / "2310-12345"
    out = tmp_path / "out"

    # Lần đầu thành công: có output hợp lệ.
    assert run_m2.process_paper_m2(paper, "2310.12345", _settings(), out).status == run_m2.STATUS_OK
    folder = out / "2310-12345"
    assert (folder / "hierarchy.json").exists()

    # Lần sau lỗi: không được ghi hierarchy rỗng và phải dọn output cũ.
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tree_builder, "build_tree", boom)
    failed = run_m2.process_paper_m2(paper, "2310.12345", _settings(), out)
    assert failed.status == run_m2.STATUS_FAILED
    assert not (folder / "hierarchy.json").exists()
    assert not (folder / "refs.bib").exists()


def test_process_paper_empty_record_writes_nothing(tmp_path):
    papers = _copy_papers(tmp_path / "raw", ["2402-99999"])
    out = tmp_path / "out"
    result = run_m2.process_paper_m2(papers / "2402-99999", "2402.99999", _settings(), out)
    assert result.status == run_m2.STATUS_EMPTY
    assert not (out / "2402-99999" / "hierarchy.json").exists()


def test_process_paper_truncates_tree_over_max_nodes(tmp_path):
    papers = _copy_papers(tmp_path / "raw", ["2310-12345"])
    result = run_m2.process_paper_m2(
        papers / "2310-12345", "2310.12345", _settings(max_nodes=1), tmp_path / "out"
    )
    assert result.status == run_m2.STATUS_OK
    assert result.truncated is True


def test_cleanup_runs_before_dedup(tmp_path, monkeypatch):
    papers = _copy_papers(tmp_path / "raw", ["2310-12345"])
    order: list[str] = []
    monkeypatch.setattr(
        bibtex_extractor, "extract_bibtex", _spy("bibtex", bibtex_extractor.extract_bibtex, order)
    )
    monkeypatch.setattr(
        latex_cleanup, "cleanup_latex", _spy("cleanup", latex_cleanup.cleanup_latex, order)
    )
    monkeypatch.setattr(
        latex_cleanup, "normalize_math", _spy("normalize", latex_cleanup.normalize_math, order)
    )
    monkeypatch.setattr(
        tree_builder, "build_tree", _spy("tree", tree_builder.build_tree, order)
    )
    monkeypatch.setattr(dedup, "dedup_bibtex", _spy("dedup_bibtex", dedup.dedup_bibtex, order))
    monkeypatch.setattr(dedup, "dedup_fulltext", _spy("dedup_fulltext", dedup.dedup_fulltext, order))
    run_m2.process_paper_m2(papers / "2310-12345", "2310.12345", _settings(), tmp_path / "out")
    assert order.index("bibtex") < order.index("cleanup")
    assert order.index("tree") < order.index("dedup_bibtex") < order.index("dedup_fulltext")


# ---------------------------------------------------------------------------
# Vòng lặp dải + checkpoint/resume
# ---------------------------------------------------------------------------


def test_run_stage_writes_outputs_and_summary(tmp_path):
    _copy_papers(tmp_path / "raw", NEW_ID_PAPERS)
    assert run_m2.run_stage(_args(tmp_path, _write_config(tmp_path))) == run_m2.EXIT_OK
    processed = tmp_path / "processed" / "papers_t"
    assert (processed / "2310-12345" / "hierarchy.json").exists()
    assert (processed / "2310-12345" / "refs.bib").exists()
    assert (tmp_path / "reports" / "checkpoint_m2_t.jsonl").exists()
    summary = read_json(tmp_path / "reports" / "summary_stats_m2_t.json")
    for key in ("wall_seconds", "n_papers", "avg_seconds_per_paper", "peak_ram_mb", "n_ram_samples"):
        assert key in summary
    assert "n_refs_total" not in summary
    assert "n_refs_co_arxiv_id_total" not in summary
    assert (summary["n_papers_total"], summary["n_ok"], summary["n_empty_record"]) == (3, 2, 1)
    assert summary["nodes_by_kind"]["document"] == 3
    assert summary["is_resumed_run"] is False


def test_one_failing_paper_does_not_stop_range(tmp_path, monkeypatch, caplog):
    _copy_papers(tmp_path / "raw", NEW_ID_PAPERS)
    real_build_tree = tree_builder.build_tree

    def boom(latex, version=1):
        if "Our method is simple" in latex:
            raise RuntimeError("boom")
        return real_build_tree(latex, version)

    monkeypatch.setattr(tree_builder, "build_tree", boom)
    with caplog.at_level(logging.ERROR):
        code = run_m2.run_stage(_args(tmp_path, _write_config(tmp_path)))
    assert code == run_m2.EXIT_OK
    summary = read_json(tmp_path / "reports" / "summary_stats_m2_t.json")
    assert (summary["n_failed"], summary["n_ok"], summary["n_empty_record"]) == (1, 1, 1)
    assert summary["failures"][0]["arxiv_id"] == "2401.00001"
    assert any("2401.00001" in record.getMessage() for record in caplog.records)


def test_resume_skips_done_papers_and_merges_summary(tmp_path, monkeypatch):
    _copy_papers(tmp_path / "raw", NEW_ID_PAPERS)
    args = _args(tmp_path, _write_config(tmp_path))
    calls: list[str] = []
    real_process = run_m2.process_paper_m2

    def counting(*call_args, **call_kwargs):
        calls.append(call_args[1])
        return real_process(*call_args, **call_kwargs)

    monkeypatch.setattr(run_m2, "process_paper_m2", counting)
    assert run_m2.run_stage(args) == run_m2.EXIT_OK
    assert len(calls) == 3
    assert run_m2.run_stage(args) == run_m2.EXIT_OK
    assert len(calls) == 3  # lần hai không xử lý lại bài nào
    summary = read_json(tmp_path / "reports" / "summary_stats_m2_t.json")
    assert summary["n_papers_resumed_skipped"] == 3
    assert summary["is_resumed_run"] is True
    assert summary["n_papers"] == 0  # Profiler chỉ đo lần chạy này
    assert summary["n_papers_total"] == 3  # nhưng số liệu gộp từ checkpoint
    assert summary["n_bib_entries_before_dedup"] == 4


def test_no_resume_reruns_everything(tmp_path, monkeypatch):
    _copy_papers(tmp_path / "raw", NEW_ID_PAPERS)
    args = _args(tmp_path, _write_config(tmp_path))
    calls: list[str] = []
    real_process = run_m2.process_paper_m2

    def counting(*call_args, **call_kwargs):
        calls.append(call_args[1])
        return real_process(*call_args, **call_kwargs)

    monkeypatch.setattr(run_m2, "process_paper_m2", counting)
    run_m2.run_stage(args)
    args.no_resume = True
    assert run_m2.run_stage(args) == run_m2.EXIT_OK
    assert len(calls) == 6
    summary = read_json(tmp_path / "reports" / "summary_stats_m2_t.json")
    assert summary["is_resumed_run"] is False
    assert summary["n_papers_resumed_skipped"] == 0


def test_retry_failed_reprocesses_only_failed_papers(tmp_path, monkeypatch):
    _copy_papers(tmp_path / "raw", NEW_ID_PAPERS)
    args = _args(tmp_path, _write_config(tmp_path))
    real_build_tree = tree_builder.build_tree
    broken = {"on": True}

    def flaky(latex, version=1):
        if broken["on"] and "Our method is simple" in latex:
            raise RuntimeError("boom")
        return real_build_tree(latex, version)

    monkeypatch.setattr(tree_builder, "build_tree", flaky)
    assert run_m2.run_stage(args) == run_m2.EXIT_OK
    assert read_json(tmp_path / "reports" / "summary_stats_m2_t.json")["n_failed"] == 1

    # Sửa lỗi rồi bật --retry_failed: chỉ bài failed được chạy lại, không cần --no_resume.
    broken["on"] = False
    calls: list[str] = []
    real_process = run_m2.process_paper_m2

    def counting(*call_args, **call_kwargs):
        calls.append(call_args[1])
        return real_process(*call_args, **call_kwargs)

    monkeypatch.setattr(run_m2, "process_paper_m2", counting)
    args.retry_failed = True
    assert run_m2.run_stage(args) == run_m2.EXIT_OK
    assert calls == ["2401.00001"]
    summary = read_json(tmp_path / "reports" / "summary_stats_m2_t.json")
    assert (summary["n_failed"], summary["n_ok"], summary["n_empty_record"]) == (0, 2, 1)


def test_resume_skip_honors_retry_failed():
    failed = {"status": run_m2.STATUS_FAILED}
    ok = {"status": run_m2.STATUS_OK}
    assert run_m2._resume_skip(failed) is True
    assert run_m2._resume_skip(failed, retry_failed=True) is False
    assert run_m2._resume_skip(ok, retry_failed=True) is True


def test_flush_checkpoint_continues_after_write_error(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.jsonl"
    calls = {"n": 0}
    real_append = run_m2.append_jsonl

    def flaky(target, record):
        calls["n"] += 1
        if calls["n"] <= 2:  # cả lần đầu và lần thử lại của bản ghi 1 đều lỗi
            raise OSError("disk full")
        return real_append(target, record)

    monkeypatch.setattr(run_m2, "append_jsonl", flaky)
    run_m2._flush_checkpoint(path, [{"arxiv_id": "a"}, {"arxiv_id": "b"}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"arxiv_id": "b"' in lines[0]


def test_run_stage_missing_input_returns_bad_input(tmp_path):
    assert run_m2.run_stage(_args(tmp_path, _write_config(tmp_path))) == run_m2.EXIT_BAD_INPUT


def test_run_stage_no_matching_papers_returns_bad_input(tmp_path):
    _copy_papers(tmp_path / "raw", ["2310-12345"])
    args = _args(tmp_path, _write_config(tmp_path), id_start="2506.00001", id_end="2506.99999")
    assert run_m2.run_stage(args) == run_m2.EXIT_BAD_INPUT


def test_load_checkpoint_filters_by_range_and_stage(tmp_path):
    path = tmp_path / "checkpoint_m2_t.jsonl"
    append_jsonl(path, {"stage": "m2", "id_start": "A", "id_end": "B", "arxiv_id": "2310.12345"})
    append_jsonl(path, {"stage": "m1", "id_start": "A", "id_end": "B", "arxiv_id": "2310.99999"})
    append_jsonl(path, {"stage": "m2", "id_start": "X", "id_end": "Y", "arxiv_id": "2310.88888"})
    assert list(run_m2.load_checkpoint(path, "A", "B")) == ["2310.12345"]
    assert run_m2.load_checkpoint(tmp_path / "khong-co.jsonl", "A", "B") == {}


def test_process_range_flushes_checkpoint_when_interrupted(tmp_path, monkeypatch):
    _copy_papers(tmp_path / "raw", ["2310-12345", "2401-00001"])
    paths = run_m2.M2Paths.build("t", tmp_path / "raw", tmp_path / "processed", tmp_path / "reports")
    paths.papers_out.mkdir(parents=True, exist_ok=True)
    papers = run_m2.list_papers_in_range(paths.papers_in, "2310.00001", "2402.99999")
    context = run_m2._RangeContext(
        paths=paths,
        settings=_settings(),
        id_start="2310.00001",
        id_end="2402.99999",
        checkpoint_every=1,
        profiler=None,
    )
    real_process_one = run_m2._process_one
    seen = {"n": 0}

    def interrupting(*call_args, **call_kwargs):
        seen["n"] += 1
        if seen["n"] == 2:
            raise KeyboardInterrupt
        return real_process_one(*call_args, **call_kwargs)

    monkeypatch.setattr(run_m2, "_process_one", interrupting)
    results, resumed, interrupted = run_m2._process_range(context, papers, {})
    assert interrupted is True
    assert (len(results), resumed) == (1, 0)
    assert list(run_m2.load_checkpoint(paths.checkpoint, "2310.00001", "2402.99999")) == [
        "2310.12345"
    ]


# ---------------------------------------------------------------------------
# Timeout / worker
# ---------------------------------------------------------------------------


def test_process_one_handles_timeout_and_unexpected_error(tmp_path):
    settings = _settings()
    timeout_worker = SimpleNamespace(run=lambda *a, **k: None)
    timed_out = run_m2._process_one(timeout_worker, tmp_path, "2506.00001", settings, tmp_path)
    assert timed_out.status == run_m2.STATUS_TIMEOUT

    def explode(*_args, **_kwargs):
        raise RuntimeError("worker down")

    error_worker = SimpleNamespace(run=explode)
    failed = run_m2._process_one(error_worker, tmp_path, "2506.00002", settings, tmp_path)
    assert failed.status == run_m2.STATUS_FAILED
    assert failed.stage == "process"


def test_paper_worker_recovers_after_timeout(tmp_path):
    settings = run_m2.M2Settings(use_worker=True)
    worker = run_m2.PaperWorker(settings, target=_hanging_worker)
    try:
        first = worker.run(tmp_path, "2506.00001", tmp_path, timeout=30.0)
        assert first == {"arxiv_id": "2506.00001", "status": run_m2.STATUS_OK}
        assert worker.run(tmp_path, "2506.00002-hang", tmp_path, timeout=1.0) is None
        again = worker.run(tmp_path, "2506.00003", tmp_path, timeout=30.0)
        assert again == {"arxiv_id": "2506.00003", "status": run_m2.STATUS_OK}
    finally:
        worker.close()


def test_paper_worker_recovers_after_crash(tmp_path):
    settings = run_m2.M2Settings(use_worker=True)
    worker = run_m2.PaperWorker(settings, target=_crashing_worker)
    try:
        crashed = worker.run(tmp_path, "2506.00001-die", tmp_path, timeout=30.0)
        assert crashed["status"] == run_m2.STATUS_FAILED
        assert crashed["stage"] == "worker"
        # Worker chết không được làm hỏng các bài sau.
        again = worker.run(tmp_path, "2506.00002", tmp_path, timeout=30.0)
        assert again == {"arxiv_id": "2506.00002", "status": run_m2.STATUS_OK}
    finally:
        worker.close()


def test_run_stage_with_worker_processes_paper(tmp_path):
    _copy_papers(tmp_path / "raw", ["2310-12345"])
    config = _write_config(tmp_path, m2={"use_worker": True})
    args = _args(
        tmp_path, config, id_start="2310.00001", id_end="2310.99999", no_worker=False
    )
    assert run_m2.run_stage(args) == run_m2.EXIT_OK
    assert (tmp_path / "processed" / "papers_t" / "2310-12345" / "hierarchy.json").exists()


def test_run_stage_worker_crash_recovery_e2e(monkeypatch, tmp_path):
    """E2E qua run_stage: bài 1 OK -> bài 2 crash worker -> worker tự hồi phục -> bài 3 OK."""
    raw_papers = tmp_path / "raw" / "papers_t"
    for name in ("2506-00098", "2506-00099", "2506-00100"):
        (raw_papers / name).mkdir(parents=True)
    config = _write_config(tmp_path, m2={"use_worker": True})
    args = _args(
        tmp_path,
        config,
        id_start="2506.00098",
        id_end="2506.00100",
        no_worker=False,
    )
    orig_worker = run_m2.PaperWorker
    monkeypatch.setattr(
        run_m2, "PaperWorker", lambda settings: orig_worker(settings, target=_crashing_worker)
    )
    assert run_m2.run_stage(args) == run_m2.EXIT_OK
    summary = read_json(args.reports / f"summary_stats_m2_{args.name}.json")
    assert summary["n_papers_total"] == 3
    assert summary["n_ok"] == 2
    assert summary["n_failed"] == 1
    assert summary["failures"][0]["stage"] == "worker"
    assert summary["failures"][0]["arxiv_id"] == "2506.00099"
    checkpoint_lines = list(iter_jsonl(args.reports / f"checkpoint_m2_{args.name}.jsonl"))
    assert len(checkpoint_lines) == 3
    assert checkpoint_lines[0]["status"] == run_m2.STATUS_OK
    assert checkpoint_lines[1]["status"] == run_m2.STATUS_FAILED
    assert checkpoint_lines[2]["status"] == run_m2.STATUS_OK


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


def test_build_summary_ratios_and_resume_flags():
    args = SimpleNamespace(id_start="2506.00788", id_end="2506.05787")
    settings = _settings()
    result = run_m2.PaperResult(
        arxiv_id="2506.00788",
        status=run_m2.STATUS_OK,
        n_versions=1,
        n_versions_done=1,
        max_depth=2,
        n_bib_before=4,
        n_bib_after=3,
    )
    profile = {"wall_seconds": 1.0, "n_papers": 1, "n_refs_total": 5, "n_refs_co_arxiv_id_total": 2}
    summary = run_m2.build_summary(args, settings, [result], 0, profile)
    assert "n_refs_total" not in summary
    assert "n_refs_co_arxiv_id_total" not in summary
    assert summary["bib_dedup_ratio"] == 0.25
    assert summary["success_rate"] == 1.0
    assert summary["max_tree_depth"] == 2
    assert summary["is_resumed_run"] is False
    empty = run_m2.build_summary(args, settings, [], 0, {})
    assert empty["success_rate"] is None
    assert empty["bib_dedup_ratio"] is None


def test_build_summary_partial_truncated_and_tree_denominator():
    args = SimpleNamespace(id_start="2506.00788", id_end="2506.05787")
    settings = _settings()
    results = [
        run_m2.PaperResult(
            arxiv_id="2506.00788",
            status=run_m2.STATUS_OK,
            n_versions=2,
            n_versions_done=2,
            max_depth=4,
            partial=True,
            truncated=True,
        ),
        run_m2.PaperResult(arxiv_id="2506.00789", status=run_m2.STATUS_EMPTY),
        run_m2.PaperResult(
            arxiv_id="2506.00790", status=run_m2.STATUS_SKIPPED_LARGE, n_versions=3
        ),
        run_m2.PaperResult(arxiv_id="2506.00791", status=run_m2.STATUS_TIMEOUT),
    ]
    summary = run_m2.build_summary(args, settings, results, 0, {})
    assert summary["n_partial"] == 1
    assert summary["n_truncated"] == 1
    assert summary["n_timeout"] == 1
    assert summary["n_papers_with_tree"] == 1
    # Chỉ chia cho bài dựng được cây (1), không phải cả 4 bài.
    assert summary["avg_versions_per_paper"] == 2.0
    assert summary["avg_tree_depth"] == 4.0


# ---------------------------------------------------------------------------
# M2Settings.from_config
# ---------------------------------------------------------------------------


def test_m2_settings_from_missing_config(tmp_path):
    assert run_m2.M2Settings.from_config(tmp_path / "khong-co.yaml") == run_m2.M2Settings()


def test_m2_settings_from_config_without_m2_section(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"pipeline": {"checkpoint_every": 1}}), encoding="utf-8")
    assert run_m2.M2Settings.from_config(path) == run_m2.M2Settings()
