"""Unit tests cho src/m1_crawling/profiler.py (không phụ thuộc mạng)."""

import time

import pytest

from src.common.utils_io import read_json
from src.m1_crawling.profiler import MB, Profiler, dir_size


def test_dir_size_counts_nested_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 50)

    assert dir_size(tmp_path) == 150


def test_dir_size_of_missing_directory_is_zero(tmp_path):
    assert dir_size(tmp_path / "does_not_exist") == 0


def test_short_block_still_has_at_least_two_samples():
    with Profiler() as prof:
        pass

    summary = prof.summary()
    assert summary.n_ram_samples >= 2  # mẫu lúc vào và mẫu lúc ra
    assert summary.peak_ram_mb >= summary.avg_ram_mb > 0
    assert summary.wall_seconds >= 0


def test_periodic_sampling_collects_more_samples_while_running():
    with Profiler(interval=0.01) as prof:
        time.sleep(0.15)

    assert prof.summary().n_ram_samples >= 3


def test_disk_is_measured_on_enter_and_exit(tmp_path):
    with Profiler(watch_dir=tmp_path) as prof:
        (tmp_path / "out.bin").write_bytes(b"x" * (2 * MB))

    summary = prof.summary()
    assert summary.final_disk_mb == pytest.approx(2.0)
    assert summary.max_disk_mb >= summary.final_disk_mb


def test_max_disk_remembers_temporary_files(tmp_path):
    with Profiler(watch_dir=tmp_path, interval=0.01, disk_every=1) as prof:
        temp = tmp_path / "temp.bin"
        temp.write_bytes(b"x" * (3 * MB))
        time.sleep(0.1)  # để thread nền kịp đo lúc file tạm còn tồn tại
        temp.unlink()

    summary = prof.summary()
    assert summary.max_disk_mb == pytest.approx(3.0)
    assert summary.final_disk_mb == pytest.approx(0.0)


def test_disk_is_none_without_watch_dir():
    with Profiler() as prof:
        pass

    summary = prof.summary()
    assert summary.max_disk_mb is None
    assert summary.final_disk_mb is None


def test_record_paper_aggregates_per_paper_stats():
    with Profiler() as prof:
        prof.record_paper("2310.12345", seconds=2.0, n_refs=30, n_refs_co_arxiv_id=10)
        prof.record_paper("2401.00001", seconds=4.0, n_refs=20, n_refs_co_arxiv_id=5)

    summary = prof.summary()
    assert summary.n_papers == 2
    assert summary.avg_seconds_per_paper == pytest.approx(3.0)
    assert (summary.n_refs_total, summary.n_refs_co_arxiv_id_total) == (50, 15)


def test_average_time_is_none_when_no_paper_recorded():
    with Profiler() as prof:
        pass

    assert prof.summary().avg_seconds_per_paper is None


def test_exception_inside_block_is_not_swallowed_and_thread_stops():
    prof = Profiler(interval=0.01)

    with pytest.raises(ValueError), prof:
        raise ValueError("boom")

    assert prof._thread is not None
    assert not prof._thread.is_alive()
    assert prof.summary().n_ram_samples >= 2


def test_summary_before_use_is_all_zero():
    summary = Profiler().summary()

    assert summary.peak_ram_mb == 0
    assert summary.n_ram_samples == 0


def test_save_writes_readable_json(tmp_path):
    with Profiler() as prof:
        prof.record_paper("2310.12345", seconds=1.0, n_refs=3, n_refs_co_arxiv_id=1)

    prof.save(tmp_path / "reports" / "profile.json")

    data = read_json(tmp_path / "reports" / "profile.json")
    assert data["n_papers"] == 1
    assert data["max_disk_mb"] is None
