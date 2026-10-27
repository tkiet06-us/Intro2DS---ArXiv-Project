"""Unit tests cho src/m1_crawling/profiler.py (không phụ thuộc mạng)."""

import threading
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


def test_record_paper_accumulates_steps_of_the_same_paper():
    with Profiler() as prof:
        prof.record_paper("2310.12345", seconds=3.0)  # bước arXiv (không có reference)
        prof.record_paper("2310.12345", seconds=1.0, n_refs=30, n_refs_co_arxiv_id=10)  # bước S2

    summary = prof.summary()
    assert summary.n_papers == 1
    assert summary.avg_seconds_per_paper == pytest.approx(4.0)
    assert (summary.n_refs_total, summary.n_refs_co_arxiv_id_total) == (30, 10)


def test_wall_time_per_paper_differs_from_summed_step_time():
    with Profiler() as prof:
        time.sleep(0.05)
        prof.record_paper("2310.12345", seconds=10.0)

    summary = prof.summary()
    assert summary.avg_seconds_per_paper == pytest.approx(10.0)
    assert 0 < summary.avg_wall_seconds_per_paper < 10.0


def test_record_paper_is_safe_from_several_threads():
    with Profiler() as prof:
        threads = [
            threading.Thread(target=prof.record_paper, args=(f"2310.{i:05d}", 1.0, 2, 1))
            for i in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    summary = prof.summary()
    assert (summary.n_papers, summary.n_refs_total) == (20, 40)


# ---------- lấy mẫu lỗi không được làm hỏng phép đo ----------


def test_disk_sampling_failure_keeps_ram_samples_and_thread_alive(tmp_path, monkeypatch):
    def broken_dir_size(path):
        raise OSError("boom")

    monkeypatch.setattr("src.m1_crawling.profiler.dir_size", broken_dir_size)

    with Profiler(watch_dir=tmp_path, interval=0.01, disk_every=1) as prof:
        time.sleep(0.1)

    summary = prof.summary()
    assert summary.n_ram_samples >= 3  # luồng nền vẫn lấy mẫu RAM sau các lần đo đĩa lỗi
    assert summary.n_sample_errors >= 3
    assert summary.max_disk_mb is None


def test_ram_sampling_failure_is_counted_and_later_samples_still_work(monkeypatch):
    original = Profiler._sample_ram
    state = {"calls": 0}

    def flaky(self):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("lần đầu lỗi")
        original(self)

    monkeypatch.setattr(Profiler, "_sample_ram", flaky)

    with Profiler() as prof:
        pass

    summary = prof.summary()
    assert summary.n_sample_errors == 1
    assert summary.n_ram_samples == 1  # mẫu cuối lúc thoát vẫn lấy được


def test_sampling_failure_on_exit_does_not_mask_the_real_error(monkeypatch):
    def always_broken(self):
        raise RuntimeError("lỗi lấy mẫu")

    monkeypatch.setattr(Profiler, "_sample_ram", always_broken)

    with pytest.raises(ValueError, match="lỗi thật"), Profiler():
        raise ValueError("lỗi thật")


def test_only_first_sampling_error_logs_a_traceback(monkeypatch, caplog):
    def always_broken(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(Profiler, "_sample_ram", always_broken)

    with Profiler(interval=0.01) as prof:
        time.sleep(0.1)

    with_traceback = [record for record in caplog.records if record.exc_info]
    assert prof.summary().n_sample_errors >= 3
    assert len(with_traceback) == 1


def test_no_errors_are_reported_in_a_normal_run():
    with Profiler() as prof:
        pass

    assert prof.summary().n_sample_errors == 0