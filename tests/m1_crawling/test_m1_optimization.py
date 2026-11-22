"""Test cho đợt tối ưu crawler M1: downloader.py, extractor.py, hàng đợi trích xuất, discovery.

Không gọi mạng thật: dùng lại phiên HTTP/đồng hồ giả của ``test_crawler.py``.
"""

import base64
import gzip
import io
import json
import os
import tarfile
import threading
import time
from itertools import pairwise
from pathlib import Path

import pytest
import requests

from src.m1_crawling import crawler, discovery, downloader, extractor
from src.m1_crawling.profiler import MB, Profiler
from tests.m1_crawling.test_crawler import (
    BIB,
    PNG,
    TEX,
    FakeClock,
    FakeResponse,
    FakeSession,
    discovery_rows,
    good_script,
    make_fetcher,
    make_tar_gz,
    one_version_row,
    read_log,
    run_crawl,
)

URL = "https://export.arxiv.org/src/2506.00788v1"


class BrokenResponse(FakeResponse):
    """Trả một phần body rồi đứt kết nối (như khi đọc treo quá timeout)."""

    def iter_content(self, chunk_size=1):
        yield self._body
        raise requests.exceptions.ChunkedEncodingError("dut ket noi")


class UnreadableResponse(FakeResponse):
    """Body tuyệt đối không được đọc (dùng cho version chỉ có PDF)."""

    def iter_content(self, chunk_size=1):
        raise AssertionError("khong duoc doc body cua PDF")


def range_of(session: FakeSession, call: int) -> str | None:
    return session.kwargs[call]["headers"].get("Range")


# ---------------------------------------------------------------- Pha 1: tải xuống đĩa
def test_download_streams_to_disk_and_writes_meta_before_final_name(tmp_path: Path):
    body = os.urandom(200_000)
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(body=body))
    dest = tmp_path / "spool" / "2506-00788v1.src"
    result = fetcher.download(URL, dest)
    assert result.path == dest and result.body == b""  # body không nằm trong RAM
    assert dest.read_bytes() == body and result.size == len(body)
    assert not crawler.part_path(dest).exists()
    meta = json.loads(crawler.meta_path(dest).read_text(encoding="utf-8"))
    assert meta["http_status"] == 200 and meta["attempts"] == 1 and meta["url"] == URL
    assert meta["content_type"] == "application/x-eprint-tar"


def test_download_resumes_with_range_inside_one_call(tmp_path: Path):
    body = b"0123456789"

    def script(url, n):
        if n == 1:
            return BrokenResponse(body=body[:4], headers={"Content-Length": "10"})
        return FakeResponse(status=206, body=body[4:], headers={"Content-Range": "bytes 4-9/10"})

    fetcher, session, _ = make_fetcher(script)
    dest = tmp_path / "x.src"
    result = fetcher.download(URL, dest)
    assert dest.read_bytes() == body and result.attempts == 2
    assert range_of(session, 0) is None and range_of(session, 1) == "bytes=4-"
    assert [a["bytes"] for a in result.attempt_log] == [4, 10]
    assert result.attempt_log[0]["error"].startswith("ChunkedEncodingError")


def test_download_resumes_part_file_left_by_a_killed_process(tmp_path: Path):
    """Tiến trình trước bị tắt giữa chừng: lần chạy mới tải TIẾP từ kích thước file .part."""
    body = b"0123456789"
    dest = tmp_path / "x.src"
    crawler.part_path(dest).write_bytes(body[:6])
    script = lambda url, n: FakeResponse(  # noqa: E731
        status=206, body=body[6:], headers={"Content-Range": "bytes 6-9/10"}
    )
    fetcher, session, _ = make_fetcher(script)
    fetcher.download(URL, dest)
    assert dest.read_bytes() == body and len(session.calls) == 1
    assert range_of(session, 0) == "bytes=6-"


def test_download_restarts_when_server_ignores_range_or_range_is_stale(tmp_path: Path):
    body = b"0123456789"
    dest = tmp_path / "x.src"
    crawler.part_path(dest).write_bytes(b"cu-")
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(body=body))  # 200: gửi lại từ đầu
    fetcher.download(URL, dest)
    assert dest.read_bytes() == body

    crawler.part_path(dest).write_bytes(body + b"thua")  # .part dài hơn file thật -> 416

    def script(url, n):
        return FakeResponse(status=416) if n == 1 else FakeResponse(body=body)

    fetcher, session, _ = make_fetcher(script)
    fetcher.download(URL, dest)
    assert dest.read_bytes() == body and range_of(session, 1) is None

    crawler.part_path(dest).write_bytes(b"abc")  # Content-Range lệch: bỏ .part, tải lại

    def lech(url, n):
        if n == 1:
            return FakeResponse(status=206, body=b"zz", headers={"Content-Range": "bytes 9-9/10"})
        return FakeResponse(body=body)

    fetcher, _, _ = make_fetcher(lech)
    fetcher.download(URL, dest)
    assert dest.read_bytes() == body


def test_download_too_large_removes_part_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(crawler, "MAX_DOWNLOAD_BYTES", 10)
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(body=b"x" * 100))
    dest = tmp_path / "x.src"
    with pytest.raises(crawler.TooLargeError) as info:
        fetcher.download(URL, dest)
    assert info.value.downloaded == 100 and info.value.attempts == 1
    assert not dest.exists() and not crawler.part_path(dest).exists()


def test_only_one_request_at_a_time_across_threads():
    """Ràng buộc 1: hai luồng cùng gọi thì request không chồng nhau và cách nhau >= delay."""
    clock = FakeClock()
    active, starts, overlaps = [0], [], []

    class SlowSession(FakeSession):
        def get(self, url, **kwargs):
            active[0] += 1
            overlaps.append(active[0])
            starts.append(clock.now)
            time.sleep(0.01)  # giữ kết nối mở một lúc để luồng kia có cơ hội chen vào
            clock.now += 0.5
            active[0] -= 1
            return FakeResponse(body=b"x")

    fetcher = crawler.PoliteFetcher(
        "UA", 3.0, session=SlowSession(None), sleep=clock.sleep, clock=clock.monotonic
    )
    threads = [
        threading.Thread(target=lambda: [fetcher.get(URL) for _ in range(5)]) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(starts) == 10 and max(overlaps) == 1
    assert all(later - earlier >= 3.0 for earlier, later in pairwise(starts))


def test_note_external_request_makes_the_first_request_wait():
    fetcher, _, clock = make_fetcher(lambda url, n: FakeResponse(body=b"x"))
    fetcher.note_external_request()  # discovery.py vừa gọi arXiv
    fetcher.get(URL)
    assert clock.sleeps == [pytest.approx(3.0)]


def test_real_http_download_resumes_with_range_and_skips_pdf_body(tmp_path: Path):
    """Thư viện requests thật + server HTTP thật trên localhost (không ra Internet)."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    body = os.urandom(300_000)
    seen: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            asked = self.headers.get("Range")
            seen.append((self.path, asked))
            if self.path == "/src/pdf":
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Length", str(50_000_000))  # không bao giờ gửi body
                self.end_headers()
                return
            if asked is None:  # lần đầu: hứa đủ nhưng chỉ gửi 100.000 byte rồi đứt
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body[:100_000])
                return
            start = int(asked.removeprefix("bytes=").rstrip("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            self.wfile.write(body[start:])

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        fetcher = crawler.PoliteFetcher("UA", 3.0, sleep=lambda _s: None, backoff=(0.0,))
        result = fetcher.download(f"{base}/src/tar", tmp_path / "a.src")
        pdf = fetcher.download(f"{base}/src/pdf", tmp_path / "b.src")
    finally:
        server.shutdown()
    assert (tmp_path / "a.src").read_bytes() == body and result.attempts == 2
    # Đứt giữa một khối 64 KB thì phần lẻ của khối đó bị bỏ: tải tiếp từ bội số của 64 KB.
    assert seen[:2] == [("/src/tar", None), ("/src/tar", "bytes=65536-")]
    assert pdf.body_skipped and pdf.content_length == 50_000_000
    assert not (tmp_path / "b.src").exists()


# ---------------------------------------------------------------- Pha 2: bớt byte, bớt treo
def pdf_response():
    return UnreadableResponse(
        headers={"Content-Type": "application/pdf", "Content-Length": "123456"}
    )


def test_pdf_body_is_never_downloaded(tmp_path: Path):
    fetcher, session, _ = make_fetcher(lambda url, n: pdf_response())
    dest = tmp_path / "x.src"
    result = fetcher.download(URL, dest)
    assert result.body_skipped and result.path is None and result.content_length == 123456
    assert session.calls == [URL] and list(tmp_path.iterdir()) == []


def test_get_still_returns_pdf_body():
    headers = {"Content-Type": "application/pdf"}
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(body=b"%PDF-1.5", headers=headers))
    assert fetcher.get(URL).body == b"%PDF-1.5"


def test_pdf_only_version_gets_empty_record_with_size_from_content_length(tmp_path: Path):
    summary, _, _ = run_crawl(
        tmp_path, script=lambda url, n: pdf_response(), papers=one_version_row()
    )
    record = read_log(tmp_path)[-1]
    assert record["status"] == "empty" and record["reason"] == "pdf_only"
    assert record["archive_bytes"] == record["size_before"] == 123456
    assert record["n_files_before"] == 1 and record["body_skipped"] is True
    v1 = tmp_path / "2506-00789" / "v1"
    assert v1.is_dir() and not any(v1.iterdir())  # vẫn có bản ghi rỗng
    assert summary["bytes_skipped_pdf"] == 123456 and summary["by_status"] == {"empty": 1}
    assert run_crawl(tmp_path, script=lambda url, n: pdf_response(), papers=one_version_row())[
        1
    ].calls == []


def test_read_timeout_defaults_to_30_seconds_and_comes_from_config(tmp_path: Path):
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(body=b"x"))
    fetcher.get(URL)
    assert session.kwargs[0]["timeout"] == (15.0, 30.0)
    config = tmp_path / "c.yaml"
    config.write_text(
        "crawler:\n  timeout_read: 12\n  extract_workers: 2\n  max_spool_gb: 0.5\n",
        encoding="utf-8",
    )
    settings = crawler.CrawlerSettings.from_config(config)
    assert (settings.timeout_read, settings.extract_workers) == (12.0, 2)
    assert settings.max_spool_gb == 0.5
    assert settings.make_fetcher("a@b.c")._timeout == (15.0, 12.0)
    assert crawler.CrawlerSettings.from_config(tmp_path / "khong-co.yaml").timeout_read == 30.0


def test_every_attempt_is_logged_with_wait_and_transfer_time(tmp_path: Path):
    def script(url, n):
        if n == 1:
            return FakeResponse(status=503)
        if n == 2:
            return requests.ConnectionError("reset")
        return FakeResponse(body=make_tar_gz({"main.tex": TEX}))

    summary, _, clock = run_crawl(tmp_path, script=script, papers=one_version_row())
    record = read_log(tmp_path)[-1]
    assert record["status"] == "ok" and record["attempts"] == 3
    assert [a["error"] for a in record["attempt_log"]][:2] == ["HTTP 503", "ConnectionError: reset"]
    assert record["attempt_log"][2]["error"] is None
    assert record["wait_seconds"] == pytest.approx(sum(clock.sleeps))
    assert record["seconds"] >= record["wait_seconds"] >= 40  # backoff 10 + 30 và hai lần chờ 3s
    assert record["wall_seconds"] == pytest.approx(
        record["seconds"] + record["extract_seconds"], abs=0.02
    )
    assert summary["retry_attempts_total"] == 2
    assert summary["attempt_errors_by_type"] == {"HTTP 503": 1, "ConnectionError": 1}
    assert summary["wait_seconds_total"] == pytest.approx(record["wait_seconds"], abs=0.1)


def test_failed_download_record_keeps_attempt_log(tmp_path: Path):
    run_crawl(tmp_path, script=lambda url, n: FakeResponse(status=500), papers=one_version_row())
    record = read_log(tmp_path)[-1]
    assert record["status"] == "error" and record["attempts"] == 4
    assert len(record["attempt_log"]) == 4


def test_attempt_log_is_capped():
    """Tải tiếp nhiều lần (mỗi lần được thêm 1 byte) không làm attempt_log phình vô hạn."""
    body = bytes(range(40))

    def script(url, n):
        have = n - 1
        headers = {"Content-Range": f"bytes {have}-{have}/40"} if have else {"Content-Length": "40"}
        if n == 40:
            return FakeResponse(status=206, body=body[39:], headers=headers)
        return BrokenResponse(status=206 if have else 200, body=body[have:n], headers=headers)

    fetcher, _, _ = make_fetcher(script)
    result = fetcher.get(URL)
    assert result.body == body and result.attempts == 40
    assert len(result.attempt_log) == downloader.MAX_ATTEMPT_LOG


# ---------------------------------------------------------------- Pha 5: 3 giây tính từ lúc gửi
def timed_fetcher(request_seconds: float):
    clock = FakeClock()

    class TimedSession(FakeSession):
        def get(self, url, **kwargs):
            self.calls.append(url)
            self.kwargs.append(kwargs)
            self.sent.append(clock.now)
            outcome = self.script(url, len(self.calls))
            clock.now += request_seconds  # thời gian truyền
            return outcome

    session = TimedSession(lambda url, n: FakeResponse(body=b"x"))
    session.sent = []
    fetcher = crawler.PoliteFetcher(
        "UA", 3.0, session=session, sleep=clock.sleep, clock=clock.monotonic
    )
    return fetcher, session, clock


def test_delay_counts_from_when_the_previous_request_was_sent():
    fetcher, session, clock = timed_fetcher(2.0)
    for _ in range(3):
        fetcher.get(URL)
    assert clock.sleeps == [pytest.approx(1.0), pytest.approx(1.0)]
    assert [b - a for a, b in pairwise(session.sent)] == [3.0, 3.0]


def test_request_slower_than_the_delay_is_followed_without_extra_sleep():
    fetcher, session, clock = timed_fetcher(7.0)  # truyền lâu hơn delay
    for _ in range(3):
        fetcher.get(URL)
    assert clock.sleeps == []  # request trước vừa đóng là gửi tiếp được
    assert [b - a for a, b in pairwise(session.sent)] == [7.0, 7.0]


def test_delay_from_send_also_applies_to_retries_and_redirects():
    clock = FakeClock()

    def script(url, n):
        clock.now += 1.0
        if n == 1:
            return FakeResponse(status=301, headers={"Location": "/src/new"})
        if n == 2:
            return FakeResponse(status=503)
        return FakeResponse(body=b"ok")

    session = FakeSession(script)
    sent = []
    real_get = session.get
    session.get = lambda url, **kw: (sent.append(clock.now), real_get(url, **kw))[1]
    fetcher = crawler.PoliteFetcher(
        "UA", 3.0, session=session, sleep=clock.sleep, clock=clock.monotonic, backoff=(0.5,),
    )  # fmt: skip
    assert fetcher.get(URL).body == b"ok"
    assert all(b - a >= 3.0 for a, b in pairwise(sent)) and len(sent) == 3


# ---------------------------------------------------------------- Pha 3: extractor theo luồng
def plain_tar(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def add_symlink(tar: tarfile.TarFile) -> None:
    link = tarfile.TarInfo("link.tex")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    tar.addfile(link)


BIG_TAR_GZ = make_tar_gz({"main.tex": TEX, "big.bin": os.urandom(300_000), "z.tex": TEX})
TWO_MEMBER_TAR = plain_tar({"a.tex": b"\\begin{x}" + b"y" * 4991, "b.tex": b"\\begin{x}" + b"y" * 4991})
FIXTURES = {
    "tar_gz": make_tar_gz(
        {"main.tex": TEX, "refs.bib": BIB, "fig/a.png": PNG, "sections/intro.tex": b"Intro."}
    ),
    "tar": plain_tar({"main.tex": TEX, "refs.bib": BIB}),
    "gz_don": gzip.compress(TEX),
    "gz_lon": gzip.compress(b"\\documentclass{a}\n" + b"x" * 700_000),
    "van_ban_tho": TEX,
    "tex_khong_preamble": b"\\def\\a{1}\n\\bye\n",
    "pdf": b"%PDF-1.5\n" + b"x" * 100,
    "pdf_trong_gz": gzip.compress(b"%PDF-1.4\n\\begin{x}"),
    "html": b"<!DOCTYPE html><html>No source</html>",
    "html_trong_gz": gzip.compress(b"<!DOCTYPE html><html>Service unavailable</html>"),
    "zip": b"PK\x03\x04" + b"x" * 100,
    "nhi_phan": b"\x00\x01\x02binary",
    "postscript_gz": gzip.compress(b"%!PS-Adobe-3.0\n\\def\\x{1}\n"),
    "khong_marker": b"plain text with a \\ backslash and nothing TeX-like",
    "rong": b"",
    "tar_khong_tex": make_tar_gz({"a.png": PNG, "notes.txt": b"hi"}),
    "tar_chi_bib": make_tar_gz({"refs.bib": BIB}),
    "tar_gz_cut": BIG_TAR_GZ[: len(BIG_TAR_GZ) // 2],
    "tar_cut_giua_member": TWO_MEMBER_TAR[: 512 + 5120 + 512 + 100],
    "gz_hong": gzip.compress(TEX)[:-12],
    "duong_dan_nguy_hiem": make_tar_gz(
        {"main.tex": TEX, "../evil.tex": b"x", "/etc/passwd.tex": b"x", "a/../../b.tex": b"x"}
    ),
    "symlink": make_tar_gz({"main.tex": TEX}, extra=add_symlink),
    "rac_macos": make_tar_gz({"main.tex": TEX, "__MACOSX/._main.tex": b"j", "._x.tex": b"j"}),
    "ky_tu_windows": make_tar_gz({"main.tex": TEX, "we:ird?.tex": b"x"}),
    "trung_ten": make_tar_gz({"main.tex": TEX, "Main.tex": b"other"}),
}


def via_path(tmp_path: Path, data: bytes):
    src, dest = tmp_path / "goi.src", tmp_path / "tmp"
    src.write_bytes(data)
    result = extractor.extract_from_path(src, dest)
    on_disk = {
        path.relative_to(dest).as_posix(): path.read_bytes()
        for path in dest.rglob("*")
        if path.is_file()
    }
    return result, on_disk


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_streaming_extractor_matches_the_in_memory_one(tmp_path: Path, name: str):
    """Hồi quy: đường mới (đọc từ đĩa theo luồng) cho kết quả giống hệt đường cũ (trong RAM)."""
    data = FIXTURES[name]
    old = extractor.extract_tex_sources(data)
    new, on_disk = via_path(tmp_path, data)
    assert (new.status, new.reason) == (old.status, old.reason)
    assert on_disk == old.files  # tên file + nội dung từng byte
    assert new.sizes == {name_: len(content) for name_, content in old.files.items()}
    assert (new.size_before, new.n_files_before) == (old.size_before, old.n_files_before)
    assert new.skipped == old.skipped and new.size_after == old.size_after


@pytest.mark.parametrize(
    ("limit_name", "value", "name"),
    [
        ("MAX_UNCOMPRESSED_BYTES", 1000, "tar_gz"),
        ("MAX_UNCOMPRESSED_BYTES", 1000, "gz_lon"),
        ("MAX_KEPT_FILE_BYTES", 10, "tar_gz"),
    ],
)
def test_streaming_extractor_matches_under_tight_limits(
    tmp_path: Path, monkeypatch, limit_name: str, value: int, name: str
):
    monkeypatch.setattr(crawler, limit_name, value)
    old = extractor.extract_tex_sources(FIXTURES[name])
    new, on_disk = via_path(tmp_path, FIXTURES[name])
    assert (new.status, new.reason, new.size_before) == (old.status, old.reason, old.size_before)
    assert on_disk == old.files and new.skipped == old.skipped


def test_single_gz_is_decompressed_in_blocks_not_into_ram(tmp_path: Path, monkeypatch):
    """File .gz đơn lớn: không có lần read() nào lấy cả file (trần 1 MB mỗi khối)."""
    sizes: list[int] = []
    real = extractor._read_block

    def spy(reader, size):
        block = real(reader, size)
        sizes.append(len(block))
        return block

    monkeypatch.setattr(extractor, "_read_block", spy)
    # base64 để nội dung khó nén nhưng không chứa byte NUL (có NUL sẽ bị coi là nhị phân).
    payload = b"\\documentclass{a}\n" + base64.b64encode(os.urandom(4 * 1024 * 1024))
    result, on_disk = via_path(tmp_path, gzip.compress(payload, compresslevel=1))
    assert result.status == "ok" and on_disk == {"main.tex": payload}
    assert max(sizes) <= extractor.COPY_BYTES


def test_disk_write_error_is_not_mistaken_for_a_corrupt_archive(tmp_path: Path):
    """Lỗi GHI (ở đây: tên file đụng tên thư mục) phải nổ ra, không được thành corrupt_archive
    (corrupt_archive sau vài lượt thử sẽ bị chốt rỗng, tức là mất dữ liệu trong im lặng)."""
    src, dest = tmp_path / "goi.src", tmp_path / "tmp"
    src.write_bytes(make_tar_gz({"a.tex": TEX, "a.tex/b.tex": TEX}))
    with pytest.raises(OSError):
        extractor.extract_from_path(src, dest)


def test_write_version_accepts_a_staged_directory(tmp_path: Path):
    staged = extractor.staging_dir(tmp_path, 3)
    (staged / "main.tex").write_bytes(TEX)
    extractor.write_version(tmp_path, 3, staged)
    assert (tmp_path / "v3" / "main.tex").read_bytes() == TEX and not staged.exists()


# ---------------------------------------------------------------- Pha 3: spool, hàng đợi
def test_spool_is_empty_and_removed_after_a_clean_run(tmp_path: Path):
    out, spool = tmp_path / "out", tmp_path / "spool"
    run_crawl(out, spool_dir=spool)
    assert not spool.exists() and not (out / crawler.SPOOL_DIRNAME).exists()
    run_crawl(tmp_path / "mac-dinh")  # không chỉ định: dùng <out>/.spool rồi dọn
    assert not (tmp_path / "mac-dinh" / crawler.SPOOL_DIRNAME).exists()


EXPECTED_VERSIONS = [("2506.00788", 1), ("2506.00788", 2), ("2506.00789", 1), ("hep-th/9901001", 1)]


def test_extraction_runs_on_a_background_thread_and_results_arrive_in_order(
    tmp_path: Path, monkeypatch
):
    """Nhiều luồng trích xuất: KẾT QUẢ về luồng tải theo đúng thứ tự nộp (bộ đếm lỗi liên tiếp
    dựa vào thứ tự này). Thứ tự DÒNG trong crawl_log.jsonl thì không được đảm bảo, vì mỗi luồng
    tự ghi dòng của mình khi xong; load_log_state đọc theo (bài, version) nên không cần."""
    names: list[str] = []
    handled: list[tuple[str, int]] = []
    real_extract, real_progress = crawler.extract_from_path, crawler._log_progress

    def spy_extract(path, dest_tmp):
        names.append(threading.current_thread().name)
        return real_extract(path, dest_tmp)

    def spy_progress(record, *args):
        handled.append((record["arxiv_id"], record["version"]))
        return real_progress(record, *args)

    monkeypatch.setattr(crawler, "extract_from_path", spy_extract)
    monkeypatch.setattr(crawler, "_log_progress", spy_progress)
    summary, _, _ = run_crawl(tmp_path, extract_workers=2)
    assert len(names) == 4 and all(name.startswith("extract") for name in names)
    assert handled == EXPECTED_VERSIONS
    assert sorted((r["arxiv_id"], r["version"]) for r in read_log(tmp_path)) == sorted(
        EXPECTED_VERSIONS
    )
    assert summary["by_status"] == {"ok": 3, "empty": 1}  # 2506.00789 là bài chỉ có PDF


def test_single_extract_worker_keeps_log_lines_in_version_order(tmp_path: Path):
    """Mặc định (1 luồng trích xuất, hàng đợi FIFO) thì thứ tự dòng log cũng cố định."""
    run_crawl(tmp_path, extract_workers=1)
    assert [(r["arxiv_id"], r["version"]) for r in read_log(tmp_path)] == EXPECTED_VERSIONS


def test_spooled_source_is_extracted_on_restart_without_downloading_again(tmp_path: Path):
    """Bị tắt sau khi tải xong nhưng trước khi trích xuất: lần sau KHÔNG tải lại."""
    out, spool = tmp_path / "out", tmp_path / "spool"
    spool.mkdir()
    src = spool / "2506-00789v1.src"
    src.write_bytes(make_tar_gz({"main.tex": TEX, "fig.png": PNG}))
    crawler.meta_path(src).write_text(
        json.dumps({"url": "https://export.arxiv.org/src/2506.00789v1", "http_status": 200,
                    "content_type": "application/x-eprint-tar", "attempts": 2, "seconds": 9.5}),
        encoding="utf-8",
    )  # fmt: skip
    never = lambda url, n: pytest.fail("khong duoc tai lai")  # noqa: E731
    summary, session, _ = run_crawl(out, script=never, papers=one_version_row(), spool_dir=spool)
    assert session.calls == [] and summary["by_status"] == {"ok": 1}
    assert (out / "2506-00789" / "v1" / "main.tex").read_bytes() == TEX
    assert (out / "2506-00789" / "metadata.json").exists()
    record = read_log(out)[-1]
    assert record["attempts"] == 2 and record["seconds"] == 9.5 and record["size_before"] > 0
    assert not spool.exists()  # gói và meta đã được dọn


def test_part_file_in_spool_is_resumed_by_the_next_run(tmp_path: Path):
    out, spool = tmp_path / "out", tmp_path / "spool"
    body = make_tar_gz({"main.tex": TEX})
    spool.mkdir()
    (spool / "2506-00789v1.part").write_bytes(body[:20])

    def script(url, n):
        return FakeResponse(
            status=206,
            body=body[20:],
            headers={"Content-Range": f"bytes 20-{len(body) - 1}/{len(body)}"},
        )

    summary, session, _ = run_crawl(out, script=script, papers=one_version_row(), spool_dir=spool)
    assert range_of(session, 0) == "bytes=20-" and summary["by_status"] == {"ok": 1}
    assert (out / "2506-00789" / "v1" / "main.tex").read_bytes() == TEX


def test_orphans_and_broken_spool_files_are_cleaned_up(tmp_path: Path):
    out, spool = tmp_path / "out", tmp_path / "spool"
    run_crawl(out, papers=one_version_row(), spool_dir=spool)  # 2506.00789v1 đã xong
    spool.mkdir()
    for name in ("2506-00789v1.src", "2506-00789v1.meta.json", "2506-00789v1.part"):
        (spool / name).write_bytes(b"mo coi")
    (spool / "2506-00788v1.src").write_bytes(b"thieu meta")  # .src không có meta: tải lại
    (spool / "2506-00788v2.meta.json").write_text("{}", encoding="utf-8")  # meta lẻ
    (spool / "9999-00001v1.src").write_bytes(b"cua dai khac")  # không thuộc lần chạy này
    _, session, _ = run_crawl(out, papers=discovery_rows()[:2], spool_dir=spool)
    assert [c.rsplit("/", 1)[-1] for c in session.calls] == ["2506.00788v1", "2506.00788v2"]
    assert sorted(p.name for p in spool.iterdir()) == ["9999-00001v1.src"]


def test_fresh_run_ignores_and_clears_the_spool(tmp_path: Path):
    out, spool = tmp_path / "out", tmp_path / "spool"
    spool.mkdir()
    (spool / "2506-00789v1.src").write_bytes(b"rac")
    (spool / "2506-00789v1.meta.json").write_text('{"http_status": 200}', encoding="utf-8")
    summary, session, _ = run_crawl(
        out, script=lambda url, n: FakeResponse(body=make_tar_gz({"main.tex": TEX})),
        papers=one_version_row(), spool_dir=spool, resume=False,
    )  # fmt: skip
    assert len(session.calls) == 1 and summary["by_status"] == {"ok": 1}


def test_crash_between_download_and_extraction_loses_nothing(tmp_path: Path, monkeypatch):
    """Trích xuất nổ (giả lập bị tắt): gói vẫn nằm trong spool, lần sau làm tiếp không cần mạng."""
    out, spool = tmp_path / "out", tmp_path / "spool"
    real = crawler.extract_version
    monkeypatch.setattr(
        crawler, "extract_version", lambda job: (_ for _ in ()).throw(RuntimeError("tat may"))
    )
    summary, session, _ = run_crawl(out, papers=one_version_row(), spool_dir=spool)
    assert len(session.calls) == 1 and summary["by_status"] == {"error": 1}
    assert not (out / crawler.LOG_NAME).exists()  # chưa có dòng log nào: version chưa xong
    assert (spool / "2506-00789v1.src").exists()
    monkeypatch.setattr(crawler, "extract_version", real)
    summary, session, _ = run_crawl(out, papers=one_version_row(), spool_dir=spool)
    assert session.calls == [] and summary["by_status"] == {"empty": 1}


def spooled_job(tmp_path: Path, version: int, size: int) -> crawler.Spooled:
    return crawler.Spooled(
        {"arxiv_id": "2506.00788"}, version, tmp_path, tmp_path / "log.jsonl",
        {"size": size}, tmp_path / f"v{version}.src",
    )  # fmt: skip


def test_download_thread_waits_while_the_spool_is_over_its_limit(tmp_path: Path, monkeypatch):
    release = threading.Event()
    handled: list[int] = []

    def slow_extract(job):
        release.wait(5)
        return {"arxiv_id": "2506.00788", "version": job.version, "status": "ok"}

    monkeypatch.setattr(crawler, "extract_version", slow_extract)
    pipeline = crawler.ExtractPipeline(
        lambda record: handled.append(record["version"]), workers=1, max_spool_bytes=100
    )
    try:
        pipeline.submit(spooled_job(tmp_path, 1, 60))  # dưới ngưỡng: không chờ
        assert handled == []
        blocked = threading.Thread(target=pipeline.submit, args=(spooled_job(tmp_path, 2, 60),))
        blocked.start()  # 120 byte chưa trích xuất > 100: luồng tải phải chờ
        blocked.join(0.3)
        assert blocked.is_alive() and handled == []
        release.set()
        blocked.join(5)
        assert not blocked.is_alive() and handled[0] == 1
        pipeline.flush()
        assert handled == [1, 2]
    finally:
        release.set()
        pipeline.close()


def test_pipeline_queue_is_bounded_by_count(tmp_path: Path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(
        crawler,
        "extract_version",
        lambda job: release.wait(5) and {"arxiv_id": "x", "version": job.version, "status": "ok"},
    )
    handled: list[int] = []
    pipeline = crawler.ExtractPipeline(lambda r: handled.append(r["version"]), workers=1)
    try:
        for version in (1, 2):
            pipeline.submit(spooled_job(tmp_path, version, 0))
        third = threading.Thread(target=pipeline.submit, args=(spooled_job(tmp_path, 3, 0),))
        third.start()  # hàng đợi tối đa 2 x workers
        third.join(0.3)
        assert third.is_alive()
        release.set()
        third.join(5)
        pipeline.flush()
        assert handled == [1, 2, 3]
    finally:
        release.set()
        pipeline.close()


def test_max_disk_gb_counts_the_spool(tmp_path: Path, monkeypatch):
    out, spool = tmp_path / "out", tmp_path / "spool"
    spool.mkdir()
    (spool / "goi-dang-cho.bin").write_bytes(b"x" * 4096)
    monkeypatch.setattr(crawler, "DISK_CHECK_EVERY", 1)
    fetcher, _, _ = make_fetcher(good_script)
    with pytest.raises(crawler.DiskLimitError):
        crawler.run(discovery_rows(), out, fetcher, spool_dir=spool, max_disk_gb=3000 / 1024**3)


def test_stop_event_stops_downloads_but_finishes_queued_extraction(tmp_path: Path):
    stop = threading.Event()

    def script(url, n):
        if n == 2:
            stop.set()  # "Ctrl+C" tới trong lúc đang tải version thứ hai
        return good_script(url, n)

    fetcher, session, _ = make_fetcher(script)
    summary = crawler.run(discovery_rows(), tmp_path, fetcher, stop_event=stop)
    assert len(session.calls) == 2 and summary["stopped"] is True
    assert summary["by_status"] == {"ok": 2} and summary["versions_not_attempted"] == 2
    resumed, session2, _ = run_crawl(tmp_path)
    assert len(session2.calls) == 2 and resumed["stopped"] is False


def test_profiler_adds_spool_to_peak_disk_but_not_to_final_size(tmp_path: Path):
    out, spool = tmp_path / "out", tmp_path / "spool"
    out.mkdir()
    spool.mkdir()
    with Profiler(watch_dir=out, extra_dirs=[spool], interval=0.01, disk_every=1) as prof:
        (out / "giu-lai.bin").write_bytes(b"x" * MB)
        (spool / "goi.src").write_bytes(b"x" * (3 * MB))
        time.sleep(0.1)
        (spool / "goi.src").unlink()
    summary = prof.summary()
    assert summary.max_disk_mb == pytest.approx(4.0)
    assert summary.final_disk_mb == pytest.approx(1.0)


# ---------------------------------------------------------------- Pha 4: discovery từng dòng
def kaggle_line(arxiv_id: str, n_versions: int = 1) -> str:
    versions = [
        {"version": f"v{i}", "created": f"Mon, {i} Jun 2025 12:00:00 GMT"}
        for i in range(1, n_versions + 1)
    ]
    return json.dumps({"id": arxiv_id, "title": "T 2506.99999", "versions": versions})


def test_kaggle_snapshot_is_read_line_by_line_and_filtered_by_range(tmp_path: Path):
    path = tmp_path / "kaggle.json"
    lines = [
        kaggle_line("0704.0001"),
        kaggle_line("hep-th/9901001"),
        kaggle_line("2506.00787"),  # ngay dưới dải
        kaggle_line("2506.00788", 2),
        kaggle_line("2506.05787"),
        kaggle_line("2506.05788"),  # ngay trên dải
        '{"id": "2506.00790", "versions": [',  # dòng bị cắt ngang
        kaggle_line("2507.00003", 3),
        kaggle_line("2508.00001"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    one = dict(discovery.iter_kaggle_rows(path, [("2506.00788", "2506.05787")]))
    assert sorted(one) == ["2506.00788", "2506.05787"] and len(one["2506.00788"]) == 2
    two = dict(
        discovery.iter_kaggle_rows(path, [("2506.00788", "2506.00788"), ("2507.00001", "2507.00009")])
    )
    assert sorted(two) == ["2506.00788", "2507.00003"]
    span = dict(discovery.iter_kaggle_rows(path, [("2506.05787", "2508.00001")]))
    assert sorted(span) == ["2506.05787", "2506.05788", "2507.00003", "2508.00001"]


def test_month_prefixes_cover_ranges_that_cross_months_and_years():
    assert discovery.month_prefixes([("2506.00788", "2506.05787")]) == [b"2506."]
    assert discovery.month_prefixes([("2511.00001", "2602.00005")]) == [
        b"2511.", b"2512.", b"2601.", b"2602.",
    ]  # fmt: skip
