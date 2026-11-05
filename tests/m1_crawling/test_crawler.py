"""Test cho src/m1_crawling/crawler.py (task P2-01).

Không gọi mạng thật: dùng phiên HTTP giả trả về archive dựng ngay trong bộ nhớ. Nhóm test:
(1) giải nén/lọc, (2) an toàn, (3) tải lịch sự + quy định arXiv, (4) vòng lặp, Checkpoint/Resume,
metadata, CLI, (5) HTTP thật qua localhost, (6) API cho Runner, (7) các sửa đổi theo review PR
(retry lỗi mềm, redirect, n_versions, số liệu thời gian/RAM, lỗi metadata...).
"""

import gzip
import inspect
import io
import json
import os
import shutil
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests

from src.m1_crawling import crawler
from src.m1_crawling.crawler import (
    CrawlAbortedError,
    CrawlBlockedError,
    FetchError,
    PoliteFetcher,
    extract_tex_sources,
)

TEX = (
    b"\\documentclass{article}\n\\begin{document}\n\\begin{figure}\n"
    b"\\includegraphics{a.png}\n\\end{figure}\n\\end{document}\n"
)
BIB = b"@article{foo, title={Foo}}\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000


def make_tar_gz(members: dict[str, bytes], *, extra=None) -> bytes:
    """Dựng .tar.gz trong bộ nhớ. ``extra`` là hàm nhận TarFile để thêm member đặc biệt."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        if extra:
            extra(tar)
    return buffer.getvalue()


# ---------------------------------------------------------------- (1) Giải nén và lọc
def test_tar_gz_keeps_only_tex_and_bib_and_counts_sizes():
    data = make_tar_gz(
        {
            "main.tex": TEX,
            "refs.bib": BIB,
            "fig/a.png": PNG,
            "style.sty": b"\\ProvidesPackage{x}",
            "sections/intro.tex": b"Intro.",
        }
    )
    result = extract_tex_sources(data)
    assert result.status == "ok" and result.reason is None
    assert sorted(result.files) == ["main.tex", "refs.bib", "sections/intro.tex"]
    assert result.n_files_before == 5
    assert result.size_before == len(TEX) + len(BIB) + len(PNG) + len(b"\\ProvidesPackage{x}") + 6
    assert result.size_after == len(TEX) + len(BIB) + 6
    assert result.size_after < result.size_before


def test_figure_commands_are_kept_verbatim():
    result = extract_tex_sources(make_tar_gz({"main.tex": TEX, "a.png": PNG}))
    assert result.files["main.tex"] == TEX  # không sửa một byte nào
    assert b"\\begin{figure}" in result.files["main.tex"]
    assert b"\\includegraphics{a.png}" in result.files["main.tex"]


def test_images_never_enter_corpus():
    result = extract_tex_sources(make_tar_gz({"main.tex": TEX, "a.png": PNG, "b.jpg": PNG}))
    assert not any(name.endswith((".png", ".jpg", ".pdf", ".eps")) for name in result.files)


def test_single_gz_with_tex():
    result = extract_tex_sources(gzip.compress(TEX))
    assert result.status == "ok"
    assert result.files == {"main.tex": TEX}
    assert result.size_before == result.size_after == len(TEX)


def test_plain_tex_after_transport_decoding():
    # urllib3 có thể tự giải nén Content-Encoding gzip, khi đó nhận về văn bản thô.
    assert extract_tex_sources(TEX).files == {"main.tex": TEX}


def test_plain_tar_without_gzip():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("main.tex")
        info.size = len(TEX)
        tar.addfile(info, io.BytesIO(TEX))
    assert extract_tex_sources(buffer.getvalue()).status == "ok"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"%PDF-1.5\n" + b"x" * 100, "pdf_only"),
        (b"PK\x03\x04" + b"x" * 100, "word_or_zip"),
        (b"<!DOCTYPE html><html>No source</html>", "html_response"),
        (b"\x00\x01\x02binary", "not_tex"),
        (make_tar_gz({"a.png": PNG, "notes.txt": b"hi"}), "no_tex"),
        (make_tar_gz({"refs.bib": BIB}), "no_tex"),  # chỉ có .bib: không dùng được
    ],
)
def test_sources_without_tex_become_empty_records(payload, reason):
    result = extract_tex_sources(payload)
    assert result.status == "empty" and result.reason == reason
    assert result.files == {}


def test_corrupt_gzip_does_not_crash():
    broken = gzip.compress(TEX)[:-12]
    assert extract_tex_sources(broken).status == "empty"


# ---------------------------------------------------------------- (2) An toàn
def test_path_traversal_and_absolute_paths_are_rejected():
    data = make_tar_gz(
        {"main.tex": TEX, "../evil.tex": b"x", "/etc/passwd.tex": b"x", "a/../../b.tex": b"x"}
    )
    result = extract_tex_sources(data)
    assert list(result.files) == ["main.tex"]
    assert len(result.skipped) == 3


def test_symlinks_and_special_members_are_ignored():
    def add_special(tar: tarfile.TarFile) -> None:
        link = tarfile.TarInfo("link.tex")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)

    result = extract_tex_sources(make_tar_gz({"main.tex": TEX}, extra=add_special))
    assert list(result.files) == ["main.tex"]


def test_macos_junk_is_skipped():
    data = make_tar_gz({"main.tex": TEX, "__MACOSX/._main.tex": b"junk", "._x.tex": b"junk"})
    assert list(extract_tex_sources(data).files) == ["main.tex"]


def test_windows_invalid_characters_are_sanitised():
    result = extract_tex_sources(make_tar_gz({"main.tex": TEX, "we:ird?.tex": b"x"}))
    assert "we_ird_.tex" in result.files


def test_case_insensitive_duplicates_are_skipped():
    result = extract_tex_sources(make_tar_gz({"main.tex": TEX, "Main.tex": b"other"}))
    assert list(result.files) == ["main.tex"]
    assert result.skipped == ["Main.tex"]


def test_oversized_kept_file_is_skipped(monkeypatch):
    monkeypatch.setattr(crawler, "MAX_KEPT_FILE_BYTES", 10)
    result = extract_tex_sources(make_tar_gz({"main.tex": TEX, "small.tex": b"tiny"}))
    assert list(result.files) == ["small.tex"]
    assert result.skipped == ["main.tex"]


def test_decompression_bomb_is_refused(monkeypatch):
    monkeypatch.setattr(crawler, "MAX_UNCOMPRESSED_BYTES", 1000)
    result = extract_tex_sources(make_tar_gz({"big.tex": b"x" * 5000}))
    assert result.status == "empty" and result.reason == "too_large"
    assert extract_tex_sources(gzip.compress(b"\\" * 5000)).reason == "too_large"


def test_write_version_is_atomic_and_empty_version_creates_folder(tmp_path: Path):
    crawler.write_version(tmp_path, 1, {"main.tex": b"a", "sec/x.tex": b"b"})
    assert (tmp_path / "v1" / "sec" / "x.tex").read_bytes() == b"b"
    crawler.write_version(tmp_path, 2, {})
    assert (tmp_path / "v2").is_dir() and not any((tmp_path / "v2").iterdir())
    assert not list(tmp_path.glob(".v*.tmp"))
    crawler.write_version(tmp_path, 1, {"only.tex": b"c"})  # ghi đè sạch, không lẫn file cũ
    assert sorted(p.name for p in (tmp_path / "v1").iterdir()) == ["only.tex"]


# ---------------------------------------------------------------- (3) Tải lịch sự
class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {"Content-Type": "application/x-eprint-tar"}
        self.closed = False

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    """Phiên HTTP giả: ``script`` là hàm (url) -> FakeResponse | Exception."""

    def __init__(self, script):
        self.script = script
        self.headers = {}
        self.calls: list[str] = []
        self.kwargs: list[dict] = []  # tham số của từng lần gọi (vd allow_redirects)

    def get(self, url, **kwargs):
        self.calls.append(url)
        self.kwargs.append(kwargs)
        outcome = self.script(url, len(self.calls))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClock:
    """Đồng hồ giả: ``sleep`` chỉ cộng giờ, nên test chạy tức thì."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_fetcher(script, clock=None, delay=3.0, backoff=(10.0, 30.0, 60.0)):
    clock = clock or FakeClock()
    session = FakeSession(script)
    fetcher = PoliteFetcher(
        "UA-test (contact: a@b.c)",
        delay,
        session=session,
        sleep=clock.sleep,
        clock=clock.monotonic,
        backoff=backoff,
    )
    return fetcher, session, clock


def test_delay_below_three_seconds_is_refused():
    with pytest.raises(ValueError, match="3"):
        PoliteFetcher("UA", 2.9, session=FakeSession(lambda *_: FakeResponse()))


def test_user_agent_is_set_and_url_uses_export_domain():
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(body=b"x"))
    assert session.headers["User-Agent"] == "UA-test (contact: a@b.c)"
    assert crawler.SRC_URL.startswith("https://export.arxiv.org/src/")
    assert "//arxiv.org" not in crawler.SRC_URL
    fetcher.get(crawler.SRC_URL.format(arxiv_id="hep-th/9901001", version=2))
    assert session.calls == ["https://export.arxiv.org/src/hep-th/9901001v2"]


def test_consecutive_requests_are_at_least_three_seconds_apart():
    fetcher, _, clock = make_fetcher(lambda url, n: FakeResponse(body=b"x"))
    for _ in range(4):
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")
        clock.now += 0.5  # request tốn 0,5 giây
    assert len(clock.sleeps) == 3
    assert all(s == pytest.approx(2.5) for s in clock.sleeps)  # 0,5 + 2,5 = 3,0


def test_403_stops_immediately_without_retry():
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(status=403))
    with pytest.raises(CrawlBlockedError):
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert len(session.calls) == 1


def test_404_returns_empty_body_without_retry():
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(status=404))
    result = fetcher.get("https://export.arxiv.org/src/2506.00788v9")
    assert result.status_code == 404 and result.body == b""
    assert len(session.calls) == 1


def test_5xx_and_network_errors_are_retried_with_backoff_then_succeed():
    def script(url, n):
        return [
            FakeResponse(status=503),
            requests.ConnectionError("reset"),
            FakeResponse(body=b"ok"),
        ][n - 1]

    fetcher, session, clock = make_fetcher(script)
    result = fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert result.body == b"ok" and result.attempts == 3
    assert len(session.calls) == 3
    assert [s for s in clock.sleeps if s >= 10] == [10.0, 30.0]


def test_429_honours_retry_after():
    def script(url, n):
        if n == 1:
            return FakeResponse(status=429, headers={"Retry-After": "120"})
        return FakeResponse(body=b"ok")

    fetcher, _, clock = make_fetcher(script)
    fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert 120.0 in clock.sleeps


def test_gives_up_after_all_retries():
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(status=500))
    with pytest.raises(FetchError):
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert len(session.calls) == 4  # 1 lần đầu + 3 lần thử lại


def test_unexpected_status_is_not_retried():
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(status=400))
    with pytest.raises(FetchError):
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert len(session.calls) == 1


def test_oversized_download_is_refused(monkeypatch):
    monkeypatch.setattr(crawler, "MAX_DOWNLOAD_BYTES", 10)
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(body=b"x" * 100))
    with pytest.raises(FetchError):
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")


# ---------------------------------------------------------------- (4) Vòng lặp, resume, CLI
def discovery_rows():
    return [
        {
            "arxiv_id": "2506.00788",
            "n_versions": 2,
            "v1_date": "2025-06-01T12:00:00Z",
            "latest_version_date": "2025-06-09T08:00:00Z",
            "version_dates": {"v1": "2025-06-01T12:00:00Z", "v2": "2025-06-09T08:00:00Z"},
            "title": "Paper One",
            "authors": ["Nguyen, A", "B C"],
            "venue": None,
        },
        {
            "arxiv_id": "2506.00789",
            "n_versions": 1,
            "v1_date": "2025-06-02T00:00:00Z",
            "latest_version_date": "2025-06-02T00:00:00Z",
            "version_dates": {"v1": "2025-06-02T00:00:00Z"},
            "title": "PDF only",
            "authors": ["C"],
            "venue": "Nature",
        },
        {
            "arxiv_id": "hep-th/9901001",
            "n_versions": 1,
            "v1_date": "1999-01-04T00:00:00Z",
            "latest_version_date": "1999-01-04T00:00:00Z",
            "version_dates": {},
            "title": "Old id",
            "authors": [],
            "venue": None,
        },
    ]


def good_script(url, n):
    if "2506.00789" in url:
        return FakeResponse(body=b"%PDF-1.4 fake")
    return FakeResponse(body=make_tar_gz({"main.tex": TEX, "refs.bib": BIB, "fig.png": PNG}))


def run_crawl(tmp_path, script=good_script, papers=None, **kwargs):
    fetcher, session, clock = make_fetcher(script)
    summary = crawler.run(papers or discovery_rows(), tmp_path, fetcher, **kwargs)
    return summary, session, clock


def read_log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "crawl_log.jsonl").read_text().splitlines()]


def test_full_run_layout_matches_mock_and_logs_sizes(tmp_path: Path):
    summary, session, _ = run_crawl(tmp_path)
    assert len(session.calls) == 4  # 2 + 1 + 1 version
    assert (tmp_path / "2506-00788" / "v1" / "main.tex").read_bytes() == TEX
    assert (tmp_path / "2506-00788" / "v2" / "refs.bib").exists()
    assert not list(tmp_path.rglob("*.png"))
    assert (tmp_path / "hep-th_9901001" / "v1" / "main.tex").exists()
    # PDF-only: có folder v1 RỖNG ("record rỗng"), có metadata, không bị bỏ qua.
    pdf_v1 = tmp_path / "2506-00789" / "v1"
    assert pdf_v1.is_dir() and not any(pdf_v1.iterdir())
    assert (tmp_path / "2506-00789" / "metadata.json").exists()
    log = read_log(tmp_path)
    ok = next(r for r in log if r["arxiv_id"] == "2506.00788" and r["version"] == 1)
    assert ok["status"] == "ok" and ok["size_before"] > ok["size_after"] > 0
    assert ok["n_files_before"] == 3 and ok["n_files_after"] == 2
    empty = next(r for r in log if r["arxiv_id"] == "2506.00789")
    assert empty["status"] == "empty" and empty["reason"] == "pdf_only"
    assert summary["by_status"] == {"ok": 3, "empty": 1}
    assert summary["size_after_over_before"] < 1 and summary["exit_code"] == 0


def test_metadata_json_follows_paper_schema(tmp_path: Path):
    run_crawl(tmp_path)
    meta = json.loads((tmp_path / "2506-00788" / "metadata.json").read_text(encoding="utf-8"))
    assert meta == {
        "arxiv_id": "2506.00788",
        "title": "Paper One",
        "authors": ["Nguyen, A", "B C"],
        "submission_date": "2025-06-01",
        "revised_dates": ["2025-06-09"],
        "venue": None,
    }
    old = json.loads((tmp_path / "hep-th_9901001" / "metadata.json").read_text(encoding="utf-8"))
    assert old["arxiv_id"] == "hep-th/9901001" and old["revised_dates"] == []


def test_metadata_falls_back_to_latest_version_date_when_unpatched():
    row = {**discovery_rows()[0], "version_dates": {"v1": "2025-06-01T12:00:00Z"}}
    assert crawler.build_metadata(row)["revised_dates"] == ["2025-06-09"]


def test_can_skip_metadata(tmp_path: Path):
    run_crawl(tmp_path, write_metadata=False)
    assert not list(tmp_path.rglob("metadata.json"))


def test_resume_skips_finished_versions_and_makes_no_requests(tmp_path: Path):
    run_crawl(tmp_path)
    summary, session, _ = run_crawl(tmp_path)
    assert session.calls == []
    assert summary["by_status"] == {"ok": 3, "empty": 1}
    assert len(read_log(tmp_path)) == 4  # không ghi thêm dòng log


def test_resume_after_crash_continues_where_it_stopped(tmp_path: Path):
    def crashing(url, n):
        if n == 3:
            raise KeyboardInterrupt
        return good_script(url, n)

    fetcher, _, _ = make_fetcher(crashing)
    with pytest.raises(KeyboardInterrupt):
        crawler.run(discovery_rows(), tmp_path, fetcher)
    assert len(read_log(tmp_path)) == 2
    summary, session, _ = run_crawl(tmp_path)
    assert len(session.calls) == 2  # chỉ tải 2 version còn lại
    assert summary["by_status"] == {"ok": 3, "empty": 1}


def test_partial_log_line_from_a_crash_is_ignored(tmp_path: Path):
    run_crawl(tmp_path)
    with (tmp_path / "crawl_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"arxiv_id": "2506.00788", "vers')  # dòng cuối bị cắt ngang
    _, session, _ = run_crawl(tmp_path)
    assert session.calls == []


def test_resume_redownloads_if_folder_was_deleted(tmp_path: Path):
    run_crawl(tmp_path)
    shutil.rmtree(tmp_path / "2506-00788" / "v2")
    _, session, _ = run_crawl(tmp_path)
    assert [c.rsplit("/", 1)[-1] for c in session.calls] == ["2506.00788v2"]


def test_failed_versions_are_logged_and_retried_on_next_run(tmp_path: Path):
    def flaky(url, n):
        return FakeResponse(status=500) if "2506.00788v2" in url else good_script(url, n)

    summary, _, _ = run_crawl(tmp_path, script=flaky)
    assert summary["by_status"]["error"] == 1 and summary["exit_code"] == 1
    assert not (tmp_path / "2506-00788" / "v2").exists()  # không để folder ghi dở
    summary, session, _ = run_crawl(tmp_path)
    assert [c.rsplit("/", 1)[-1] for c in session.calls] == ["2506.00788v2"]
    assert summary["exit_code"] == 0


def test_http_404_becomes_empty_record(tmp_path: Path):
    def script(url, n):
        return FakeResponse(status=404)

    summary, _, _ = run_crawl(tmp_path, script=script, papers=discovery_rows()[:1])
    assert summary["empty_reasons"] == {"http_404": 2}
    assert (tmp_path / "2506-00788" / "v2").is_dir()


def test_blocked_run_stops_and_logs_without_hammering(tmp_path: Path):
    def script(url, n):
        return good_script(url, n) if n < 2 else FakeResponse(status=403)

    fetcher, session, _ = make_fetcher(script)
    with pytest.raises(CrawlBlockedError):
        crawler.run(discovery_rows(), tmp_path, fetcher)
    assert len(session.calls) == 2  # dừng ngay, không thử các version còn lại
    assert read_log(tmp_path)[-1]["status"] == "blocked"
    # Chạy lại: version bị chặn là non-terminal nên sẽ được thử lại, version xong thì bỏ qua.
    _, session2, _ = run_crawl(tmp_path)
    assert len(session2.calls) == 3


def test_too_many_consecutive_errors_aborts(tmp_path: Path):
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(status=500))
    with pytest.raises(CrawlAbortedError):
        crawler.run(discovery_rows(), tmp_path, fetcher, max_consecutive_errors=2)
    assert len(session.calls) == 8  # 2 version x (1 + 3 lần thử lại)


def test_select_papers_filters_id_range_and_sorts(tmp_path: Path):
    path = tmp_path / "d.jsonl"
    rows = discovery_rows()
    path.write_text(
        "\n".join(json.dumps(r) for r in reversed(rows)) + "\n{broken", encoding="utf-8"
    )
    picked = crawler.select_papers(path, "2506.00788", "2506.00789")
    assert [r["arxiv_id"] for r in picked] == ["2506.00788", "2506.00789"]
    assert len(crawler.select_papers(path, "0000.00000", "9999.99999", limit=1)) == 1


def test_cli_rejects_small_delay_and_missing_discovery(tmp_path: Path, capsys):
    base = ["--id_start", "2506.00788", "--id_end", "2506.00789", "--contact", "a@b.c"]
    assert crawler.main([*base, "--delay", "1", "--discovery", str(tmp_path / "x.jsonl")]) == 2
    assert crawler.main([*base, "--discovery", str(tmp_path / "x.jsonl")]) == 2
    assert "discovery" in capsys.readouterr().err


def test_cli_end_to_end_with_fake_network(tmp_path: Path, monkeypatch):
    discovery = tmp_path / "d.jsonl"
    discovery.write_text("\n".join(json.dumps(r) for r in discovery_rows()), encoding="utf-8")
    clock = FakeClock()
    session = FakeSession(good_script)
    real_init = PoliteFetcher.__init__

    def patched_init(self, user_agent, delay=3.0, **kwargs):
        real_init(
            self, user_agent, delay, session=session, sleep=clock.sleep, clock=clock.monotonic
        )

    monkeypatch.setattr(PoliteFetcher, "__init__", patched_init)
    out = tmp_path / "papers"
    code = crawler.main(
        ["--id-start", "2506.00788", "--id-end", "2506.00789", "--contact", "a@b.c",
         "--discovery", str(discovery), "--out", str(out)]
    )  # fmt: skip
    assert code == 0 and len(session.calls) == 3
    assert "a@b.c" in session.headers["User-Agent"]
    assert json.loads((out / "crawl_summary.json").read_text())["versions_expected"] == 3


# ---------------------------------------------------------------- (5) HTTP thật qua localhost
def _serve(handler_factory):
    server = HTTPServer(("127.0.0.1", 0), handler_factory)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_real_requests_handles_content_encoding_and_all_source_types():
    """arXiv hay gắn Content-Encoding: x-gzip lên .tar.gz/.gz; requests sẽ tự giải nén một lớp.

    Chạy bằng server HTTP thật trên localhost + thư viện requests thật để chắc rằng việc nhận
    diện theo nội dung vẫn đúng trong cả ba trường hợp.
    """
    tar_gz = make_tar_gz({"main.tex": TEX, "a.png": PNG})
    bodies = {
        "/src/tar_with_encoding": (tar_gz, "application/x-eprint-tar", "x-gzip"),
        "/src/gz_with_encoding": (gzip.compress(TEX), "application/x-eprint", "x-gzip"),
        "/src/tar_plain": (tar_gz, "application/x-eprint-tar", None),
        "/src/pdf": (b"%PDF-1.5 x", "application/pdf", None),
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload, ctype, encoding = bodies[self.path]
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            if encoding:
                self.send_header("Content-Encoding", encoding)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = _serve(Handler)
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        fetcher = PoliteFetcher("UA", 3.0, sleep=lambda _s: None)
        outcomes = {
            name: extract_tex_sources(fetcher.get(f"{base}/src/{name}").body)
            for name in ("tar_with_encoding", "gz_with_encoding", "tar_plain", "pdf")
        }
    finally:
        server.shutdown()
    assert outcomes["tar_with_encoding"].files == {"main.tex": TEX}
    assert outcomes["gz_with_encoding"].files == {"main.tex": TEX}
    assert outcomes["tar_plain"].files == {"main.tex": TEX}
    assert outcomes["pdf"].reason == "pdf_only"


def test_metadata_still_written_when_schema_module_is_unavailable(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.schemas", None)  # import src.schemas -> ImportError
    monkeypatch.setattr(crawler, "_schema_warned", False)
    run_crawl(tmp_path, papers=discovery_rows()[:1])
    meta = json.loads((tmp_path / "2506-00788" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["submission_date"] == "2025-06-01" and meta["authors"] == ["Nguyen, A", "B C"]


def test_metadata_is_written_before_any_download(tmp_path: Path):
    """Bị chặn (403) ngay request đầu tiên vẫn phải có metadata.json của bài đang xử lý."""
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(status=403))
    with pytest.raises(CrawlBlockedError):
        crawler.run(discovery_rows(), tmp_path, fetcher)
    assert (tmp_path / "2506-00788" / "metadata.json").exists()


# ---------------------------------------------------------------- (6) API cho Runner (stub P2-01)
def write_discovery(tmp_path: Path) -> Path:
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in discovery_rows()), encoding="utf-8")
    return path


def test_public_api_keeps_original_stub_signatures():
    paper = list(inspect.signature(crawler.crawl_paper).parameters)
    rng = list(inspect.signature(crawler.crawl_range).parameters)
    assert paper[:2] == ["arxiv_id", "out_dir"]
    assert rng[:4] == ["id_start", "id_end", "out_dir", "resume"]
    assert inspect.signature(crawler.crawl_range).parameters["resume"].default is True


def test_crawl_range_positional_like_the_stub(tmp_path: Path):
    fetcher, session, _ = make_fetcher(good_script)
    out = tmp_path / "out"
    summary = crawler.crawl_range(
        "2506.00788",
        "2506.00789",
        out,
        True,
        discovery_path=write_discovery(tmp_path),
        fetcher=fetcher,
    )
    assert summary["versions_expected"] == 3 and len(session.calls) == 3
    assert json.loads((out / "crawl_summary.json").read_text(encoding="utf-8"))["papers"] == 2


def test_crawl_range_resume_false_downloads_again(tmp_path: Path):
    discovery, out = write_discovery(tmp_path), tmp_path / "out"
    kwargs = {"discovery_path": discovery, "limit": 1}
    crawler.crawl_range(
        "2506.00788", "2506.00789", out, fetcher=make_fetcher(good_script)[0], **kwargs
    )
    again, session, _ = make_fetcher(good_script)
    crawler.crawl_range("2506.00788", "2506.00789", out, True, fetcher=again, **kwargs)
    assert session.calls == []
    fresh, session2, _ = make_fetcher(good_script)
    crawler.crawl_range("2506.00788", "2506.00789", out, False, fetcher=fresh, **kwargs)
    assert len(session2.calls) == 2  # bài 2506.00788 có 2 version, tải lại cả hai


def test_crawl_range_errors(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(crawler.CONTACT_ENV, raising=False)
    discovery = write_discovery(tmp_path)
    with pytest.raises(FileNotFoundError):
        crawler.crawl_range("2506.00788", "2506.00789", tmp_path, discovery_path=tmp_path / "no")
    with pytest.raises(ValueError, match="khong co bai nao"):
        crawler.crawl_range("9999.00001", "9999.00002", tmp_path, discovery_path=discovery)
    with pytest.raises(ValueError, match="email"):  # thiếu contact và biến môi trường
        crawler.crawl_range("2506.00788", "2506.00789", tmp_path, discovery_path=discovery)


def test_contact_can_come_from_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(crawler.CONTACT_ENV, "env@mail.test")
    assert crawler._require_contact(None) == "env@mail.test"
    assert crawler._require_contact("explicit@mail.test") == "explicit@mail.test"


def test_crawl_paper_looks_up_discovery_and_reports_versions(tmp_path: Path):
    fetcher, session, _ = make_fetcher(good_script)
    out = tmp_path / "out"
    summary = crawler.crawl_paper(
        "2506.00788", out, discovery_path=write_discovery(tmp_path), fetcher=fetcher
    )
    assert len(session.calls) == 2
    assert [v["status"] for v in summary["versions"]] == ["ok", "ok"]
    assert summary["versions"][0]["size_before"] > summary["versions"][0]["size_after"] > 0
    assert (out / "2506-00788" / "metadata.json").exists()


def test_crawl_paper_accepts_row_and_rejects_unknown_or_malformed_ids(tmp_path: Path):
    fetcher, session, _ = make_fetcher(good_script)
    row = discovery_rows()[2]  # ID cũ hep-th/9901001
    crawler.crawl_paper("hep-th/9901001", tmp_path / "o", row=row, fetcher=fetcher)
    assert session.calls == ["https://export.arxiv.org/src/hep-th/9901001v1"]
    with pytest.raises(ValueError, match="khong co trong discovery"):
        crawler.crawl_paper(
            "2506.99999", tmp_path, discovery_path=write_discovery(tmp_path), fetcher=fetcher
        )
    with pytest.raises(ValueError):
        crawler.crawl_paper("../etc", tmp_path, fetcher=fetcher)


def test_cli_requires_contact(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv(crawler.CONTACT_ENV, raising=False)
    base = [
        "--id_start",
        "2506.00788",
        "--id_end",
        "2506.00789",
        "--discovery",
        str(write_discovery(tmp_path)),
    ]
    assert crawler.main(base) == 2
    assert "contact" in capsys.readouterr().err

# ---------------------------------------------------------------- (7) Sửa đổi theo review PR
HTML_GZ = gzip.compress(b"<!DOCTYPE html><html>Service unavailable</html>")
ATOM = (
    '<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
    "<entry><id>http://arxiv.org/abs/2506.00788v2</id></entry></feed>"
)
EMPTY_ATOM = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def html_script(url, n):
    return FakeResponse(body=HTML_GZ)


def one_version_row():
    """Bài 2506.00789 (1 version) để test từng lượt thử lại."""
    return discovery_rows()[1:2]


# ---- #1: html_response / corrupt_archive được thử lại, rồi mới chốt empty
def test_html_response_is_retried_then_finalised_after_max_soft_retries(tmp_path: Path):
    v1 = tmp_path / "2506-00789" / "v1"
    for attempt in range(1, crawler.MAX_SOFT_RETRIES):
        summary, session, _ = run_crawl(tmp_path, script=html_script, papers=one_version_row())
        assert len(session.calls) == 1
        assert summary["by_status"] == {"error": 1} and summary["exit_code"] == 1
        assert not v1.exists()  # KHÔNG ghi thư mục v{n} khi còn được thử lại
        last = read_log(tmp_path)[-1]
        assert last["reason"] == "html_response" and last["soft_errors"] == attempt
    # Lượt cuối cùng: hết lượt thử, chốt empty và ghi record rỗng
    summary, _, _ = run_crawl(tmp_path, script=html_script, papers=one_version_row())
    assert summary["by_status"] == {"empty": 1}
    assert summary["empty_reasons"] == {"html_response": 1} and summary["exit_code"] == 0
    assert v1.is_dir() and not any(v1.iterdir())
    _, session, _ = run_crawl(tmp_path, script=html_script, papers=one_version_row())
    assert session.calls == []  # đã chốt, resume bỏ qua


def test_html_response_recovers_when_server_returns_real_source(tmp_path: Path):
    run_crawl(tmp_path, script=html_script, papers=one_version_row())
    good = lambda url, n: FakeResponse(body=make_tar_gz({"main.tex": TEX}))  # noqa: E731
    summary, _, _ = run_crawl(tmp_path, script=good, papers=one_version_row())
    assert summary["by_status"] == {"ok": 1}
    assert (tmp_path / "2506-00789" / "v1" / "main.tex").read_bytes() == TEX


def test_corrupt_archive_is_retried_not_finalised(tmp_path: Path):
    broken = make_tar_gz({"a.tex": TEX, "big.bin": os.urandom(300_000)})
    script = lambda url, n: FakeResponse(body=broken[: len(broken) // 2])  # noqa: E731
    summary, _, _ = run_crawl(tmp_path, script=script, papers=one_version_row())
    assert summary["by_status"] == {"error": 1}
    assert read_log(tmp_path)[-1]["reason"] == "corrupt_archive"
    assert not (tmp_path / "2506-00789" / "v1").exists()


def test_soft_error_counter_survives_an_unrelated_network_error(tmp_path: Path):
    run_crawl(tmp_path, script=html_script, papers=one_version_row())
    run_crawl(tmp_path, script=lambda url, n: FakeResponse(status=500), papers=one_version_row())
    assert read_log(tmp_path)[-1]["soft_errors"] == 1  # lỗi mạng không đưa bộ đếm về 0
    run_crawl(tmp_path, script=html_script, papers=one_version_row())
    assert read_log(tmp_path)[-1]["soft_errors"] == 2


def test_download_too_large_is_a_terminal_empty_record(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(crawler, "MAX_DOWNLOAD_BYTES", 10)
    script = lambda url, n: FakeResponse(body=b"x" * 100)  # noqa: E731
    summary, session, _ = run_crawl(tmp_path, script=script, papers=one_version_row())
    assert len(session.calls) == 1  # không thử lại
    assert summary["empty_reasons"] == {"too_large": 1} and summary["exit_code"] == 0
    assert read_log(tmp_path)[-1]["archive_bytes"] == 100
    assert (tmp_path / "2506-00789" / "v1").is_dir()
    _, session2, _ = run_crawl(tmp_path, script=script, papers=one_version_row())
    assert session2.calls == []


def test_too_large_error_is_a_fetch_error(monkeypatch):
    monkeypatch.setattr(crawler, "MAX_DOWNLOAD_BYTES", 10)
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(body=b"x" * 100))
    with pytest.raises(crawler.TooLargeError) as info:
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert isinstance(info.value, FetchError) and info.value.downloaded == 100


def test_short_download_versus_content_length_is_retried():
    def script(url, n):
        if n == 1:
            return FakeResponse(body=b"abc", headers={"Content-Length": "10"})
        return FakeResponse(body=b"0123456789", headers={"Content-Length": "10"})

    fetcher, session, _ = make_fetcher(script)
    assert fetcher.get("https://export.arxiv.org/src/2506.00788v1").body == b"0123456789"
    assert len(session.calls) == 2


def test_content_length_not_compared_when_body_was_decoded_and_raw_is_unknown():
    headers = {"Content-Length": "10", "Content-Encoding": "gzip"}
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(body=b"abc", headers=headers))
    assert fetcher.get("https://export.arxiv.org/src/2506.00788v1").body == b"abc"
    assert len(session.calls) == 1  # không đủ dữ kiện để so, nên không báo thiếu oan


def test_real_server_truncated_body_is_retried_then_gives_up():
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"only ten b")  # hứa 100 byte nhưng chỉ gửi 10

        def log_message(self, *args):
            pass

    server = _serve(Handler)
    try:
        fetcher = PoliteFetcher("UA", 3.0, sleep=lambda _s: None, backoff=(0.0, 0.0, 0.0))
        with pytest.raises(FetchError):
            fetcher.get(f"http://127.0.0.1:{server.server_address[1]}/src/x")
    finally:
        server.shutdown()
    assert len(hits) == 4  # 1 lần đầu + 3 lần thử lại


# ---- #2: lỗi bất ngờ ở một version không giết cả pipeline
def test_write_version_failure_is_logged_and_run_continues(tmp_path: Path, monkeypatch):
    real = crawler.write_version

    def flaky_write(folder, version, files):
        if folder.name == "2506-00788" and version == 1:
            raise OSError("disk full")
        return real(folder, version, files)

    monkeypatch.setattr(crawler, "write_version", flaky_write)
    summary, session, _ = run_crawl(tmp_path)
    assert len(session.calls) == 4  # vẫn đi tiếp qua các version và bài còn lại
    assert summary["by_status"] == {"ok": 2, "empty": 1, "error": 1} and summary["exit_code"] == 1
    bad = next(r for r in read_log(tmp_path) if r["status"] == "error")
    assert bad["reason"].startswith("OSError") and "disk full" in bad["reason"]
    assert not (tmp_path / "2506-00788" / "v1").exists()
    monkeypatch.setattr(crawler, "write_version", real)  # lần chạy sau tự hồi phục
    _, session2, _ = run_crawl(tmp_path)
    assert [c.rsplit("/", 1)[-1] for c in session2.calls] == ["2506.00788v1"]


def test_extract_failure_is_logged_with_exception_type_and_run_continues(
    tmp_path: Path, monkeypatch
):
    real = crawler.extract_tex_sources
    calls = []

    def flaky_extract(data):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("giai nen qua lon")
        return real(data)

    monkeypatch.setattr(crawler, "extract_tex_sources", flaky_extract)
    summary, session, _ = run_crawl(tmp_path)
    assert len(session.calls) == 4 and summary["by_status"]["error"] == 1
    assert read_log(tmp_path)[0]["reason"].startswith("MemoryError")


def test_run_safety_net_catches_a_crash_of_crawl_version_itself(tmp_path: Path, monkeypatch):
    real = crawler.crawl_version

    def exploding(fetcher, row, version, folder, log_path, **kwargs):
        if row["arxiv_id"] == "2506.00788" and version == 1:
            raise RuntimeError("boom")
        return real(fetcher, row, version, folder, log_path, **kwargs)

    monkeypatch.setattr(crawler, "crawl_version", exploding)
    summary, session, _ = run_crawl(tmp_path)
    assert len(session.calls) == 3 and summary["by_status"]["error"] == 1
    assert read_log(tmp_path)[0]["reason"] == "RuntimeError: boom"


# ---- #3: n_versions thiếu/sai
@pytest.mark.parametrize("bad", [None, 0, -1, "abc", True])
def test_n_versions_invalid_values_raise(bad):
    with pytest.raises(ValueError, match="n_versions"):
        crawler._n_versions({"arxiv_id": "2506.00001", "n_versions": bad})


def test_n_versions_missing_key_raises_and_valid_values_pass():
    with pytest.raises(ValueError, match="2506.00001"):
        crawler._n_versions({"arxiv_id": "2506.00001"})
    assert crawler._n_versions({"arxiv_id": "x", "n_versions": 4}) == 4
    assert crawler._n_versions({"arxiv_id": "x", "n_versions": "3"}) == 3


def test_run_refuses_before_any_request_when_n_versions_is_missing(tmp_path: Path):
    rows = discovery_rows()
    del rows[0]["n_versions"]
    fetcher, session, _ = make_fetcher(good_script)
    with pytest.raises(ValueError, match="2506.00788"):
        crawler.run(rows, tmp_path, fetcher, api_fallback=False)
    assert session.calls == [] and list(tmp_path.iterdir()) == []  # chưa tải, chưa tạo gì


def test_missing_n_versions_is_resolved_through_the_api(tmp_path: Path):
    rows = discovery_rows()
    del rows[0]["n_versions"]

    def script(url, n):
        if "/api/query" in url:
            assert "id_list=2506.00788" in url and "max_results=1" in url
            return FakeResponse(body=ATOM.encode(), headers={"Content-Type": "application/atom+xml"})
        return good_script(url, n)

    summary, session, _ = run_crawl(tmp_path, script=script, papers=rows)
    assert session.calls[0].startswith("https://export.arxiv.org/api/query")
    assert summary["n_versions_from_api"] == ["2506.00788"]
    assert summary["by_status"] == {"ok": 3, "empty": 1}  # đủ v1 và v2 của bài đó


def test_api_that_does_not_know_the_paper_still_raises(tmp_path: Path):
    rows = discovery_rows()
    del rows[0]["n_versions"]

    def script(url, n):
        return FakeResponse(body=EMPTY_ATOM.encode()) if "/api/query" in url else good_script(url, n)

    fetcher, _, _ = make_fetcher(script)
    with pytest.raises(ValueError, match="2506.00788"):
        crawler.run(rows, tmp_path, fetcher)


def test_api_lookup_is_batched_and_goes_through_the_polite_fetcher():
    ids = [f"2506.{i:05d}" for i in range(150)]
    fetcher, session, clock = make_fetcher(lambda url, n: FakeResponse(body=EMPTY_ATOM.encode()))
    assert crawler.fetch_n_versions_from_api(fetcher, ids) == {}
    assert len(session.calls) == 2  # 100 + 50
    assert session.calls[1].count(",") == 49
    assert all(s == pytest.approx(3.0) for s in clock.sleeps)  # vẫn cách nhau 3 giây


def test_api_403_still_stops_everything():
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(status=403))
    with pytest.raises(CrawlBlockedError):
        crawler.fetch_n_versions_from_api(fetcher, ["2506.00788"])


def test_find_ids_not_in_discovery_reports_gaps():
    rows = [{"arxiv_id": "2506.00788"}, {"arxiv_id": "2506.00790"}]
    assert crawler.find_ids_not_in_discovery(rows, "2506.00788", "2506.00791") == [
        "2506.00789",
        "2506.00791",
    ]
    assert crawler.find_ids_not_in_discovery(rows, "2506.00788", "2507.00001") is None  # 2 tháng


def test_crawl_range_summary_lists_ids_missing_from_discovery(tmp_path: Path):
    out = tmp_path / "out"
    fetcher, _, _ = make_fetcher(good_script)
    summary = crawler.crawl_range(
        "2506.00788", "2506.00791", out, discovery_path=write_discovery(tmp_path), fetcher=fetcher
    )
    assert summary["ids_not_in_discovery"] == ["2506.00790", "2506.00791"]
    assert summary["ids_not_in_discovery_count"] == 2
    saved = json.loads((out / "crawl_summary.json").read_text(encoding="utf-8"))
    assert saved["ids_not_in_discovery"] == ["2506.00790", "2506.00791"]


def test_coverage_is_skipped_when_limit_cuts_the_selection(tmp_path: Path):
    fetcher, _, _ = make_fetcher(good_script)
    summary = crawler.crawl_range(
        "2506.00788",
        "2506.00791",
        tmp_path / "out",
        discovery_path=write_discovery(tmp_path),
        limit=1,
        fetcher=fetcher,
    )
    assert summary["ids_not_in_discovery"] is None


# ---- #4, #5: RAM và thời gian
def test_peak_and_current_rss_are_sane():
    peak, current = crawler._peak_rss_mb(), crawler._current_rss_mb()
    assert peak is None or peak > 0
    assert current is None or current > 0
    if peak is not None and current is not None:
        assert peak + 1 >= current  # đỉnh không thể thấp hơn mức hiện tại


def test_log_and_summary_have_wall_seconds_and_rss_that_survive_resume(tmp_path: Path):
    summary, _, _ = run_crawl(tmp_path)
    assert all("wall_seconds" in r and "rss_mb" in r for r in read_log(tmp_path))
    assert summary["versions_with_wall_seconds"] == 4
    assert set(summary["seconds_per_paper"]) == {"2506.00788", "2506.00789", "hep-th/9901001"}
    assert summary["seconds_per_paper_max"] >= summary["seconds_per_paper_mean"] >= 0
    assert "rss_mb_mean" in summary and "peak_rss_mb" in summary
    resumed, session, _ = run_crawl(tmp_path)  # không tải gì, nhưng số liệu phải y nguyên
    assert session.calls == []
    assert resumed["wall_seconds_total"] == summary["wall_seconds_total"]
    assert resumed["seconds_per_paper"] == summary["seconds_per_paper"]


# ---- #5: nhận diện nội dung không phải TeX
@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (gzip.compress(b"%!PS-Adobe-3.0\n\\def\\x{1}\n"), "not_tex"),  # PostScript có dấu "\"
        (gzip.compress(b"%PDF-1.4\n\\begin{x}"), "pdf_only"),  # PDF bọc gzip
        (b"\x89PNG\r\n\x1a\n\\begin{document}", "not_tex"),  # ảnh PNG
        (b"plain text with a \\ backslash and nothing TeX-like", "not_tex"),
    ],
)
def test_non_tex_single_files_are_rejected(payload, reason):
    result = extract_tex_sources(payload)
    assert result.status == "empty" and result.reason == reason and result.files == {}


def test_plain_tex_without_latex_preamble_is_still_accepted():
    assert extract_tex_sources(b"\\def\\a{1}\n\\bye\n").status == "ok"


# ---- #6: tar hỏng giữa chừng là corrupt_archive, không phải not_tex
def test_truncated_tar_gz_is_corrupt_archive_not_not_tex():
    data = make_tar_gz({"main.tex": TEX, "big.bin": os.urandom(300_000), "z.tex": TEX})
    result = extract_tex_sources(data[: len(data) // 2])
    assert result.status == "empty" and result.reason == "corrupt_archive"


def test_plain_tar_cut_in_the_middle_of_a_member_is_corrupt_archive():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name in ("a.tex", "b.tex"):
            info = tarfile.TarInfo(name)
            info.size = 5000
            tar.addfile(info, io.BytesIO(b"\\begin{x}" + b"y" * 4991))
    data = buffer.getvalue()
    cut = 512 + 5120 + 512 + 100  # hết member a, mới đọc được 100 byte của member b
    assert extract_tex_sources(data[:cut]).reason == "corrupt_archive"
    assert extract_tex_sources(data).status == "ok"  # nguyên vẹn thì vẫn đọc được


# ---- #8: redirect
def test_redirects_are_never_followed_automatically():
    fetcher, session, _ = make_fetcher(lambda url, n: FakeResponse(body=b"x"))
    fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert session.kwargs[0]["allow_redirects"] is False


def test_redirect_to_another_host_is_refused_and_never_requested():
    location = "https://arxiv.org/src/2506.00788v1"
    script = lambda url, n: FakeResponse(status=301, headers={"Location": location})  # noqa: E731
    fetcher, session, _ = make_fetcher(script)
    with pytest.raises(crawler.ForbiddenRedirectError, match="arxiv.org"):
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert session.calls == ["https://export.arxiv.org/src/2506.00788v1"]  # không chạm arxiv.org


def test_redirect_without_location_is_refused():
    fetcher, _, _ = make_fetcher(lambda url, n: FakeResponse(status=302))
    with pytest.raises(crawler.ForbiddenRedirectError):
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")


def test_redirect_within_export_host_is_followed_and_waits_the_delay():
    def script(url, n):
        if n == 1:
            return FakeResponse(status=301, headers={"Location": "/src/new/2506.00788v1"})
        return FakeResponse(body=b"ok")

    fetcher, session, clock = make_fetcher(script)
    assert fetcher.get("https://export.arxiv.org/src/2506.00788v1").body == b"ok"
    assert session.calls[1] == "https://export.arxiv.org/src/new/2506.00788v1"
    assert clock.sleeps == [pytest.approx(3.0)]  # bước nhảy cũng cách request trước 3 giây


def test_endless_redirects_give_up():
    script = lambda url, n: FakeResponse(status=302, headers={"Location": "/src/loop"})  # noqa: E731
    fetcher, session, _ = make_fetcher(script)
    with pytest.raises(FetchError, match="redirect"):
        fetcher.get("https://export.arxiv.org/src/2506.00788v1")
    assert len(session.calls) == crawler.MAX_REDIRECTS + 1


def test_forbidden_redirect_becomes_an_error_record_not_a_crash(tmp_path: Path):
    location = "https://arxiv.org/src/2506.00789v1"
    script = lambda url, n: FakeResponse(status=302, headers={"Location": location})  # noqa: E731
    summary, session, _ = run_crawl(tmp_path, script=script, papers=one_version_row())
    assert len(session.calls) == 1 and summary["by_status"] == {"error": 1}
    assert "redirect" in read_log(tmp_path)[-1]["reason"]


def test_real_server_redirect_to_arxiv_org_is_not_followed():
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(302)
            self.send_header("Location", "https://arxiv.org/src/2506.00788v1")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = _serve(Handler)
    try:
        fetcher = PoliteFetcher("UA", 3.0, sleep=lambda _s: None)
        with pytest.raises(crawler.ForbiddenRedirectError):
            fetcher.get(f"http://127.0.0.1:{server.server_address[1]}/src/2506.00788v1")
    finally:
        server.shutdown()
    assert hits == ["/src/2506.00788v1"]  # requests thật cũng không tự theo redirect


# ---- #7, #9: lỗi metadata và ngày revise thiếu
def test_invalid_metadata_is_reported_in_summary_and_file_and_paper_is_still_crawled(
    tmp_path: Path, monkeypatch
):
    real = crawler.build_metadata

    def bad(row):
        if row["arxiv_id"] == "2506.00788":
            raise ValueError("authors rong")
        return real(row)

    monkeypatch.setattr(crawler, "build_metadata", bad)
    summary, session, _ = run_crawl(tmp_path)
    assert summary["metadata_errors"] == [
        {"arxiv_id": "2506.00788", "error": "ValueError: authors rong"}
    ]
    assert not (tmp_path / "2506-00788" / "metadata.json").exists()
    assert len(session.calls) == 4  # bài vẫn được crawl
    saved = json.loads((tmp_path / crawler.METADATA_ERRORS_NAME).read_text(encoding="utf-8"))
    assert saved[0]["arxiv_id"] == "2506.00788"


def test_no_metadata_errors_file_when_everything_is_fine(tmp_path: Path):
    summary, _, _ = run_crawl(tmp_path)
    assert summary["metadata_errors"] == []
    assert not (tmp_path / crawler.METADATA_ERRORS_NAME).exists()


def test_missing_middle_version_date_is_flagged_not_silently_shifted(tmp_path: Path):
    row = {
        **discovery_rows()[0],
        "n_versions": 4,
        "version_dates": {
            "v1": "2025-06-01T12:00:00Z",
            "v2": "2025-06-03T00:00:00Z",
            "v4": "2025-06-09T08:00:00Z",  # thiếu v3
        },
    }
    assert crawler._revised_dates(row) == (["2025-06-03", "2025-06-09"], [3])
    summary, _, _ = run_crawl(tmp_path, papers=[row])
    assert summary["revised_dates_incomplete"] == [
        {"arxiv_id": "2506.00788", "missing_versions": [3]}
    ]
    complete, _, _ = run_crawl(tmp_path / "ok", papers=discovery_rows()[:1])
    assert complete["revised_dates_incomplete"] == []


# ---- CLI
def test_cli_exits_2_when_discovery_lacks_n_versions(tmp_path: Path, capsys):
    rows = discovery_rows()
    del rows[0]["n_versions"]
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    code = crawler.main(
        ["--id_start", "2506.00788", "--id_end", "2506.00789", "--contact", "a@b.c",
         "--discovery", str(path), "--out", str(tmp_path / "o"), "--no_api_fallback"]
    )  # fmt: skip
    assert code == 2 and "n_versions" in capsys.readouterr().err


def test_printable_summary_shortens_long_collections():
    shown = crawler._printable({"short": [1, 2], "long": list(range(50)), "n": 3})
    assert shown["short"] == [1, 2] and shown["n"] == 3
    assert "50" in shown["long"] and crawler.SUMMARY_NAME in shown["long"]