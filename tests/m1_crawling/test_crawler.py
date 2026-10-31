"""Test cho src/m1_crawling/crawler.py (task P2-01).

Không gọi mạng thật: dùng phiên HTTP giả trả về archive dựng ngay trong bộ nhớ. Nhóm test:
(1) giải nén/lọc, (2) an toàn, (3) tải lịch sự + quy định arXiv, (4) vòng lặp, Checkpoint/Resume,
metadata, CLI.
"""

import gzip
import inspect
import io
import json
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

    def get(self, url, **kwargs):
        self.calls.append(url)
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