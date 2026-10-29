"""Unit tests cho src/m1_crawling/s2_api.py (không gọi mạng thật, dùng tests/mock_data)."""

import threading
import time
from pathlib import Path

import pytest
import requests

from src.common.utils_io import read_json, write_json
from src.m1_crawling.s2_api import (
    BATCH_SIZE,
    S2Client,
    S2Error,
    S2Reference,
    S2Settings,
    count_saved_references,
    crawl_references,
    crawl_references_batch,
    fetch_references,
    fetch_references_batch,
    parse_reference,
    run_s2_stage,
    to_references_json,
)

MOCK_DIR = Path(__file__).resolve().parents[1] / "mock_data"


class FakeResponse:
    """Response giả, đủ các thuộc tính mà S2Client dùng."""

    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Session giả: trả lần lượt các phần tử của `responses`; phần tử là Exception thì ném ra."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        return self._next(url, **kwargs)

    def post(self, url, **kwargs):
        return self._next(url, **kwargs)


class FakeClock:
    """Đồng hồ giả: sleep chỉ cộng thời gian ảo, nên test chạy tức thì và chính xác."""

    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(responses, delay=0.0, max_retries=5, api_key="test-key"):
    """Tạo client dùng session và đồng hồ giả; trả (client, session, clock)."""
    session, clock = FakeSession(responses), FakeClock()
    settings = S2Settings(delay_seconds=delay, max_retries=max_retries, api_key=api_key)
    client = S2Client(settings, session=session, sleep=clock.sleep, clock=clock.monotonic)
    return client, session, clock


def cited(paper_id, arxiv=None, title="T", authors=("A",), **extra):
    """Dựng một phần tử `data` của response S2 (GET /references)."""
    external = {"ArXiv": arxiv} if arxiv else {}
    paper = {
        "paperId": paper_id,
        "title": title,
        "authors": [{"authorId": "1", "name": name} for name in authors],
        "externalIds": external,
        **extra,
    }
    return {"citedPaper": paper}


def batch_ref(paper_id, arxiv=None, title="T", authors=("A",), **extra):
    """Dựng một phần tử trong danh sách ``references`` của batch response."""
    external = {"ArXiv": arxiv} if arxiv else {}
    return {
        "paperId": paper_id,
        "title": title,
        "authors": [{"authorId": "1", "name": name} for name in authors],
        "externalIds": external,
        **extra,
    }


def batch_paper(arxiv_id_key, refs):
    """Dựng một phần tử trong list response của POST /paper/batch."""
    return {"paperId": f"s2-{arxiv_id_key}", "references": refs}


# ---------- parse_reference ----------


def test_parse_reference_survives_null_external_ids():
    ref = parse_reference({"paperId": "p1", "title": "X", "externalIds": None, "authors": None})

    assert ref is not None
    assert ref.arxiv_id is None
    assert ref.authors == []


def test_parse_reference_returns_none_for_empty_record():
    assert parse_reference(None) is None
    assert parse_reference({}) is None


def test_parse_reference_strips_version_suffix():
    ref = parse_reference({"paperId": "p1", "externalIds": {"ArXiv": "1706.03762v5"}})

    assert ref.arxiv_id == "1706.03762"


def test_parse_reference_keeps_old_style_arxiv_id():
    ref = parse_reference({"paperId": "p1", "externalIds": {"ArXiv": "hep-th/9503124"}})

    assert ref.arxiv_id == "hep-th/9503124"


def test_parse_reference_ignores_malformed_arxiv_id():
    ref = parse_reference({"paperId": "p1", "externalIds": {"ArXiv": "not-an-id"}})

    assert ref is not None
    assert ref.arxiv_id is None


def test_parse_reference_drops_authors_without_name():
    cited_paper = {"paperId": "p1", "authors": [{"authorId": "1", "name": None}, {"name": "B"}]}

    assert parse_reference(cited_paper).authors == ["B"]


# ---------- S2Client.get_json ----------


def test_get_json_sends_api_key_header():
    client, session, _ = make_client([FakeResponse(200, {"ok": 1})], api_key="secret")

    assert client.get_json("/x", {}) == {"ok": 1}
    assert session.calls[0][1]["headers"] == {"x-api-key": "secret"}
    assert session.calls[0][1]["timeout"] > 0


def test_get_json_omits_header_without_api_key(monkeypatch):
    monkeypatch.delenv("S2_API_KEY", raising=False)
    client, session, _ = make_client([FakeResponse(200, {})], api_key=None)

    client.get_json("/x", {})

    assert session.calls[0][1]["headers"] == {}


def test_get_json_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "from-env")
    client, session, _ = make_client([FakeResponse(200, {})], api_key=None)

    client.get_json("/x", {})

    assert session.calls[0][1]["headers"] == {"x-api-key": "from-env"}


def test_get_json_retries_on_429_and_honours_retry_after():
    responses = [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"ok": 1})]
    client, session, clock = make_client(responses)

    result = client.get_json("/x", {})

    assert result == {"ok": 1}
    assert len(session.calls) == 2
    assert 7.0 in clock.sleeps


def test_get_json_uses_exponential_backoff_without_retry_after():
    responses = [FakeResponse(503), FakeResponse(503), FakeResponse(200, {"ok": 1})]
    client, _, clock = make_client(responses)

    client.get_json("/x", {})

    assert clock.sleeps == [2.0, 4.0]


def test_get_json_retries_after_timeout():
    client, session, _ = make_client([requests.Timeout(), FakeResponse(200, {"ok": 1})])

    assert client.get_json("/x", {}) == {"ok": 1}
    assert len(session.calls) == 2


def test_get_json_returns_none_on_404_without_retry():
    client, session, _ = make_client([FakeResponse(404)])

    assert client.get_json("/x", {}) is None
    assert len(session.calls) == 1


def test_get_json_raises_without_retry_on_client_error():
    client, session, _ = make_client([FakeResponse(400)])

    with pytest.raises(requests.HTTPError):
        client.get_json("/x", {})
    assert len(session.calls) == 1


def test_get_json_raises_s2error_after_exhausting_retries():
    client, session, _ = make_client([FakeResponse(429)] * 3, max_retries=3)

    with pytest.raises(S2Error):
        client.get_json("/x", {})
    assert len(session.calls) == 3


class BadJsonResponse(FakeResponse):
    """Response 200 nhưng thân không phải JSON hợp lệ."""

    def json(self):
        raise ValueError("Expecting value")


def test_get_json_does_not_sleep_after_the_last_attempt():
    client, _, clock = make_client([FakeResponse(429)] * 3, max_retries=3)

    with pytest.raises(S2Error):
        client.get_json("/x", {})

    # Chỉ ngủ giữa các lần thử (2s, 4s); không ngủ 8s sau lần thứ ba vì không còn lần nào nữa.
    assert clock.sleeps == [2.0, 4.0]


def test_get_json_does_not_sleep_after_the_last_network_error():
    client, _, clock = make_client([requests.Timeout(), requests.Timeout()], max_retries=2)

    with pytest.raises(S2Error):
        client.get_json("/x", {})

    assert clock.sleeps == [2.0]


def test_get_json_retries_when_body_is_not_valid_json():
    client, session, clock = make_client([BadJsonResponse(200), FakeResponse(200, {"ok": 1})])

    assert client.get_json("/x", {}) == {"ok": 1}
    assert len(session.calls) == 2
    assert clock.sleeps == [2.0]


def test_get_json_raises_s2error_when_body_is_always_invalid_json():
    client, session, clock = make_client([BadJsonResponse(200)] * 2, max_retries=2)

    with pytest.raises(S2Error, match="JSON hỏng"):
        client.get_json("/x", {})
    assert len(session.calls) == 2
    assert clock.sleeps == [2.0]


def test_get_json_enforces_delay_between_requests():
    client, _, clock = make_client([FakeResponse(200, {}), FakeResponse(200, {})], delay=1.0)

    client.get_json("/x", {})
    client.get_json("/x", {})

    # Request đầu không phải chờ; request thứ hai gọi ngay nên phải chờ đủ 1 giây.
    assert clock.sleeps == [1.0]


def test_get_json_counts_processing_time_towards_delay():
    client, _, clock = make_client([FakeResponse(200, {}), FakeResponse(200, {})], delay=1.0)

    client.get_json("/x", {})
    clock.now += 0.4  # giả lập thời gian xử lý giữa hai request
    client.get_json("/x", {})

    assert clock.sleeps == [pytest.approx(0.6)]


# ---------- fetch_references ----------


def test_fetch_references_follows_pagination():
    page1 = {"offset": 0, "next": 2, "data": [cited("p1", "1706.03762"), cited("p2")]}
    page2 = {"offset": 2, "data": [cited("p3", "1810.04805")]}
    client, session, _ = make_client([FakeResponse(200, page1), FakeResponse(200, page2)])

    refs = fetch_references("2310.12345", client)

    assert [r.s2_id for r in refs] == ["p1", "p2", "p3"]
    assert [call[1]["params"]["offset"] for call in session.calls] == [0, 2]


def test_fetch_references_returns_empty_list_when_paper_has_no_references():
    client, _, _ = make_client([FakeResponse(200, {"offset": 0, "data": []})])

    assert fetch_references("2310.12345", client) == []


def test_fetch_references_returns_none_when_paper_not_in_s2():
    client, _, _ = make_client([FakeResponse(404)])

    assert fetch_references("2310.12345", client) is None


def test_fetch_references_skips_missing_cited_paper_but_keeps_sparse_one():
    page = {
        "data": [
            {"citedPaper": {"paperId": None, "title": None, "externalIds": None}},
            {"citedPaper": None},
        ]
    }
    client, _, _ = make_client([FakeResponse(200, page)])

    refs = fetch_references("2310.12345", client)

    assert len(refs) == 1  # bản ghi chỉ toàn None vẫn là dict khác rỗng nên được giữ
    assert refs[0].s2_id is None


def test_fetch_references_stops_if_next_does_not_advance():
    stuck = {"offset": 0, "next": 0, "data": [cited("p1")]}
    client, session, _ = make_client([FakeResponse(200, stuck)])

    assert len(fetch_references("2310.12345", client)) == 1
    assert len(session.calls) == 1


def test_fetch_references_rejects_invalid_id_without_calling_s2():
    client, session, _ = make_client([])

    with pytest.raises(ValueError):
        fetch_references("2310-12345", client)
    assert session.calls == []


def test_fetch_references_keeps_slash_of_old_style_id():
    client, session, _ = make_client([FakeResponse(200, {"data": []})])

    fetch_references("hep-th/9901001", client)

    assert session.calls[0][0].endswith("/paper/arXiv:hep-th/9901001/references")
    assert len(session.calls) == 1


def test_fetch_references_makes_single_attempt_for_new_style_id_on_404():
    client, session, _ = make_client([FakeResponse(404)])

    assert fetch_references("2310.12345", client) is None
    assert len(session.calls) == 1


# ---------- to_references_json ----------


def test_to_references_json_keeps_only_arxiv_refs():
    refs = [
        S2Reference(s2_id="p1", arxiv_id="1706.03762", title="Attention"),
        S2Reference(s2_id="p2", title="Một cuốn sách"),
    ]

    result = to_references_json("2310.12345", refs)

    assert list(result) == ["1706-03762"]


def test_to_references_json_matches_mock_format():
    refs = [
        S2Reference(
            s2_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            arxiv_id="1706.03762",
            title="Attention Is All You Need",
            authors=["Vaswani"],
            publication_date="2017-06-12",
        )
    ]

    result = to_references_json("2310.12345", refs)

    assert result == read_json(MOCK_DIR / "2310-12345" / "references.json")


def test_to_references_json_matches_mock_format_for_old_style_ids():
    refs = [
        S2Reference(
            s2_id="0000000000000000000000000000000000000000",
            arxiv_id="hep-th/9503124",
            title="String Theory Dynamics In Various Dimensions",
            authors=["Edward Witten"],
            publication_date="1995-03-17",
        )
    ]

    result = to_references_json("hep-th/9901001", refs)

    assert result == read_json(MOCK_DIR / "hep-th_9901001" / "references.json")


# ---------- crawl_references ----------


def test_crawl_references_writes_both_files(tmp_path):
    page = {"data": [cited("p1", "1706.03762"), cited("p2", title="Sách không có arXiv")]}
    client, _, _ = make_client([FakeResponse(200, page)])

    result = crawl_references("2310.12345", tmp_path, client)

    assert (result.status, result.n_refs, result.n_refs_co_arxiv_id) == ("done", 2, 1)
    assert list(read_json(tmp_path / "references.json")) == ["1706-03762"]
    everything = read_json(tmp_path / "references_all.json")
    assert [row["s2_id"] for row in everything] == ["p1", "p2"]
    assert everything[1]["arxiv_id"] is None


def test_crawl_references_skips_finished_paper_when_resuming(tmp_path):
    (tmp_path / "references.json").write_text("{}", encoding="utf-8")
    client, session, _ = make_client([])

    result = crawl_references("2310.12345", tmp_path, client, resume=True)

    assert result.status == "skipped"
    assert session.calls == []


def test_crawl_references_redoes_paper_without_resume(tmp_path):
    (tmp_path / "references.json").write_text("{}", encoding="utf-8")
    client, session, _ = make_client([FakeResponse(200, {"data": [cited("p1", "1706.03762")]})])

    result = crawl_references("2310.12345", tmp_path, client, resume=False)

    assert result.status == "done"
    assert len(session.calls) == 1
    assert list(read_json(tmp_path / "references.json")) == ["1706-03762"]


def test_crawl_references_reports_not_found_and_writes_nothing(tmp_path):
    client, _, _ = make_client([FakeResponse(404)])

    result = crawl_references("2310.12345", tmp_path, client)

    assert result.status == "not_found"
    assert not list(tmp_path.iterdir())


# ---------- S2Settings ----------


def test_settings_from_config_reads_s2_section(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "s2_api:\n  delay_seconds: 2\n  timeout: 10\n  max_retries: 7\nml:\n  top_k: 5\n",
        encoding="utf-8",
    )

    settings = S2Settings.from_config(config)

    assert (settings.delay_seconds, settings.timeout, settings.max_retries) == (2, 10, 7)


def test_settings_from_config_uses_defaults_for_missing_keys(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("s2_api:\n  delay_seconds: 1\n", encoding="utf-8")

    settings = S2Settings.from_config(config)

    assert settings.max_retries == 5
    assert settings.base_url.startswith("https://api.semanticscholar.org")


def test_settings_from_real_project_config():
    settings = S2Settings.from_config(Path(__file__).resolve().parents[2] / "configs/config.yaml")

    assert settings.delay_seconds >= 1


# ---------- thread-safety của S2Client ----------


def test_shared_client_keeps_delay_between_threads():
    delay, n_threads = 0.05, 4
    session = FakeSession([FakeResponse(200, {})] * n_threads)
    client = S2Client(S2Settings(delay_seconds=delay, api_key="k"), session=session)
    threads = [threading.Thread(target=client.get_json, args=("/x", {})) for _ in range(n_threads)]

    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.monotonic() - started

    # n request liên tiếp phải cách nhau >= delay nên tổng thời gian >= (n - 1) * delay.
    assert len(session.calls) == n_threads
    assert elapsed >= (n_threads - 1) * delay - 1e-3


# ---------- run_s2_stage ----------


def test_run_s2_stage_isolates_failures_and_reports_every_paper(tmp_path):
    # Batch request: 3 bài, 1 thành công, 1 thất bại (lỗi HTTP), 1 not_found (null trong batch).
    # run_s2_stage gọi POST /paper/batch 1 lần cho cả 3 bài trong 1 chunk.
    responses = [
        FakeResponse(400),  # POST /paper/batch lỗi → toàn chunk failed
    ]
    client, _, _ = make_client(responses)
    seen = []

    results = run_s2_stage(
        ["2310.12345", "2401.00001", "2402.99999"],
        tmp_path,
        client,
        on_result=lambda result, seconds: seen.append((result.status, seconds)),
    )

    assert all(r.status == "failed" for r in results)
    assert [status for status, _ in seen] == ["failed", "failed", "failed"]
    assert all(seconds >= 0 for _, seconds in seen)


def test_run_s2_stage_batch_success_and_not_found(tmp_path):
    batch = [
        batch_paper("2310.12345", [batch_ref("p1", "1706.03762")]),
        None,  # S2 không có bài này
    ]
    client, _, _ = make_client([FakeResponse(200, batch)])

    results = run_s2_stage(["2310.12345", "2401.00001"], tmp_path, client)

    assert [r.status for r in results] == ["done", "not_found"]
    assert (tmp_path / "2310-12345" / "references.json").exists()
    assert not (tmp_path / "2401-00001").exists()


def test_run_s2_stage_accepts_generator_and_resumes(tmp_path):
    done_dir = tmp_path / "2310-12345"
    done_dir.mkdir()
    (done_dir / "references.json").write_text("{}", encoding="utf-8")
    # Chỉ bài 2401.00001 cần crawl → batch chỉ gọi 1 lần POST cho 1 bài.
    client, session, _ = make_client([FakeResponse(200, [batch_paper("2401.00001", [])])])

    results = run_s2_stage((i for i in ["2310.12345", "2401.00001"]), tmp_path, client)

    assert [r.status for r in results] == ["skipped", "done"]
    assert len(session.calls) == 1  # bài đã xong không tốn request


def test_run_s2_stage_marks_invalid_id_as_failed_without_stopping(tmp_path):
    # ID sai dạng: bị loại trước khi gửi batch; chỉ bài hợp lệ đi vào batch.
    client, session, _ = make_client([FakeResponse(200, [batch_paper("2401.00001", [])])])

    results = run_s2_stage(["2310-12345", "2401.00001"], tmp_path, client)

    assert [r.status for r in results] == ["failed", "done"]
    assert len(session.calls) == 1


# ---------- count_saved_references ----------


def test_count_saved_references_scans_every_finished_paper(tmp_path):
    first = [batch_paper("2310.12345", [
        batch_ref("p1", "1706.03762"), batch_ref("p2"), batch_ref("p3", "1810.04805"),
    ])]
    second = [batch_paper("2401.00001", [batch_ref("p4", "1301.3781")])]
    client, _, _ = make_client([FakeResponse(200, first), FakeResponse(200, second)])
    run_s2_stage(["2310.12345", "2401.00001"], tmp_path, client, batch_size=1)

    assert count_saved_references(tmp_path) == (2, 4, 3)


def test_count_saved_references_is_independent_of_resume(tmp_path):
    client, _, _ = make_client([FakeResponse(200, [batch_paper("2310.12345", [batch_ref("p1", "1706.03762")])])])
    run_s2_stage(["2310.12345"], tmp_path, client)

    # Chạy lại: bài bị skipped nhưng thống kê corpus vẫn đủ.
    run_s2_stage(["2310.12345"], tmp_path, make_client([])[0])

    assert count_saved_references(tmp_path) == (1, 1, 1)


def test_count_saved_references_tolerates_missing_all_file_and_unfinished_papers(tmp_path):
    legacy = tmp_path / "2310-12345"
    legacy.mkdir()
    write_json(legacy / "references.json", {"1706-03762": {}, "1810-04805": {}})
    (tmp_path / "2401-00001").mkdir()  # bài chưa có references.json: không được tính

    assert count_saved_references(tmp_path) == (1, 0, 2)


def test_count_saved_references_of_empty_root_is_zero(tmp_path):
    assert count_saved_references(tmp_path) == (0, 0, 0)


def test_run_s2_stage_reports_skipped_papers_with_zero_counts(tmp_path):
    folder = tmp_path / "2310-12345"
    write_json(folder / "references.json", {"1706-03762": {}})
    seen = []

    run_s2_stage(
        ["2310.12345"], tmp_path, make_client([])[0], on_result=lambda r, s: seen.append(r)
    )

    # Hợp đồng: on_result nhận cả bài skipped với số đếm 0, bên gọi phải tự lọc.
    assert [(r.status, r.n_refs, r.n_refs_co_arxiv_id) for r in seen] == [("skipped", 0, 0)]


# ---------- S2Client.post_json ----------


def test_post_json_sends_correct_method_and_headers():
    client, session, _ = make_client([FakeResponse(200, [{"ok": 1}])], api_key="secret")

    result = client.post_json("/paper/batch", {"fields": "title"}, {"ids": ["ARXIV:1"]})

    assert result == [{"ok": 1}]
    # POST phải điều hướng đến phương thức .post() trên session.
    assert session.calls[0][1]["headers"]["Content-Type"] == "application/json"
    assert session.calls[0][1]["headers"]["x-api-key"] == "secret"
    # json= được truyền vào, không phải data=
    assert session.calls[0][1]["json"] == {"ids": ["ARXIV:1"]}


def test_post_json_retries_on_429():
    responses = [FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(200, [{"ok": 1}])]
    client, session, clock = make_client(responses)

    result = client.post_json("/paper/batch", {}, {"ids": []})

    assert result == [{"ok": 1}]
    assert len(session.calls) == 2
    assert 3.0 in clock.sleeps


def test_post_json_returns_none_on_404():
    client, session, _ = make_client([FakeResponse(404)])

    assert client.post_json("/paper/batch", {}, {"ids": []}) is None
    assert len(session.calls) == 1


# ---------- fetch_references_batch ----------


def test_fetch_references_batch_maps_results_by_arxiv_id():
    batch = [
        batch_paper("2310.12345", [batch_ref("p1", "1706.03762"), batch_ref("p2")]),
        batch_paper("2401.00001", []),
    ]
    client, session, _ = make_client([FakeResponse(200, batch)])

    result = fetch_references_batch(["2310.12345", "2401.00001"], client)

    assert len(result["2310.12345"]) == 2
    assert result["2310.12345"][0].s2_id == "p1"
    assert result["2401.00001"] == []
    # Request đi đến /paper/batch với ARXIV: prefix.
    assert "/paper/batch" in session.calls[0][0]
    assert session.calls[0][1]["json"] == {"ids": ["ARXIV:2310.12345", "ARXIV:2401.00001"]}


def test_fetch_references_batch_marks_null_item_as_not_found():
    batch = [None, batch_paper("2401.00001", [])]
    client, _, _ = make_client([FakeResponse(200, batch)])

    result = fetch_references_batch(["2310.12345", "2401.00001"], client)

    assert result["2310.12345"] is None
    assert result["2401.00001"] == []


def test_fetch_references_batch_splits_into_chunks(monkeypatch):
    # Kiểm tra chunk chia đúng khi có nhiều hơn BATCH_SIZE bài.
    n = BATCH_SIZE + 1
    ids = [f"2310.{str(i).zfill(5)}" for i in range(n)]
    # Mỗi chunk trả một batch list.
    chunk1 = [batch_paper(aid, []) for aid in ids[:BATCH_SIZE]]
    chunk2 = [batch_paper(ids[BATCH_SIZE], [])]
    client, session, _ = make_client([
        FakeResponse(200, chunk1),
        FakeResponse(200, chunk2),
    ])

    result = fetch_references_batch(ids, client)

    assert len(result) == n
    assert len(session.calls) == 2  # 2 POST requests


def test_fetch_references_batch_fallbacks_on_full_page(monkeypatch):
    # Bài có đúng S2_PAGE_SIZE refs trong batch → fallback sang GET /references.
    from src.m1_crawling.s2_api import S2_PAGE_SIZE

    big_refs = [batch_ref(f"p{i}") for i in range(S2_PAGE_SIZE)]
    batch = [batch_paper("2310.12345", big_refs)]
    # Request 1: POST batch, trả đầy đủ 1000 refs.
    # Request 2+: GET /references (pagination) chỉ trả 1 trang.
    get_page = {"data": [{"citedPaper": batch_ref(f"p{i}")} for i in range(S2_PAGE_SIZE)]}
    client, session, _ = make_client([
        FakeResponse(200, batch),
        FakeResponse(200, get_page),  # fallback GET
    ])

    result = fetch_references_batch(["2310.12345"], client)

    assert len(result["2310.12345"]) == S2_PAGE_SIZE
    assert len(session.calls) == 2  # 1 POST + 1 GET fallback


# ---------- crawl_references_batch ----------


def test_crawl_references_batch_writes_files_and_skips_done(tmp_path):
    done_dir = tmp_path / "2310-12345"
    done_dir.mkdir()
    (done_dir / "references.json").write_text("{}", encoding="utf-8")
    batch = [batch_paper("2401.00001", [batch_ref("p1", "1706.03762"), batch_ref("p2")])]
    client, session, _ = make_client([FakeResponse(200, batch)])

    results = crawl_references_batch(
        ["2310.12345", "2401.00001"], tmp_path, client, resume=True
    )

    assert [r.status for r in results] == ["skipped", "done"]
    assert (r := results[1]).n_refs == 2 and r.n_refs_co_arxiv_id == 1
    assert len(session.calls) == 1  # chỉ POST cho bài chưa xong
    assert (tmp_path / "2401-00001" / "references.json").exists()


def test_crawl_references_batch_handles_not_found_and_invalid_id(tmp_path):
    batch = [None]  # chỉ bài hợp lệ đi vào batch; ID sai bị loại sớm.
    client, _, _ = make_client([FakeResponse(200, batch)])

    results = crawl_references_batch(["2310-bad", "2401.00001"], tmp_path, client)

    assert results[0].status == "failed"   # ID sai dạng
    assert results[1].status == "not_found"  # S2 không có bài


def test_crawl_references_batch_marks_all_failed_on_batch_error(tmp_path):
    client, _, _ = make_client([FakeResponse(400)])  # POST lỗi không thể retry

    results = crawl_references_batch(["2310.12345", "2401.00001"], tmp_path, client)

    assert all(r.status == "failed" for r in results)
    assert not any(tmp_path.iterdir())  # không ghi file nào

# ---------- fallback robustness  ----------


def test_fetch_references_batch_fallback_failure_skips_only_that_paper(monkeypatch):
    """Fallback GET lỗi → chỉ paper đó bị bỏ qua, không kill cả batch."""
    from src.m1_crawling import s2_api

    big_refs = [batch_ref(f"p{i}") for i in range(s2_api.S2_PAGE_SIZE)]
    batch = [
        batch_paper("2310.12345", big_refs),
        batch_paper("2401.00001", [batch_ref("p1", "1706.03762")]),
    ]
    client, _, _ = make_client([FakeResponse(200, batch)])

    def fake_fetch(arxiv_id, client=None):
        raise s2_api.S2Error("network down")

    monkeypatch.setattr(s2_api, "fetch_references", fake_fetch)

    result = fetch_references_batch(["2310.12345", "2401.00001"], client)

    # Paper A: không có trong result vì fallback fail → caller sẽ đánh failed.
    assert "2310.12345" not in result
    # Paper B: không bị ảnh hưởng, vẫn parse bình thường.
    assert result["2401.00001"][0].s2_id == "p1"


def test_fetch_references_batch_fallback_none_keeps_batch_refs(monkeypatch):
    """Fallback GET trả None → giữ refs từ batch, không bị overwrite thành []."""
    from src.m1_crawling import s2_api

    big_refs = [batch_ref(f"p{i}", arxiv="1706.03762") for i in range(s2_api.S2_PAGE_SIZE)]
    batch = [batch_paper("2310.12345", big_refs)]
    client, _, _ = make_client([FakeResponse(200, batch)])

    def fake_fetch(arxiv_id, client=None):
        return None  # S2 nói không có bài dù batch vừa trả refs

    monkeypatch.setattr(s2_api, "fetch_references", fake_fetch)

    result = fetch_references_batch(["2310.12345"], client)

    # Không được để result thành [] (bug cũ: `or []`).
    assert len(result["2310.12345"]) == s2_api.S2_PAGE_SIZE
    assert result["2310.12345"][0].arxiv_id == "1706.03762"


def test_crawl_references_batch_marks_paper_failed_when_fallback_fails(tmp_path, monkeypatch):
    """End-to-end: fallback fail → chỉ paper đó failed, paper khác vẫn done."""
    from src.m1_crawling import s2_api

    big_refs = [batch_ref(f"p{i}") for i in range(s2_api.S2_PAGE_SIZE)]
    batch = [
        batch_paper("2310.12345", big_refs),
        batch_paper("2401.00001", [batch_ref("p1", "1706.03762")]),
    ]
    client, _, _ = make_client([FakeResponse(200, batch)])

    def fake_fetch(arxiv_id, client=None):
        raise s2_api.S2Error("network down")

    monkeypatch.setattr(s2_api, "fetch_references", fake_fetch)

    results = crawl_references_batch(["2310.12345", "2401.00001"], tmp_path, client)

    # Paper A (fallback fail) → failed, không ghi file.
    assert results[0].status == "failed"
    assert not (tmp_path / "2310-12345" / "references.json").exists()
    # Paper B (không fallback) → done bình thường.
    assert results[1].status == "done"
    assert (tmp_path / "2401-00001" / "references.json").exists()

# ---------- bảo vệ zip() khi batch trả sai số phần tử ----------


def test_fetch_references_batch_raises_when_fewer_items_than_ids():
    """Batch trả ít hơn số ID yêu cầu → S2Error thay vì zip cắt ngầm."""
    batch = [batch_paper("2310.12345", [batch_ref("p1", "1706.03762")])]  # 1 item cho 2 IDs
    client, _, _ = make_client([FakeResponse(200, batch)])

    with pytest.raises(S2Error, match="1 phần tử, mong đợi 2"):
        fetch_references_batch(["2310.12345", "2401.00001"], client)


def test_fetch_references_batch_raises_when_more_items_than_ids():
    """Batch trả nhiều hơn số ID yêu cầu → S2Error thay vì bỏ qua item dư."""
    batch = [
        batch_paper("2310.12345", []),
        batch_paper("2401.00001", []),
        batch_paper("2402.99999", []),
    ]
    client, _, _ = make_client([FakeResponse(200, batch)])

    with pytest.raises(S2Error, match="3 phần tử, mong đợi 2"):
        fetch_references_batch(["2310.12345", "2401.00001"], client)


def test_crawl_references_batch_marks_all_failed_on_length_mismatch(tmp_path):
    """Batch trả sai số phần tử → cả chunk failed, không ghi file nào.

    Cùng cơ chế với batch lỗi HTTP: caller coi toàn chunk là failed để lần Resume
    sau retry, thay vì lưu dữ liệu có thể bị ghép sai vị trí.
    """
    client, _, _ = make_client([FakeResponse(200, [])])  # 0 item cho 2 IDs

    results = crawl_references_batch(["2310.12345", "2401.00001"], tmp_path, client)

    assert all(r.status == "failed" for r in results)
    assert not any(tmp_path.iterdir())

# ---------- timing batch: amortize network time (review P2-02) ----------


class SlowSession(FakeSession):
    """Session giả có delay thật để kiểm tra thời gian batch được phân bổ."""

    def __init__(self, responses, slow_seconds=0.05):
        super().__init__(responses)
        self._slow = slow_seconds

    def post(self, url, **kwargs):
        time.sleep(self._slow)
        return super().post(url, **kwargs)


def test_crawl_references_batch_attributes_batch_time_to_papers(tmp_path):
    """on_result phải nhận thời gian gồm request batch, không chỉ file I/O."""
    batch = [
        batch_paper("2310.12345", [batch_ref("p1", "1706.03762")]),
        batch_paper("2401.00001", []),
    ]
    session = SlowSession([FakeResponse(200, batch)], slow_seconds=0.10)
    client = S2Client(S2Settings(delay_seconds=0.0, api_key="k"), session=session)
    seen = []

    crawl_references_batch(
        ["2310.12345", "2401.00001"], tmp_path, client,
        on_result=lambda r, s: seen.append((r.arxiv_id, s)),
    )

    # Batch mất ~0.10s, chia 2 bài → mỗi bài nhận >= 0.05s (cộng file I/O).
    for arxiv_id, seconds in seen:
        assert seconds >= 0.03, f"{arxiv_id} chỉ được {seconds:.3f}s, kỳ vọng >= 0.03s"


def test_crawl_references_batch_skipped_papers_get_no_network_time(tmp_path):
    """Bài skipped không được cộng thời gian mạng."""
    done_dir = tmp_path / "2310-12345"
    done_dir.mkdir()
    (done_dir / "references.json").write_text("{}", encoding="utf-8")
    session = SlowSession(
        [FakeResponse(200, [batch_paper("2401.00001", [])])], slow_seconds=0.10
    )
    client = S2Client(S2Settings(delay_seconds=0.0, api_key="k"), session=session)
    seen = []

    crawl_references_batch(
        ["2310.12345", "2401.00001"], tmp_path, client,
        on_result=lambda r, s: seen.append((r.arxiv_id, r.status, s)),
    )

    by_id = {aid: (status, seconds) for aid, status, seconds in seen}
    # Skipped: 0 network, chỉ overhead cực nhỏ.
    assert by_id["2310.12345"][0] == "skipped"
    assert by_id["2310.12345"][1] < 0.05
    # Done: nhận toàn bộ ~0.10s vì chỉ có 1 bài trong to_fetch.
    assert by_id["2401.00001"][0] == "done"
    assert by_id["2401.00001"][1] >= 0.08


def test_crawl_references_batch_not_found_gets_batch_time(tmp_path):
    """not_found vẫn có mặt trong batch response → nhận phần network amortized."""
    batch = [None, batch_paper("2401.00001", [])]
    session = SlowSession([FakeResponse(200, batch)], slow_seconds=0.10)
    client = S2Client(S2Settings(delay_seconds=0.0, api_key="k"), session=session)
    seen = []

    crawl_references_batch(
        ["2310.12345", "2401.00001"], tmp_path, client,
        on_result=lambda r, s: seen.append((r.arxiv_id, r.status, s)),
    )

    by_id = {aid: (status, seconds) for aid, status, seconds in seen}
    # Cả 2 bài đều nhận ~0.10s / 2 = 0.05s.
    assert by_id["2310.12345"][0] == "not_found"
    assert by_id["2310.12345"][1] >= 0.03
    assert by_id["2401.00001"][0] == "done"
    assert by_id["2401.00001"][1] >= 0.03


def test_crawl_references_batch_failed_papers_get_batch_time(tmp_path):
    """Khi batch fail, mọi bài trong chunk vẫn nhận giờ mạng đã tiêu."""
    session = SlowSession([FakeResponse(400)], slow_seconds=0.10)
    client = S2Client(
        S2Settings(delay_seconds=0.0, api_key="k", max_retries=1), session=session
    )
    seen = []

    crawl_references_batch(
        ["2310.12345", "2401.00001"], tmp_path, client,
        on_result=lambda r, s: seen.append((r.arxiv_id, r.status, s)),
    )

    for arxiv_id, status, seconds in seen:
        assert status == "failed"
        # Batch lỗi nhưng đã tiêu ~0.10s; chia 2 bài → >= 0.03s.
        assert seconds >= 0.03, f"{arxiv_id} chỉ được {seconds:.3f}s"


def test_crawl_references_batch_all_skipped_does_not_divide_by_zero(tmp_path):
    """to_fetch rỗng: không có phép chia nào cho len(to_fetch) = 0."""
    for folder in ["2310-12345", "2401-00001"]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "references.json").write_text("{}", encoding="utf-8")
    client, session, _ = make_client([])  # không có response nào

    results = crawl_references_batch(
        ["2310.12345", "2401.00001"], tmp_path, client,
    )

    assert [r.status for r in results] == ["skipped", "skipped"]
    assert session.calls == []