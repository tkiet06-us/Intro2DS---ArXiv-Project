"""Unit tests cho src/m1_crawling/s2_api.py (không gọi mạng thật, dùng tests/mock_data)."""

from pathlib import Path

import pytest
import requests

from src.common.utils_io import read_json
from src.m1_crawling.s2_api import (
    S2Client,
    S2Error,
    S2Reference,
    S2Settings,
    crawl_references,
    fetch_references,
    parse_reference,
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

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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
    """Dựng một phần tử `data` của response S2."""
    external = {"ArXiv": arxiv} if arxiv else {}
    paper = {
        "paperId": paper_id,
        "title": title,
        "authors": [{"authorId": "1", "name": name} for name in authors],
        "externalIds": external,
        **extra,
    }
    return {"citedPaper": paper}


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


def test_fetch_references_encodes_slash_of_old_style_id():
    client, session, _ = make_client([FakeResponse(200, {"data": []})])

    fetch_references("hep-th/9901001", client)

    assert session.calls[0][0].endswith("/paper/arXiv:hep-th%2F9901001/references")


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
