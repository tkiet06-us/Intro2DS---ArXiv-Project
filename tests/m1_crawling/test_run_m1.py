import io, json, tarfile, threading, time
from types import SimpleNamespace
import pytest
from src.common.utils_io import read_json
from src.m1_crawling import crawler, run_m1, s2_api

TEX = b"\\documentclass{article}\n\\begin{document}\nHi.\n\\end{document}\n"

def tar_gz():
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        i = tarfile.TarInfo("main.tex"); i.size = len(TEX); t.addfile(i, io.BytesIO(TEX))
    return b.getvalue()

class Resp:
    def __init__(self, status=200, body=b"", payload=None):
        self.status_code, self._b, self._p, self.headers = status, body, payload, {"Content-Type": "x"}
    def iter_content(self, chunk_size=1): yield self._b
    def close(self): pass
    def json(self): return self._p

class ArxivSession:
    headers = {}
    def __init__(self): self.threads = set()
    def get(self, url, **kw):
        self.threads.add(threading.current_thread().name); time.sleep(0.05)
        return Resp(body=tar_gz())

class S2Session:
    def __init__(self): self.threads = set()
    def post(self, url, **kw):
        self.threads.add(threading.current_thread().name); time.sleep(0.05)
        ids = kw["json"]["ids"]
        return Resp(payload=[{"paperId": "p", "references": [
            {"paperId": "r1", "title": "T", "authors": [], "externalIds": {"ArXiv": "1706.03762"}},
            {"paperId": "r2", "title": "U", "authors": [], "externalIds": {}}]} for _ in ids])

def rows(n):
    return [{"arxiv_id": f"2506.{i:05d}", "n_versions": 2, "v1_date": "2025-06-01T00:00:00Z",
             "latest_version_date": "2025-06-02T00:00:00Z",
             "version_dates": {"v1": "2025-06-01T00:00:00Z", "v2": "2025-06-02T00:00:00Z"},
             "title": "T", "authors": ["A"], "venue": None} for i in range(788, 788 + n)]

def args(tmp, **kw):
    ns = dict(name="t", id_start="2506.00788", id_end="2506.00799", ranges=None, contact="a@b.c",
              kaggle_json=None, total_ids=5, config="configs/config.yaml", data_root=tmp / "raw",
              reports=tmp / "rep", delay=3.0, limit=None, max_disk_gb=None, max_consecutive_errors=10,
              s2_batch=2, skip_discovery=True, skip_crawl=False, skip_s2=False,
              fresh_discovery=False, fresh_crawl=False, fresh_s2=False)
    ns.update(kw); return SimpleNamespace(**ns)

def prep(tmp, n=5):
    d = tmp / "raw"; d.mkdir()
    (d / "discovery_t.jsonl").write_text("\n".join(json.dumps(r) for r in rows(n)), encoding="utf-8")
    a = ArxivSession(); s = S2Session()
    f = crawler.PoliteFetcher("UA", 3.0, session=a, sleep=lambda _s: None)
    c = s2_api.S2Client(s2_api.S2Settings(delay_seconds=0, api_key="k"), session=s)
    return a, s, f, c

def test_both_stages_run_in_own_threads_and_stats_are_written(tmp_path):
    a, s, f, c = prep(tmp_path)
    code = run_m1.run_pipeline(args(tmp_path), "a@b.c", fetcher=f, s2_client=c)
    assert code == 0
    assert a.threads == {"arxiv"} and s.threads == {"s2"}
    p = tmp_path / "raw" / "papers_t" / "2506-00788"
    assert (p / "v2" / "main.tex").exists() and (p / "references.json").exists()
    st = read_json(tmp_path / "rep" / "summary_stats_t.json")
    assert st["n_papers"] == 5 and st["crawl"]["by_status"] == {"ok": 10}
    assert st["s2"]["papers_with_references"] == 5 and st["s2"]["avg_refs_per_paper"] == 2.0
    assert st["s2"]["arxiv_id_rate"] == 0.5 and st["crawl_rates"]["version_ok_rate"] == 1.0
    assert st["pipeline"]["seconds_saved_by_overlap"] >= 0
    assert (tmp_path / "rep" / "profile_m1_t.json").exists()

def test_rerun_resumes_without_requests(tmp_path):
    a, s, f, c = prep(tmp_path)
    run_m1.run_pipeline(args(tmp_path), "a@b.c", fetcher=f, s2_client=c)
    a2 = ArxivSession(); s2 = S2Session()
    f2 = crawler.PoliteFetcher("UA", 3.0, session=a2, sleep=lambda _s: None)
    c2 = s2_api.S2Client(s2_api.S2Settings(delay_seconds=0, api_key="k"), session=s2)
    assert run_m1.run_pipeline(args(tmp_path), "a@b.c", fetcher=f2, s2_client=c2) == 0
    assert not a2.threads and not s2.threads

def test_arxiv_block_does_not_stop_s2_and_gives_exit_3(tmp_path):
    a, s, f, c = prep(tmp_path)
    a.get = lambda url, **kw: Resp(status=403)
    code = run_m1.run_pipeline(args(tmp_path), "a@b.c", fetcher=f, s2_client=c)
    assert code == 3
    st = read_json(tmp_path / "rep" / "summary_stats_t.json")
    assert st["s2"]["papers_with_references"] == 5  # S2 van xong
    assert "CrawlBlockedError" in st["pipeline"]["errors"]["arxiv"]

def test_validation():
    ns = run_m1.build_parser().parse_args(["--name", "x", "--skip_discovery"])
    assert "id_start" in run_m1._validate(ns, "a@b.c")
    ns = run_m1.build_parser().parse_args(["--name", "x", "--id-start", "2506.00001", "--id_end", "2506.00002"])
    assert "kaggle" in run_m1._validate(ns, "a@b.c")
    ns.skip_discovery = True
    assert run_m1._validate(ns, None) and run_m1._validate(ns, "a@b.c") is None
