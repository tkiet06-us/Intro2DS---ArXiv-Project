"""Kiểm thử luật gán nhãn tự động (auto_label.py) và bộ đọc thư mục bài (corpus.py)."""

import json
import math

import pytest

from src.m3_matching import auto_label as A
from src.m3_matching import corpus as C
from src.m3_matching.features import extract_arxiv_ids
from src.schemas import Bibtex, Reference

PAPER = "2410.13025"


def entry(key, title=None, authors=("Ashish Vaswani",), year=2017, journal=None, url=None,
          doi=None):
    return Bibtex(arxiv_id=PAPER, citation_key=key, entry_type="article", title=title,
                  authors=list(authors), year=year, journal=journal, url=url, doi=doi)


def cand(ref_id, title, authors=("A. Vaswani",), date="2017-06-12"):
    return Reference(arxiv_id=PAPER, ref_arxiv_id=ref_id, ref_title=title,
                     ref_authors=list(authors), ref_submission_date=date)


POOL = [
    cand("1706.03762", "Attention Is All You Need"),
    cand("1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers",
         authors=("Jacob Devlin",), date="2018-10-11"),
]


def scores(entries, pool=POOL):
    return A.score_paper(entries, pool)


# =============================================================== regex ID arXiv (features)
@pytest.mark.parametrize("text", [
    "http://dx.doi.org/10.1109/QSW59989.2023.00028",   # "2023.00028": tháng 23
    "https://doi.org/10.3389/fphy.2014.00005",          # "2014.00005": tháng 14
    "10.1287/moor.2022.0134",                           # tháng 22
    "arXiv:1706.037625",                                # 6 chữ số
])
def test_extract_arxiv_ids_rejects_doi_like_numbers(text):
    assert extract_arxiv_ids(text) == set()


def test_extract_arxiv_ids_checks_sequence_length_by_year():
    assert extract_arxiv_ids("arXiv:1412.6980") == {"1412.6980"}       # trước 2015: 4 chữ số
    assert extract_arxiv_ids("arXiv:1412.69801") == set()
    assert extract_arxiv_ids("arXiv:1501.00001") == {"1501.00001"}     # từ 2015: 5 chữ số
    assert extract_arxiv_ids("arXiv:1501.0001") == set()


def test_extract_arxiv_ids_keeps_arxiv_doi():
    assert extract_arxiv_ids("10.48550/arXiv.2403.13187") == {"2403.13187"}


# =============================================================== Rule
@pytest.mark.parametrize("name", [
    "arxiv_in_url", "title_lev>=1.00", "title_lev>=0.90|m0.05|any",
    "title_lev_compact>=1.00|any", "title_token_set>=0.95|m0.05|first|y1",
    "title_partial>=0.95|any|len20",
])
def test_rule_name_roundtrip(name):
    assert A.Rule.from_name(name).name == name


@pytest.mark.parametrize("bad", ["title_lev", "foo>=1", "title_cosine>=0.9", "title_lev>=0.9|zz"])
def test_rule_from_name_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        A.Rule.from_name(bad)


def test_arxiv_rule_needs_exactly_one_id_in_pool():
    sc = scores([
        entry("a", journal="arXiv preprint arXiv:1706.03762"),
        entry("b", journal="arXiv:1706.03762 and arXiv:1810.04805"),
        entry("c", url="https://arxiv.org/abs/2401.00001"),          # ID ngoài pool
    ])
    rule = A.Rule(kind="arxiv_id")
    assert rule.apply(sc, 0) == (0, 1.0)
    assert rule.apply(sc, 1) is None
    assert rule.apply(sc, 2) is None


def test_title_rule_threshold_and_author():
    sc = scores([
        entry("exact", title="Attention is all you need"),
        entry("noauthor", title="Attention is all you need", authors=()),
        entry("otherauthor", title="Attention is all you need", authors=("John Smith",)),
        entry("far", title="A survey of transformers"),
    ])
    rule = A.Rule.from_name("title_lev>=0.90|any")
    assert rule.apply(sc, 0)[0] == 0
    assert rule.apply(sc, 1) is None          # thiếu tác giả: không đạt ràng buộc tác giả
    assert rule.apply(sc, 2) is None
    assert rule.apply(sc, 3) is None
    assert A.Rule.from_name("title_lev>=0.90").apply(sc, 2)[0] == 0


def test_margin_rejects_two_near_identical_candidates():
    pool = [cand("2001.00001", "Learning shallow quantum circuits"),
            cand("2001.00002", "Learning shallow quantum circuit")]
    sc = scores([entry("x", title="Learning shallow quantum circuits")], pool)
    assert A.Rule.from_name("title_lev>=0.90").apply(sc, 0)[0] == 0
    assert A.Rule.from_name("title_lev>=0.90|m0.05").apply(sc, 0) is None


def test_compact_metric_ignores_split_capitals():
    pool = [cand("2203.00001", "L3Cube-MahaHate: A Tweet-based Marathi Hate Speech Dataset",
                 authors=("H. Patil",))]
    split = "L 3 C ube- M aha H ate: A Tweet-based M arathi Hate Speech Dataset"
    sc = scores([entry("x", title=split, authors=("Patil, Hrushikesh",))], pool)
    assert sc.lev_compact[0, 0] == pytest.approx(1.0)
    assert sc.lev[0, 0] < 1.0


def test_year_constraint_skipped_when_year_missing():
    rule = A.Rule.from_name("title_lev>=0.90|y1")
    assert rule.apply(scores([entry("x", title="Attention is all you need", year=None)]), 0)[0] == 0
    assert rule.apply(scores([entry("x", title="Attention is all you need", year=2030)]), 0) is None


# =============================================================== label_paper / predict_paper
def test_label_paper_covers_every_entry_with_valid_status():
    paper = C.PaperRecord(PAPER, entries=[
        entry("vaswani", title="Attention is all you need"),
        entry("devlin", title="unrelated", authors=("Jacob Devlin",),
              url="arxiv.org/abs/1810.04805"),
        entry("other", title="Something else entirely"),
    ], candidates=POOL)
    doc = A.label_paper(paper, A.default_rules())
    assert doc["arxiv_id"] == PAPER and doc["label_source"] == "auto"
    by_key = {r["citation_key"]: r for r in doc["labels"]}
    assert set(by_key) == {"vaswani", "devlin", "other"}
    assert by_key["devlin"]["groundtruth_arxiv_id"] == "1810.04805"
    assert by_key["devlin"]["rule"] == A.ARXIV_RULE
    assert by_key["vaswani"]["label_status"] == "matched" and by_key["vaswani"]["rule"]
    assert by_key["other"] == {**by_key["other"], "groundtruth_arxiv_id": None,
                               "label_status": "unresolved", "rule": None}


def test_cascade_uses_first_rule_that_fires():
    sc = scores([entry("x", title="Attention is all you need", journal="arXiv:1810.04805")])
    preds = A.predict_paper(sc, A.default_rules())
    assert preds == [("1810.04805", A.ARXIV_RULE)]   # ID ghi trong mục được ưu tiên hơn tiêu đề


def test_one_to_one_keeps_higher_priority_entry():
    sc = scores([
        entry("dup_title", title="Attention is all you need"),
        entry("by_id", title="x", journal="arXiv:1706.03762"),
    ])
    rules = A.default_rules()
    assert [p for p, _ in A.predict_paper(sc, rules)] == ["1706.03762", "1706.03762"]
    assert [p for p, _ in A.predict_paper(sc, rules, one_to_one=True)] == [None, "1706.03762"]


def test_empty_pool_gives_unresolved():
    sc = A.score_paper([entry("x", title="Attention is all you need")], [])
    assert A.predict_paper(sc, A.default_rules()) == [(None, None)]


# =============================================================== nhãn chuẩn và đo
def test_silver_gold_from_ids():
    paper = C.PaperRecord(PAPER, entries=[
        entry("in_pool", journal="arXiv:1706.03762"),
        entry("not_in_pool", journal="arXiv:2401.00001"),
        entry("no_id", title="Attention is all you need"),
    ], candidates=POOL)
    gold = A.silver_gold([(paper, scores(paper.entries))])
    assert gold == {(PAPER, "in_pool"): "1706.03762", (PAPER, "not_in_pool"): None}


def test_rule_report_arithmetic():
    pairs = [("a", "a"), ("b", "c"), ("d", None), (None, "e"), (None, None)]
    r = A.RuleReport.build("r", pairs, fired_all=3, n_all=10)
    assert (r.fired, r.correct, r.wrong_ref, r.false_on_none) == (3, 1, 1, 1)
    assert r.precision == pytest.approx(1 / 3)
    assert r.recall == pytest.approx(1 / 3)      # 3 mục có đáp án: a, c, e
    assert r.coverage_all == pytest.approx(0.3)
    empty = A.RuleReport.build("r", [(None, None)], 0, 1)
    assert math.isnan(empty.precision) and math.isnan(empty.recall)


def test_evaluate_rules_and_select():
    paper = C.PaperRecord(PAPER, entries=[
        entry("v", title="Attention is all you need", journal="arXiv:1706.03762"),
        entry("w", title="Attention is all you need", journal="arXiv:2401.00001"),
    ], candidates=POOL)
    scored = [(paper, scores(paper.entries))]
    gold = A.silver_gold(scored)
    reports = A.evaluate_rules(scored, gold, [A.Rule.from_name("title_lev>=1.00")],
                               {"default": A.default_rules()})
    title = reports[0]
    assert (title.fired, title.correct, title.false_on_none) == (2, 1, 1)
    assert reports[-1].rule == "cascade:default"
    assert A.select_rules(reports, target=0.4, min_support=1)[0].rule == "title_lev>=1.00"


def test_gold_from_labels_reads_manual_only(tmp_path):
    (tmp_path / "split.json").write_text("{}", encoding="utf-8")
    labels = [
        {"citation_key": "k1", "groundtruth_arxiv_id": "1706.03762", "label_status": "matched"},
        {"citation_key": "k2", "groundtruth_arxiv_id": None, "label_status": "no_match_confirmed"},
    ]
    doc = {"arxiv_id": PAPER, "label_source": "manual", "labels": labels}
    (tmp_path / "a.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"arxiv_id": "2410.13026", "label_source": "auto",
                                                 "labels": []}), encoding="utf-8")
    assert A.gold_from_labels(tmp_path) == {(PAPER, "k1"): "1706.03762", (PAPER, "k2"): None}


# =============================================================== corpus + CLI
def make_paper_dir(root):
    d = root / "2410-13025"
    (d / "v1").mkdir(parents=True)
    (d / "references.json").write_text(json.dumps({
        "1706-03762": {"ref_title": "Attention Is All You Need", "ref_authors": ["A. Vaswani"],
                       "ref_submission_date": "2017-06-12", "ref_s2_id": "x"},
    }), encoding="utf-8")
    (d / "v1" / "main.tex").write_text(
        "\\documentclass{article}\\begin{document}Text \\cite{vaswani}.\n"
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{vaswani} A. Vaswani, N. Shazeer. \\emph{Attention is all you need}. "
        "NeurIPS, 2017.\n"
        "\\bibitem{other} J. Smith. \\emph{A completely different paper}. ICML, 2020.\n"
        "\\end{thebibliography}\\end{document}\n", encoding="utf-8")
    return d


def test_corpus_load_paper_from_raw(tmp_path):
    d = make_paper_dir(tmp_path)
    paper = C.load_paper(d)
    assert paper.arxiv_id == PAPER and paper.entry_source == "raw"
    assert [c.ref_arxiv_id for c in paper.candidates] == ["1706.03762"]
    assert {e.citation_key for e in paper.entries} == {"vaswani", "other"}


def test_corpus_prefers_processed_refs_bib(tmp_path):
    d = make_paper_dir(tmp_path / "raw")
    processed = tmp_path / "processed"
    (processed / d.name).mkdir(parents=True)
    (processed / d.name / "refs.bib").write_text(
        "@article{vaswani2017, title={Attention is all you need}, author={Vaswani, A.}, "
        "year={2017}}",
        encoding="utf-8")
    paper = C.load_paper(d, processed)
    assert paper.entry_source == "processed"
    assert [e.citation_key for e in paper.entries] == ["vaswani2017"]


def test_cli_label_writes_label_files(tmp_path):
    make_paper_dir(tmp_path / "raw")
    out = tmp_path / "labels"
    assert A.main(["label", "--raw-root", str(tmp_path / "raw"), "--out-dir", str(out)]) == 0
    doc = json.loads((out / "2410-13025.json").read_text(encoding="utf-8"))
    by_key = {r["citation_key"]: r for r in doc["labels"]}
    assert by_key["vaswani"]["groundtruth_arxiv_id"] == "1706.03762"
    assert by_key["other"]["label_status"] == "unresolved"


def test_cli_evaluate_writes_report(tmp_path):
    make_paper_dir(tmp_path / "raw")
    out = tmp_path / "rules.csv"
    assert A.main(["evaluate", "--raw-root", str(tmp_path / "raw"), "--out", str(out)]) == 0
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("rule,n_gold,n_answerable,fired,correct")
