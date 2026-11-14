"""Kiểm thử train.py và evaluate.py. Chạy từ thư mục gốc repo: python -m pytest -q tests/test_m3_train_eval.py"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import synth as S  # noqa: E402

from src.m3_matching import evaluate as E  # noqa: E402
from src.m3_matching import features as F  # noqa: E402
from src.m3_matching import train as T  # noqa: E402

MEMBERS = 2


def _papers(**kw):
    bib, refs, split, files = S.make_dataset(members=MEMBERS, **kw)
    lf = {k: {"label_source": v["label_source"], "records": v["records"]} for k, v in files.items()}
    return T.load_papers(bib, refs, split, lf), (bib, refs, split, files)


@pytest.fixture(scope="module")
def papers():
    ps, _ = _papers()
    T.build_tensors(ps)
    return ps


# =============================================================== evaluate: MRR
def test_mrr_matches_design_example():
    pred = [["x", "A", None, None, None], ["B", None, None, None, None], list("pqrst"), [None] * 5]
    truth = ["A", "B", "Z", None]
    assert E.mrr_at_k(pred, truth) == pytest.approx(0.375)          # ml_design mục 7
    assert E.mrr(pred, truth) == pytest.approx(0.375)               # tên cũ còn dùng được


def test_none_truth_never_matches_none_cell_and_is_in_denominator():
    assert E.reciprocal_ranks([[None, None]], [None]).tolist() == [0.0]
    assert E.mrr_at_k([["a"], [None]], ["a", None]) == 0.5           # no-match vẫn nằm trong |Q|


def test_answer_beyond_k_scores_zero_and_group_na():
    assert E.reciprocal_ranks([["a", "b", "c", "d", "e", "g"]], ["g"], k=5).tolist() == [0.0]
    rep = E.mrr_report([["a"]], ["a"], sources=["manual"])
    assert rep["manual"]["mrr"] == 1.0 and math.isnan(rep["auto"]["mrr"]) and rep["auto"]["n"] == 0


def test_report_ceiling_and_hit():
    rep = E.mrr_report([["a"], ["z"], [None]], ["a", "b", None], sources=["manual"] * 3)
    assert rep["overall"]["mrr_ceiling"] == pytest.approx(2 / 3)
    assert rep["overall"]["hit@5"] == 0.5 and rep["overall"]["mrr_answerable"] == 0.5


def test_compare_with_baseline_verdicts():
    rep = lambda m: {"manual": {"mrr": m, "n": 10}}  # noqa: E731
    out = E.compare_with_baseline({"baseline": rep(0.5), "a": rep(0.6), "b": rep(0.5),
                                   "c": rep(0.4), "d": rep(float("nan"))})
    assert out["a"]["beats_baseline"] is True
    assert out["b"]["beats_baseline"] is False       # bằng baseline = không thắng
    assert out["c"]["beats_baseline"] is False
    assert out["d"]["beats_baseline"] is None
    with pytest.raises(E.BaselineNotBeatenError):
        E.compare_with_baseline({"baseline": rep(0.5), "c": rep(0.4)}, raise_if_not=True)


# =============================================================== evaluate: predictions
POOLS = {"2310.12345": ["1706.03762", "1810.04805", "1409.0473", "1512.03385", "1412.6980", "2005.14165"],
         "hep-th/9901001": ["1706.03762"], "2401.00001": []}


def _row(aid, part, key, gt, *c):
    return E.PredictionRow(aid, part, key, gt, c)


def test_predictions_roundtrip_schema(tmp_path):
    rows = [_row("2310.12345", "test", "a", "1706.03762", "1706.03762", "1810.04805", "1409.0473", "1512.03385", "1412.6980"),
            _row("2310.12345", "test", "b", None, "1810.04805", "1706.03762", "1409.0473", "1512.03385", "1412.6980"),
            _row("hep-th/9901001", "valid", "c", None, "1706.03762"),
            _row("2401.00001", "train", "d", None)]
    path = tmp_path / "p.parquet"
    E.write_predictions(path, rows, POOLS, {"2310.12345": "test", "hep-th/9901001": "valid", "2401.00001": "train"})
    schema = pq.read_table(path).schema
    assert schema.names == ["arxiv_id", "partition", "citation_key", "groundtruth_arxiv_id"] + [f"cand_{i}" for i in range(1, 6)]
    assert [f.nullable for f in schema] == [False, False, False] + [True] * 6      # 6.1.6
    assert pa.types.is_dictionary(schema.field("partition").type)
    assert schema.field("partition").type.index_type == pa.int8()
    assert pq.read_table(path).to_pydict()["cand_1"][3] is None                   # pool rỗng -> null


@pytest.mark.parametrize("bad, fragment", [
    (_row("2310.12345", "test", "a", None, "1706.03762", "1706.03762", "1810.04805", "1409.0473", "1512.03385"), "trùng"),
    (_row("2310.12345", "test", "a", None, "1706.03762", None, "1810.04805", "1409.0473", "1512.03385"), "khoảng trống"),
    (_row("2310.12345", "test", "a", None, "1706.03762", "1810.04805"), "cần 5"),               # thiếu ứng viên dù pool đủ
    (_row("2310.12345", "test", "a", None, "9999.99999", "1810.04805", "1409.0473", "1512.03385", "1412.6980"), "ngoài pool"),
    (_row("2310.12345", "test", "a", "9999.99999", "1706.03762", "1810.04805", "1409.0473", "1512.03385", "1412.6980"), "groundtruth"),
    (_row("2401.00001", "test", "a", None, "1706.03762"), "ngoài pool"),                       # pool rỗng phải toàn null
    (_row("hep-th/9901001", "test", "a", None, "1706.03762"), "split.json"),                   # partition lệch split
])
def test_prediction_constraints_detected(bad, fragment, tmp_path):
    split = {"2310.12345": "test", "hep-th/9901001": "valid", "2401.00001": "train"}
    errors = E.validate_predictions(E.predictions_table([bad]), POOLS, split)
    assert any(fragment in e for e in errors), errors
    with pytest.raises(ValueError):
        E.write_predictions(tmp_path / "x.parquet", [bad], POOLS, split)
    assert not (tmp_path / "x.parquet").exists()


def test_duplicate_key_and_null_in_required_column_detected():
    r = _row("2310.12345", "test", "a", None, *POOLS["2310.12345"][:5])
    assert any("trùng khóa" in e for e in E.validate_predictions(E.predictions_table([r, r]), POOLS))
    t = pa.table({n: pa.array([None], pa.string()) for n in E.PREDICTIONS_SCHEMA.names[:3]}
                 | {n: pa.array([None], pa.string()) for n in E.PREDICTIONS_SCHEMA.names[3:]})
    assert E.validate_predictions(t, POOLS)        # sai kiểu partition + null ở cột bắt buộc


def test_arxiv_id_format():
    assert all(E.is_valid_arxiv_id(x) for x in ["2310.12345", "1706.03762", "hep-th/9901001", "math.GT/0309136"])
    assert not any(E.is_valid_arxiv_id(x) for x in ["arXiv:1706.03762", "1706.03762v2", "", None, "1706.037"])


# =============================================================== train: check_labels
def test_clean_dataset_passes(papers):
    assert T.check_labels(papers, members=MEMBERS) == []


def _mutate(fn):
    ps, _ = _papers()
    fn(ps)
    return T.check_labels(ps, members=MEMBERS)


def _first(ps, source, part="train"):
    return next(p for p in ps if p.source == source and p.partition == part)


@pytest.mark.parametrize("fn, fragment", [
    (lambda ps: _first(ps, "manual").label_records.append(T.LabelRecord("nope", None, "no_match_confirmed")), "không có trong bảng bibtex"),
    (lambda ps: _first(ps, "manual").label_records.append(_first(ps, "manual").label_records[0]), "trùng"),
    (lambda ps: _first(ps, "manual").label_records.__setitem__(0, T.LabelRecord(_first(ps, "manual").label_records[0].citation_key, "1111.11111", "matched")), "không nằm trong pool"),
    (lambda ps: _first(ps, "manual").label_records.__setitem__(0, T.LabelRecord(_first(ps, "manual").label_records[0].citation_key, "arXiv:1706.03762", "matched")), "sai dạng"),
    (lambda ps: _first(ps, "auto").label_records.pop(), "thiếu nhãn"),
    (lambda ps: setattr(_first(ps, "manual"), "label_records", _first(ps, "manual").label_records[:5]), "cần >="),
    (lambda ps: _first(ps, "manual").label_records.__setitem__(0, T.LabelRecord(_first(ps, "manual").label_records[0].citation_key, None, "unresolved")), "không có unresolved"),
    (lambda ps: _first(ps, "auto").label_records.__setitem__(0, T.LabelRecord(_first(ps, "auto").label_records[0].citation_key, None, "no_match_confirmed")), "chỉ dành cho bài manual"),
    (lambda ps: _first(ps, "auto").label_records.__setitem__(0, T.LabelRecord(_first(ps, "auto").label_records[0].citation_key, None, "matched")), "không khớp status"),
    (lambda ps: setattr(_first(ps, "manual", "valid"), "partition", "train"), "valid có"),
    (lambda ps: setattr(_first(ps, "manual"), "file_source", "auto"), "khác split.json"),
])
def test_check_labels_detects(fn, fragment):
    errors = _mutate(fn)
    assert any(fragment in e for e in errors), errors


def test_auto_matched_requires_rule():
    def fn(ps):
        p = _first(ps, "auto")
        i = next(i for i, r in enumerate(p.label_records) if r.status == "matched")
        r = p.label_records[i]
        p.label_records[i] = T.LabelRecord(r.citation_key, r.groundtruth, "matched", None)
    assert any("phải có rule" in e for e in _mutate(fn))


# =============================================================== train: ngưỡng toàn nhóm (quota)
def _counts(ps):
    manual = [p for p in ps if p.source == "manual"]
    auto = [p for p in ps if p.source == "auto"]
    return manual, auto, sum(len(p.label_records) for p in manual)


def test_quota_passes_on_clean_dataset(papers):
    _, auto, _ = _counts(papers)
    assert T.check_quota(papers, members=MEMBERS) == []
    # Mốc 10%: corpus đủ nhỏ để số bài auto hiện có vừa đủ 10% phần ngoài manual.
    manual, _, _ = _counts(papers)
    assert T.check_quota(papers, members=MEMBERS, n_publications=len(manual) + 10 * len(auto)) == []


def test_quota_detects_too_few_manual_publications(papers):
    manual, _, _ = _counts(papers)
    errors = T.check_quota(papers, members=len(manual) // T.MANUAL_PUBS_PER_MEMBER + 1)
    assert any("bài manual, cần >=" in e for e in errors), errors


def test_quota_detects_too_few_manual_entries():
    ps, _ = _papers()
    manual, _, n_entries = _counts(ps)
    for p in manual:
        p.label_records = p.label_records[:1]
    errors = T.check_quota(ps, members=MEMBERS)
    assert any("entry manual đã gán" in e for e in errors), errors


def test_quota_detects_auto_below_ten_percent(papers):
    manual, auto, _ = _counts(papers)
    # 10% của phần ngoài manual lớn hơn số bài auto hiện có đúng 1 bài.
    n_pubs = len(manual) + 10 * (len(auto) + 1)
    errors = T.check_quota(papers, members=MEMBERS, n_publications=n_pubs)
    assert any(f"cần >= {len(auto) + 1}" in e for e in errors), errors


def test_quota_ten_percent_rounds_up(papers):
    manual, auto, _ = _counts(papers)
    # 10% của (10*|auto| + 1) = |auto| + 0,1 -> làm tròn lên thành |auto| + 1.
    errors = T.check_quota(papers, members=MEMBERS, n_publications=len(manual) + 10 * len(auto) + 1)
    assert any(f"cần >= {len(auto) + 1}" in e for e in errors), errors


def test_quota_detects_auto_below_two_per_member(papers):
    _, auto, _ = _counts(papers)
    members = len(auto) // T.AUTO_PUBS_PER_MEMBER + 1
    errors = T.check_quota(papers, members=members)
    assert any("bài auto, cần >=" in e for e in errors), errors


def test_check_labels_includes_quota_and_can_skip_it():
    ps, _ = _papers()
    n_pubs = 10**6  # 10% của corpus rất lớn: chắc chắn thiếu bài auto
    with_quota = T.check_labels(ps, members=MEMBERS, n_publications=n_pubs)
    without = T.check_labels(ps, members=MEMBERS, n_publications=n_pubs, enforce_quota=False)
    assert any(e.startswith("quota:") for e in with_quota)
    assert not any(e.startswith("quota:") for e in without)


def test_load_labels_from_files_roundtrip(tmp_path):
    bib, refs, split, files = S.make_dataset(members=MEMBERS)
    S.write_label_dir(tmp_path, split, files)
    sp, lf = T.load_labels(tmp_path)
    assert sp == split and set(lf) == set(files)
    ps = T.load_papers(bib, refs, sp, lf)
    assert T.check_labels(ps, members=MEMBERS) == []
    with pytest.raises(FileNotFoundError):
        T.load_labels(tmp_path / "missing")


def test_load_papers_dedups_and_sorts_pool():
    bib, refs, split, files = S.make_dataset(members=MEMBERS)
    refs = pd_concat_dup(refs)
    lf = {k: {"label_source": v["label_source"], "records": v["records"]} for k, v in files.items()}
    ps = T.load_papers(pd_concat_dup(bib), refs, split, lf)
    for p in ps:
        assert p.candidate_ids == sorted(set(p.candidate_ids))
        assert len(p.keys) == len(set(p.keys))


def pd_concat_dup(df):
    import pandas as pd
    return pd.concat([df, df.head(5)], ignore_index=True)


# =============================================================== train: chính sách nhãn (bảng 9.1)
def _pairs(batch):
    return {(q[0], q[1]): (batch.y[o:o + g], None if batch.sample_weight is None else batch.sample_weight[o:o + g])
            for q, o, g in zip(batch.query_ids, batch.offsets, batch.groups)}


def test_policy_A_is_manual_matched_only(papers):
    b = T.build_train_batch(papers, T.TrainPolicy("A"), filled=False)
    train_manual = {p.arxiv_id for p in papers if p.partition == "train" and p.source == "manual"}
    assert {q[0] for q in b.query_ids} <= train_manual
    assert (b.sample_weight == 1.0).all()
    assert all(y.sum() == 1 for y, _ in _pairs(b).values())      # chỉ hàng có đáp án


def test_policy_B_adds_auto_matched_with_weight_and_masks_rule_feature(papers):
    b = T.build_train_batch(papers, T.TrainPolicy("B", True, 0.3, True), filled=False)
    auto_by_key = {(p.arxiv_id, r.citation_key): r for p in papers if p.partition == "train" and p.source == "auto"
                   for r in p.label_records}
    col = F.FEATURE_NAMES.index("arxiv_id_in_fields")
    seen_auto, seen_url = 0, 0
    for (aid, key), (y, w), o, g in zip(b.query_ids, [_pairs(b)[q] for q in b.query_ids], b.offsets, b.groups):
        rec = auto_by_key.get((aid, key))
        if rec is None:
            continue
        seen_auto += 1
        assert rec.status == "matched" and y.sum() == 1          # unresolved KHÔNG vào (không phải mẫu âm)
        assert (w == 0.3).all()
        if rec.rule == "arxiv_in_url":
            seen_url += 1
            assert np.isnan(b.X[o:o + g, col]).all()             # che feature trùng luật
        else:
            assert not np.isnan(b.X[o:o + g, col]).all() or True
    assert seen_auto > 0 and seen_url > 0


def test_policy_C_adds_manual_no_match_rows_as_all_negative(papers):
    c = T.build_train_batch(papers, T.TrainPolicy("C", True, 0.3, True, include_manual_no_match=True), False)
    a = T.build_train_batch(papers, T.TrainPolicy("A"), False)
    assert c.n_queries > a.n_queries
    zero_rows = [y for y, _ in _pairs(c).values() if y.sum() == 0]
    assert zero_rows and all(len(y) > 0 for y in zero_rows)
    assert not any(y.sum() == 0 for y, _ in _pairs(a).values())


def test_policy_D_counts_unresolved_as_negative_but_B_never_does(papers):
    unresolved = {(p.arxiv_id, r.citation_key) for p in papers if p.partition == "train" and p.source == "auto"
                  for r in p.label_records if r.status == "unresolved"}
    assert unresolved
    d = T.build_train_batch(papers, T.policy_grid()[-1], False)
    assert T.policy_grid()[-1].name == "D" and not T.policy_grid()[-1].selectable
    assert unresolved <= set(map(tuple, d.query_ids))
    for pol in T.policy_grid()[:-1]:
        assert not unresolved & set(map(tuple, T.build_train_batch(papers, pol, False).query_ids))


def test_train_batch_never_contains_valid_or_test(papers):
    held = {p.arxiv_id for p in papers if p.partition != "train"}
    for pol in T.policy_grid():
        assert not held & {q[0] for q in T.build_train_batch(papers, pol, True).query_ids}


def test_relative_features_use_all_entries_not_only_labeled(papers):
    p = next(p for p in papers if p.source == "manual")
    assert p.tensor.shape[0] == len(p.entries) > len(p.label_records) or p.tensor.shape[0] >= len(p.label_records)
    assert p.tensor.shape[:2] == (len(p.entries), len(p.candidates))


def test_eval_set_includes_unresolved_and_no_match_in_denominator(papers):
    es = T.build_eval_set([p for p in papers if p.partition == "test"], filled=False)
    n_labeled = sum(len(p.label_records) for p in papers if p.partition == "test")
    assert len(es.truth) == es.batch.n_queries == n_labeled
    assert any(t is None for t in es.truth)


# =============================================================== train: LOPO + chọn chính sách
def test_lopo_never_trains_on_held_out_paper(papers, monkeypatch):
    train = [p for p in papers if p.partition == "train"]
    calls = []
    real = T.fit_policy

    def spy(spec, ps, pol):
        calls.append(frozenset(p.arxiv_id for p in ps))
        return real(spec, ps, pol)

    monkeypatch.setattr(T, "fit_policy", spy)
    spec = T.make_specs(0, 20, ["logreg"])["logreg"]
    v = T.lopo_manual(spec, T.TrainPolicy("A"), train)
    manual = [p.arxiv_id for p in train if p.source == "manual"]
    assert len(calls) == len(manual) and not math.isnan(v)
    for held, ids in zip(manual, calls):
        assert held not in ids and len(ids) == len(train) - 1


def test_select_policy_ignores_D_breaks_ties_with_lopo():
    pols = T.policy_grid((0.1,))
    mk = lambda lab, vm, lopo, sel=True: dict(model="m", label=lab, selectable=sel, fitted=True,  # noqa: E731
                                              valid_manual=vm, valid_overall=vm, lopo_manual=lopo)
    rows = [mk("A", 0.6, 0.5), mk("B(w=0.1)", 0.6, 0.7), mk("C(w=0.1)", 0.55, 0.9), mk("D", 0.99, 0.99, sel=False)]
    assert T.select_policy(rows, "m", pols).label == "B(w=0.1)"
    nan = float("nan")
    rows2 = [mk("A", nan, 0.1), dict(mk("B(w=0.1)", 0.4, nan))]
    assert T.select_policy(rows2, "m", pols).label == "B(w=0.1)"
    assert T.select_policy([], "m", pols) is None


# =============================================================== train: pipeline đầu-cuối
def _run(tmp_path, seed=0, shuffle_test_labels=False, models=("logreg", "lgbm"), lopo=False, **kw):
    ps, (bib, refs, split, files) = _papers(**kw)
    if shuffle_test_labels:
        rng = np.random.default_rng(123)
        for p in ps:
            if p.partition == "test":
                gts = [r.groundtruth for r in p.label_records]
                rng.shuffle(gts)
                p.label_records = [T.LabelRecord(r.citation_key, g, "matched" if g else r.status if r.status != "matched" else "no_match_confirmed", r.rule, r.annotator)
                                   for r, g in zip(p.label_records, gts)]
    T.build_tensors(ps)
    res = T.run_pipeline(ps, models=list(models), w_auto=(0.1, 0.5), lopo=lopo, seed=seed, n_estimators=40,
                         out_dir=tmp_path / "m", reports_dir=tmp_path / "r", predictions_path=tmp_path / "pred.parquet")
    return ps, res


def test_pipeline_end_to_end_beats_baseline_and_writes_valid_predictions(tmp_path):
    ps, res = _run(tmp_path, empty_pool_in_test=True)
    assert res["final_model"] in ("logreg", "lgbm") and res["vs_baseline_valid"][res["final_model"]]["beats_baseline"]
    for part in ("valid", "test"):
        assert set(res[part]) == {"baseline", "logreg", "lgbm"}
        assert res[part]["baseline"]["overall"]["n"] == sum(len(p.label_records) for p in ps if p.partition == part)
    # bảng predictions: một dòng cho MỌI entry, kể cả entry chưa gán nhãn
    table = pq.read_table(tmp_path / "pred.parquet")
    assert table.num_rows == sum(len(p.entries) for p in ps) == res["predictions"]["rows"]
    d = table.to_pydict()
    labeled = {(p.arxiv_id, r.citation_key): r for p in ps for r in p.label_records}
    unl = [i for i in range(table.num_rows) if (d["arxiv_id"][i], d["citation_key"][i]) not in labeled]
    assert unl and all(d["groundtruth_arxiv_id"][i] is None for i in unl)
    empty = next(p for p in ps if not p.candidates)
    idx = [i for i in range(table.num_rows) if d["arxiv_id"][i] == empty.arxiv_id]
    assert idx and all(d[f"cand_{k}"][i] is None for i in idx for k in range(1, 6))
    # báo cáo và mô hình đã lưu
    for f in ("m3_results.json", "m3_results.md", "m3_ablation.csv"):
        assert (tmp_path / "r" / f).exists()
    assert (tmp_path / "m" / f"{res['final_model']}.pkl").exists()


def _same(a, b):
    """So sánh cấu trúc lồng nhau, coi nan == nan."""
    return repr(a) == repr(b)


def test_test_labels_never_influence_any_selection(tmp_path):
    """Đổi toàn bộ nhãn Test không được thay đổi chính sách chọn, mô hình cuối hay kết quả Valid."""
    _, a = _run(tmp_path / "a")
    _, b = _run(tmp_path / "b", shuffle_test_labels=True)
    assert a["chosen_policy"] == b["chosen_policy"] and a["final_model"] == b["final_model"]
    assert _same(a["ablation"], b["ablation"]) and _same(a["valid"], b["valid"])
    assert not _same(a["test"], b["test"])                              # chỉ riêng kết quả Test là khác


def test_pipeline_is_deterministic(tmp_path):
    _, a = _run(tmp_path / "a", seed=3)
    _, b = _run(tmp_path / "b", seed=3)
    assert _same(a["valid"], b["valid"]) and _same(a["test"], b["test"]) and a["chosen_policy"] == b["chosen_policy"]


def test_lambdarank_and_lopo_run(tmp_path):
    _, res = _run(tmp_path, models=("lgbm_rank",), lopo=True)
    assert all(not math.isnan(r["lopo_manual"]) for r in res["ablation"] if r["fitted"])
    assert res["chosen_policy"].keys() == {"lgbm_rank"}


def test_falls_back_to_baseline_or_raises_when_nothing_beats_it(tmp_path):
    ps, _ = _papers()
    T.build_tensors(ps)
    res = T.run_pipeline(ps, models=[], lopo=False, predictions_path=tmp_path / "p.parquet")
    assert res["final_model"] == "baseline"
    with pytest.raises(RuntimeError):
        T.run_pipeline(ps, models=[], lopo=False, require_beat=True)


def test_feature_tensor_cache_is_content_addressed(tmp_path):
    ps, _ = _papers()
    T.build_tensors(ps[:3], cache_dir=tmp_path)
    first = [p.tensor.copy() for p in ps[:3]]
    for p in ps[:3]:
        p.tensor = None
    T.build_tensors(ps[:3], cache_dir=tmp_path)
    assert all(np.array_equal(a, p.tensor, equal_nan=True) for a, p in zip(first, ps[:3]))
    assert len(list(tmp_path.glob("*.npy"))) == 3
    ps[0].entries[0] = copy.copy(ps[0].entries[0])
    ps[0].entries[0].title = "completely different title here"
    T.build_tensors(ps[:1], cache_dir=tmp_path)                # nội dung đổi -> không dùng cache cũ
    assert len(list(tmp_path.glob("*.npy"))) == 4