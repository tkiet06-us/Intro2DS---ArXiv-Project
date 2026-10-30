"""Test contract mô hình xếp hạng trong ``src.m3_matching.matching_model``.

Chạy: ``pytest tests/m3_matching/test_matching_model.py -q``
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.m3_matching.matching_model import (
    PAD,
    FeatureRankBaseline,
    MatchingModel,
    NotFittedError,
    PointwiseClassifierModel,
    RankingBatch,
    indices_to_ids,
    rank_groups,
)

NAN = np.nan


# ------------------------------------------------------------------ mô hình giả
class SumModel(MatchingModel):
    """Mô hình luật: điểm = tổng feature."""

    requires_fit = False

    def _score(self, X):
        return X.sum(axis=1)


class RecordingModel(MatchingModel):
    """Ghi lại đúng những gì _fit nhận; điểm = cột đầu."""

    seen = None

    def _fit(self, X, y, groups, sample_weight):
        self.seen = (
            X.copy(),
            y.copy(),
            groups.copy(),
            None if sample_weight is None else sample_weight.copy(),
        )

    def _score(self, X):
        return X[:, 0]


class BrokenFit(MatchingModel):
    def _fit(self, X, y, groups, sample_weight):
        raise RuntimeError("boom")

    def _score(self, X):
        return X[:, 0]


class WrongShape(MatchingModel):
    requires_fit = False

    def _score(self, X):
        return X[:, 0][:-1]


class InfScore(MatchingModel):
    requires_fit = False

    def _score(self, X):
        return np.full(X.shape[0], np.inf)


# ------------------------------------------------------------------ tiện ích
def make_batch(pools, *, names=None, y=None, qids=None, weights=None):
    """``pools``: danh sách pool; mỗi pool là list các tuple ``(ref_id, f0, f1, ...)``."""
    groups = [len(p) for p in pools]
    rows = [r for p in pools for r in p]
    ids = [r[0] for r in rows]
    n_feat = len(rows[0]) - 1 if rows else (len(names) if names else 1)
    X = np.array([r[1:] for r in rows], dtype=float).reshape(len(rows), n_feat)
    return RankingBatch(
        X, groups, y=y, sample_weight=weights, candidate_ids=ids, query_ids=qids,
        feature_names=names,
    )


def mrr_at_k(ranked_ids, truth, k=5):
    """MRR@k theo mục 7: mọi entry đã gán nhãn nằm trong mẫu số; không đáp án thì 0."""
    total = 0.0
    for row, t in zip(ranked_ids, truth):
        row = list(row)[:k]
        if t is not None and t in row:
            total += 1.0 / (row.index(t) + 1)
    return total / len(truth)


def reference_rank(scores, groups, k, tb=None, ids=None, min_score=None):
    """Cài đặt chậm, hiển nhiên đúng, để đối chiếu rank_groups."""
    out, start = [], 0
    for size in groups:
        cands = []
        for j in range(size):
            s = scores[start + j]
            nan = bool(np.isnan(s))
            cands.append(
                (nan, 0.0 if nan else -s, 0 if tb is None else tb[start + j], j,
                 None if ids is None else ids[start + j], s)
            )
        cands.sort(key=lambda t: t[:4])
        seen, res = set(), []
        for c in cands:
            if ids is not None:
                if c[4] in seen:
                    continue
                seen.add(c[4])
            if min_score is not None and not c[5] >= min_score:
                continue
            res.append(c[3])
        res = res[:k] + [PAD] * (k - len(res))
        out.append(res)
        start += size
    return np.array(out, dtype=np.intp).reshape(len(groups), k)


# ================================================================== RankingBatch
class TestRankingBatch:
    def test_dense_roundtrip(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(3, 4, 2))
        mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=bool)
        y = (rng.random((3, 4)) < 0.2).astype(float)
        y[1] = 0
        y[0] = 0
        y[2] = 0
        y[0, 1] = 1
        y[2, 3] = 1
        b = RankingBatch.from_dense(
            X, mask, y=y, sample_weight=[1.0, 0.5, 2.0],
            candidate_ids=["a", "b", "c", "d"], feature_names=("u", "v"),
        )
        assert b.groups.tolist() == [3, 2, 4]
        assert b.n_pairs == 9 and b.n_queries == 3 and b.n_features == 2
        assert b.sample_weight.tolist() == [1, 1, 1, 0.5, 0.5, 2, 2, 2, 2]
        assert b.candidate_ids.tolist() == list("abc") + list("ab") + list("abcd")
        dv = b.to_dense()
        assert dv.mask.shape == (3, 4) and (dv.mask == mask).all()
        assert np.allclose(dv.X[mask], X[mask])
        assert np.isnan(dv.X[~mask]).all()
        assert np.array_equal(dv.y[mask], y[mask])

    def test_from_dense_keeps_order_with_holes_in_mask(self):
        X = np.array([[[1.0], [99.0], [2.0], [3.0]]])
        b = RankingBatch.from_dense(X, [[True, False, True, True]])
        assert b.X[:, 0].tolist() == [1.0, 2.0, 3.0]
        assert b.groups.tolist() == [3]

    def test_from_dense_single_query_2d_inputs(self):
        X = np.arange(6.0).reshape(3, 2)  # (candidates, features)
        b = RankingBatch.from_dense(X, y=[0, 1, 0], sample_weight=[1, 2, 3])
        assert b.groups.tolist() == [3]
        assert b.sample_weight.tolist() == [1, 2, 3]

    def test_empty_batch_is_valid(self):
        b = RankingBatch(np.empty((0, 2)), [])
        assert b.n_queries == 0 and b.n_pairs == 0
        assert b.to_dense().X.shape == (0, 0, 2)

    def test_frozen(self):
        b = RankingBatch(np.zeros((1, 1)), [1])
        with pytest.raises(FrozenInstanceError):
            b.X = np.ones((1, 1))

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(X=np.array([[np.inf]]), groups=[1]),  # ±inf
            dict(X=np.zeros((2, 1)), groups=[1]),  # tổng groups lệch
            dict(X=np.zeros((1, 1)), groups=[2, -1]),  # groups âm
            dict(X=np.zeros((1, 1)), groups=[1.5]),  # groups không nguyên
            dict(X=np.zeros(3), groups=[3]),  # X không 2-D
            dict(X=np.zeros((2, 1)), groups=[2], y=[0.5, 0]),  # nhãn không nhị phân
            dict(X=np.zeros((2, 1)), groups=[2], y=[NAN, 0]),  # nhãn NaN
            dict(X=np.zeros((2, 1)), groups=[2], y=[0]),  # sai độ dài y
            dict(X=np.zeros((2, 1)), groups=[2], sample_weight=[1, -1]),
            dict(X=np.zeros((2, 1)), groups=[2], sample_weight=[1, NAN]),
            dict(X=np.zeros((2, 1)), groups=[2], candidate_ids=["a", None]),
            dict(X=np.zeros((2, 1)), groups=[2], candidate_ids=["a"]),
            dict(X=np.zeros((2, 2)), groups=[2], feature_names=("a",)),
            dict(X=np.zeros((2, 2)), groups=[2], feature_names=("a", "a")),
            dict(X=np.zeros((2, 1)), groups=[1, 1], query_ids=[("p", "k")]),
        ],
    )
    def test_validation_errors(self, kwargs):
        with pytest.raises(ValueError):
            RankingBatch(**kwargs)

    def test_require_labels_at_most_one_positive(self):
        ok = RankingBatch(np.zeros((4, 1)), [2, 2], y=[1, 0, 0, 0])
        ok.require_labels(1)
        bad = RankingBatch(np.zeros((4, 1)), [2, 2], y=[1, 1, 0, 0])
        with pytest.raises(ValueError, match="tối đa một đáp án"):
            bad.require_labels(1)
        bad.require_labels(None)  # tắt kiểm tra thì qua
        with pytest.raises(ValueError):
            RankingBatch(np.zeros((1, 1)), [1]).require_labels()

    def test_query_ids_keep_tuples(self):
        q = [("2310.1", "a"), ("2310.1", "b")]
        b = RankingBatch(np.zeros((2, 1)), [1, 1], query_ids=q)
        assert b.query_ids.shape == (2,)
        assert b.query_ids[1] == ("2310.1", "b")

    def test_select_queries(self):
        b = make_batch(
            [[("a", 1.0), ("b", 2.0)], [("c", 3.0)], [("d", 4.0), ("e", 5.0), ("f", 6.0)]],
            y=[1, 0, 0, 0, 0, 1], qids=["q0", "q1", "q2"], weights=[1, 1, 2, 3, 3, 3],
        )
        s = b.select_queries([True, False, True])
        assert s.groups.tolist() == [2, 3]
        assert s.X[:, 0].tolist() == [1, 2, 4, 5, 6]
        assert s.y.tolist() == [1, 0, 0, 0, 1]
        assert s.sample_weight.tolist() == [1, 1, 3, 3, 3]
        assert s.candidate_ids.tolist() == list("abdef")
        assert s.query_ids.tolist() == ["q0", "q2"]
        with pytest.raises(ValueError):
            b.select_queries([1, 0, 1])  # không phải bool
        with pytest.raises(ValueError):
            b.select_queries([True, False])

    def test_with_query_weights_replaces(self):
        b = make_batch([[("a", 1.0), ("b", 2.0)], [("c", 3.0)]], weights=[9, 9, 9])
        w = b.with_query_weights([0.3, 1.0])
        assert w.sample_weight.tolist() == [0.3, 0.3, 1.0]
        assert b.sample_weight.tolist() == [9, 9, 9]  # bản gốc không đổi
        with pytest.raises(ValueError):
            b.with_query_weights([1.0])

    def test_with_features_masked(self):
        b = make_batch(
            [[("a", 1.0, 1.0)], [("b", 2.0, 1.0), ("c", 3.0, 1.0)], [("d", 4.0, 1.0)]],
            names=("title", "F09"),
        )
        m = b.with_features_masked([False, True, False], ["F09"])
        assert np.isnan(m.X[1:3, 1]).all()
        assert not np.isnan(m.X[[0, 3], 1]).any()
        assert not np.isnan(m.X[:, 0]).any()
        assert not np.isnan(b.X).any()  # bản gốc không đổi
        assert not np.isnan(b.with_features_masked([True] * 3, [0]).X[:, 1]).any()
        with pytest.raises(KeyError):
            b.with_features_masked([True] * 3, ["nope"])
        with pytest.raises(ValueError):
            b.with_features_masked([1, 1, 1], ["F09"])

    def test_concat(self):
        b1 = make_batch([[("a", 1.0)]], names=("f",), y=[1], qids=[("p1", "k")])
        b2 = make_batch([[("b", 2.0), ("c", 3.0)], []], names=("f",), y=[0, 1],
                        qids=[("p2", "k"), ("p2", "z")])
        c = RankingBatch.concat([b1, b2])
        assert c.groups.tolist() == [1, 2, 0]
        assert c.query_ids.tolist() == [("p1", "k"), ("p2", "k"), ("p2", "z")]
        assert c.candidate_ids.tolist() == ["a", "b", "c"]
        with pytest.raises(ValueError):
            RankingBatch.concat([b1, make_batch([[("b", 2.0)]], names=("g",), y=[0],
                                                qids=[("p", "k")])])
        with pytest.raises(ValueError):  # một batch thiếu y
            RankingBatch.concat([b1, make_batch([[("b", 2.0)]], names=("f",),
                                                qids=[("p", "k")])])
        with pytest.raises(ValueError):
            RankingBatch.concat([])


# ================================================================== indices_to_ids
class TestIndicesToIds:
    def test_pad_becomes_none_not_last_candidate(self):
        ids = np.array(["a", "b", "c"], dtype=object)
        out = indices_to_ids([2, 0, PAD, PAD, PAD], ids)
        assert out.tolist() == ["c", "a", None, None, None]
        assert ids[PAD] == "c"  # chính là bẫy mà hàm này tránh

    def test_2d_shared_and_per_query(self):
        order = np.array([[1, PAD], [0, 1]])
        assert indices_to_ids(order, ["a", "b"]).tolist() == [["b", None], ["a", "b"]]
        per_q = [["a", "b"], ["x", "y"]]
        assert indices_to_ids(order, per_q).tolist() == [["b", None], ["x", "y"]]

    def test_empty_pool_and_range(self):
        assert indices_to_ids([[PAD, PAD]], np.empty((1, 0), dtype=object)).tolist() == [
            [None, None]
        ]
        with pytest.raises(IndexError):
            indices_to_ids([3], ["a", "b"])
        with pytest.raises(ValueError):
            indices_to_ids([[0], [0]], [["a"]])  # số hàng order và ids lệch nhau


# ================================================================== rank_groups
class TestRankGroups:
    def test_basic_order_pad_and_empty_group(self):
        scores = [0.2, 0.9, 0.5, 0.7]
        idx, top = rank_groups(scores, [3, 0, 1], k=4)
        assert idx.tolist() == [[1, 2, 0, PAD], [PAD] * 4, [0, PAD, PAD, PAD]]
        assert top[0, :3].tolist() == [0.9, 0.5, 0.2]
        assert np.isneginf(top[0, 3]) and np.isneginf(top[1]).all()
        assert top[2, 0] == 0.7

    def test_nan_after_real_scores_but_still_returned(self):
        idx, top = rank_groups([NAN, 0.1, NAN, 0.3], [4], k=5)
        assert idx[0].tolist() == [3, 1, 0, 2, PAD]  # NaN theo thứ tự vị trí
        assert np.isnan(top[0, 2]) and np.isnan(top[0, 3])

    def test_all_nan_group_is_still_ranked_by_tie_break(self):
        idx, _ = rank_groups([NAN, NAN, NAN], [3], k=5, tie_break=[2, 0, 1])
        assert idx[0].tolist() == [1, 2, 0, PAD, PAD]

    def test_ties_position_then_custom_key(self):
        assert rank_groups([1.0, 1.0, 1.0], [3], k=3)[0][0].tolist() == [0, 1, 2]
        idx, _ = rank_groups([1.0, 1.0, 1.0], [3], k=3, tie_break=[2, 0, 1])
        assert idx[0].tolist() == [1, 2, 0]

    def test_tie_break_does_not_override_score(self):
        idx, _ = rank_groups([0.1, 0.9], [2], k=2, tie_break=[0, 1])
        assert idx[0].tolist() == [1, 0]

    def test_min_score_drops_below_and_nan(self):
        idx, _ = rank_groups([0.9, 0.2, NAN, 0.6], [4], k=5, min_score=0.5)
        assert idx[0].tolist() == [0, 3, PAD, PAD, PAD]

    def test_dedup_keeps_best_scored_copy(self):
        idx, _ = rank_groups([0.2, 0.9, 0.5], [3], k=5, candidate_ids=["a", "a", "b"])
        assert idx[0].tolist() == [1, 2, PAD, PAD, PAD]

    def test_dedup_is_per_group(self):
        idx, _ = rank_groups([0.5, 0.4], [1, 1], k=2, candidate_ids=["a", "a"])
        assert idx.tolist() == [[0, PAD], [0, PAD]]

    def test_pad_only_trailing_and_k_truncates(self):
        idx, _ = rank_groups(np.arange(10.0), [10], k=3)
        assert idx[0].tolist() == [9, 8, 7]

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            rank_groups([1.0], [1], k=0)
        with pytest.raises(ValueError):
            rank_groups([1.0, 2.0], [1])
        with pytest.raises(ValueError):
            rank_groups([1.0, 2.0], [2], tie_break=[1])

    def test_no_queries(self):
        idx, top = rank_groups([], [], k=5)
        assert idx.shape == (0, 5) and top.shape == (0, 5)

    @pytest.mark.parametrize("seed", range(25))
    def test_matches_reference_on_random_input(self, seed):
        rng = np.random.default_rng(seed)
        groups = rng.integers(0, 9, size=rng.integers(1, 12))
        n = int(groups.sum())
        scores = rng.choice([0.0, 0.25, 0.5, 1.0, NAN], size=n)  # nhiều hòa điểm và NaN
        tb = rng.integers(0, 3, size=n)
        ids = rng.choice(list("abcdefgh"), size=n)
        k = int(rng.integers(1, 8))
        for use_tb, use_ids, ms in [(0, 0, None), (1, 0, None), (1, 1, None), (1, 1, 0.3)]:
            got, _ = rank_groups(
                scores, groups, k,
                tie_break=tb if use_tb else None,
                candidate_ids=ids if use_ids else None,
                min_score=ms,
            )
            want = reference_rank(
                scores, groups, k,
                tb=tb if use_tb else None, ids=ids if use_ids else None, min_score=ms,
            )
            assert np.array_equal(got, want), (seed, use_tb, use_ids, ms)
            # PAD chỉ nằm ở cuối hàng
            for row in got:
                pads = row == PAD
                assert not (pads[:-1] & ~pads[1:]).any()


# ================================================================== hợp đồng MatchingModel (dense)
class TestDenseContract:
    def test_rank_maps_back_to_dense_columns(self):
        X = np.array([[[3.0], [99.0], [1.0], [2.0]]])
        out = SumModel().rank(X, k=5, mask=[[True, False, True, True]])
        assert out.tolist() == [[0, 3, 2, PAD, PAD]]  # cột 1 (đệm) không bao giờ xuất hiện

    def test_single_query_2d_squeezes(self):
        X = np.array([[1.0], [3.0], [2.0]])
        assert SumModel().rank(X, k=2).tolist() == [1, 2]
        assert SumModel().predict_scores(X).shape == (3,)

    def test_predict_scores_padding_neg_inf_nan_preserved(self):
        X = np.array([[[1.0], [NAN], [5.0]], [[2.0], [3.0], [4.0]]])
        m = np.array([[True, True, False], [True, True, True]])
        s = SumModel().predict_scores(X, mask=m)
        assert s.shape == (2, 3)
        assert s[0, 0] == 1.0 and np.isnan(s[0, 1]) and np.isneginf(s[0, 2])
        assert s[1].tolist() == [2.0, 3.0, 4.0]

    def test_padding_cells_never_reach_score_hook(self):
        class Spy(SumModel):
            def _score(self, X):
                assert np.isfinite(X).all(), "ô đệm lọt vào _score"
                return X.sum(axis=1)

        X = np.array([[[1.0], [np.inf], [2.0]]])  # inf nằm ở ô đệm
        out = Spy().rank(X, mask=[[True, False, True]], k=3)
        assert out.tolist() == [[2, 0, PAD]]

    def test_dense_equals_ragged_on_random_data(self):
        rng = np.random.default_rng(3)
        X = rng.integers(0, 3, size=(5, 7, 2)).astype(float)  # nhiều hòa điểm
        M = rng.random((5, 7)) < 0.7
        dense = SumModel().rank(X, k=4, mask=M)
        ragged = SumModel().rank(RankingBatch.from_dense(X, M), k=4)
        for q in range(5):
            cols = np.flatnonzero(M[q])
            want = np.where(ragged[q] >= 0, cols[np.where(ragged[q] >= 0, ragged[q], 0)]
                            if cols.size else 0, PAD)
            assert dense[q].tolist() == want.tolist()

    def test_dense_tie_break_array(self):
        X = np.ones((1, 3, 1))
        assert SumModel().rank(X, k=3).tolist() == [[0, 1, 2]]
        assert SumModel().rank(X, k=3, tie_break=[2, 1, 0]).tolist() == [[2, 1, 0]]
        assert SumModel().rank(X, k=3, tie_break=[[1, 2, 0]]).tolist() == [[2, 0, 1]]

    def test_return_scores_and_k_larger_than_pool(self):
        X = np.array([[[0.5], [0.25]]])
        idx, top = SumModel().rank(X, k=4, return_scores=True)
        assert idx.tolist() == [[0, 1, PAD, PAD]]
        assert top[0, :2].tolist() == [0.5, 0.25] and np.isneginf(top[0, 2:]).all()

    def test_k_validation(self):
        with pytest.raises(ValueError):
            SumModel().rank(np.ones((1, 2, 1)), k=0)

    def test_bad_shapes(self):
        with pytest.raises(ValueError):
            SumModel().predict_scores(np.ones(4))
        with pytest.raises(ValueError):
            SumModel().predict_scores(np.ones((1, 2, 1)), mask=np.ones((1, 3), bool))
        with pytest.raises(ValueError):
            SumModel().rank(RankingBatch(np.ones((1, 1)), [1]), mask=[[True]])

    def test_empty_inputs(self):
        m = SumModel()
        assert m.rank(np.empty((0, 3, 2)), k=5).shape == (0, 5)
        assert m.rank(np.empty((2, 0, 2)), k=5).tolist() == [[PAD] * 5, [PAD] * 5]
        assert m.predict_scores(np.empty((2, 0, 2))).shape == (2, 0)


# ================================================================== fit / hooks / trạng thái
class TestFit:
    def test_dense_fit_flattens_valid_cells_and_groups(self):
        X = np.arange(12.0).reshape(3, 2, 2)
        y = np.array([[1, 0], [0, 0], [0, 0]], dtype=float)
        mask = np.array([[True, True], [True, False], [False, False]])
        m = RecordingModel().fit(X, y, mask=mask, sample_weight=[0.5, 2.0, 9.0],
                                 feature_names=["a", "b"])
        Xs, ys, groups, w = m.seen
        assert Xs.shape == (3, 2) and Xs[:, 0].tolist() == [0.0, 2.0, 4.0]
        assert ys.tolist() == [1, 0, 0]
        assert groups.tolist() == [2, 1]  # hàng không có ô thật bị bỏ
        assert w.tolist() == [0.5, 0.5, 2.0]
        assert m.is_fitted_ and m.n_features_in_ == 2 and m.feature_names == ("a", "b")

    def test_batch_fit_equivalent(self):
        b = make_batch([[("a", 1.0), ("b", 2.0)], [], [("c", 3.0)]], y=[0, 1, 1],
                       names=("f",), weights=[1, 1, 3])
        m = RecordingModel().fit(b)
        assert m.seen[2].tolist() == [2, 1]  # nhóm rỗng bị bỏ khỏi groups
        assert m.seen[3].tolist() == [1, 1, 3]

    def test_batch_fit_rejects_extra_args(self):
        b = make_batch([[("a", 1.0)]], y=[1])
        with pytest.raises(ValueError):
            RecordingModel().fit(b, y=[1])

    def test_fit_requires_labels_and_pairs(self):
        with pytest.raises(ValueError):
            RecordingModel().fit(np.ones((1, 2, 1)))  # thiếu y
        with pytest.raises(ValueError):
            RecordingModel().fit(make_batch([[("a", 1.0)]]))  # batch không có y
        with pytest.raises(ValueError, match="không có cặp"):
            RecordingModel().fit(make_batch([[]], y=[]))

    def test_fit_rejects_multiple_positives_unless_disabled(self):
        b = make_batch([[("a", 1.0), ("b", 2.0)]], y=[1, 1])
        with pytest.raises(ValueError, match="tối đa một đáp án"):
            RecordingModel().fit(b)

        class Multi(RecordingModel):
            max_positives_per_query = None

        Multi().fit(b)

    def test_dense_fit_label_errors(self):
        X = np.ones((2, 3, 1))
        with pytest.raises(ValueError):
            RecordingModel().fit(X, np.zeros((2, 2)))
        with pytest.raises(ValueError):
            RecordingModel().fit(X, np.full((2, 3), 0.5))
        with pytest.raises(ValueError):
            RecordingModel().fit(X, np.zeros((2, 3)), sample_weight=np.ones((2, 2)))
        with pytest.raises(ValueError):
            RecordingModel().fit(X, np.zeros((2, 3)), feature_names=["a", "b"])

    def test_not_fitted_error(self):
        with pytest.raises(NotFittedError):
            RecordingModel().rank(np.ones((1, 2, 1)))
        with pytest.raises(NotFittedError):
            RecordingModel().predict_scores(np.ones((1, 2, 1)))
        SumModel().rank(np.ones((1, 2, 1)))  # mô hình luật không cần fit

    def test_failed_fit_never_leaves_fitted_flag(self):
        m = BrokenFit()
        with pytest.raises(RuntimeError):
            m.fit(np.ones((1, 2, 1)), np.array([[1.0, 0.0]]))
        assert not m.is_fitted_
        ok = RecordingModel().fit(np.ones((1, 2, 1)), np.array([[1.0, 0.0]]))
        assert ok.is_fitted_

        class Flaky(RecordingModel):
            fail = False

            def _fit(self, *a):
                if self.fail:
                    raise RuntimeError("boom")
                super()._fit(*a)

        f = Flaky().fit(np.ones((1, 2, 1)), np.array([[1.0, 0.0]]))
        f.fail = True
        with pytest.raises(RuntimeError):
            f.fit(np.ones((1, 2, 1)), np.array([[1.0, 0.0]]))
        assert not f.is_fitted_  # lần fit lỗi sau không được giữ trạng thái cũ

    def test_score_hook_contract_enforced(self):
        X = np.ones((1, 3, 1))
        with pytest.raises(ValueError, match="shape"):
            WrongShape().predict_scores(X)
        with pytest.raises(ValueError, match="inf"):
            InfScore().predict_scores(X)


# ================================================================== feature names
class TestFeatureChecks:
    def test_order_mismatch_detected(self):
        b = make_batch([[("a", 1.0, 2.0), ("b", 2.0, 1.0)]], y=[1, 0], names=("x", "y"))
        m = RecordingModel().fit(b)
        swapped = make_batch([[("a", 2.0, 1.0)]], names=("y", "x"))
        with pytest.raises(ValueError, match="không khớp"):
            m.rank(swapped)
        m.rank(make_batch([[("a", 2.0, 1.0)]], names=("x", "y")))  # đúng thứ tự thì qua
        m.rank(make_batch([[("a", 2.0, 1.0)]]))  # không có tên thì chỉ kiểm tra số cột

    def test_dense_predict_accepts_feature_names(self):
        b = make_batch([[("a", 1.0, 2.0), ("b", 2.0, 1.0)]], y=[1, 0], names=("x", "y"))
        m = RecordingModel().fit(b)
        X = np.ones((1, 2, 2))
        m.rank(X, feature_names=("x", "y"))
        with pytest.raises(ValueError):
            m.rank(X, feature_names=("y", "x"))

    def test_column_count_mismatch(self):
        m = RecordingModel().fit(np.ones((1, 2, 3)), np.array([[1.0, 0.0]]))
        with pytest.raises(ValueError, match="3 feature"):
            m.rank(np.ones((1, 2, 2)))

    def test_rule_model_adopts_names_from_data(self):
        m = FeatureRankBaseline("title_lev_sim")
        b = make_batch([[("a", 0.1, 5.0), ("b", 0.9, 5.0)]], names=("title_lev_sim", "other"))
        assert m.rank_ids(b, k=2).tolist() == [["b", "a"]]
        assert m.feature_names == ("title_lev_sim", "other")
        with pytest.raises(ValueError):  # sau khi nhận tên thì giữ nghiêm thứ tự
            m.rank(make_batch([[("a", 0.1, 5.0)]], names=("other", "title_lev_sim")))


# ================================================================== ID, tie-break, Naive baseline
class TestNaiveBaseline:
    IDS = ["2001.0003", "1706.03762", "1409.0473", "1810.04805", "1512.03385", "1412.6980",
           "hep-th/9901001"]
    F01 = [0.9, NAN, 0.9, 0.5, NAN, 0.2, 0.1]

    def pool(self, order=None):
        order = list(range(len(self.IDS))) if order is None else order
        return [(self.IDS[i], self.F01[i]) for i in order]

    def test_spec_ordering_nan_last_and_id_tie_break(self):
        b = make_batch([self.pool()], names=("title_lev_sim",))
        m = FeatureRankBaseline()
        assert m.rank_ids(b, k=7)[0].tolist() == [
            "1409.0473", "2001.0003",  # hòa 0.9: ID tăng dần
            "1810.04805", "1412.6980", "hep-th/9901001",
            "1512.03385", "1706.03762",  # F01 = NaN xếp sau cùng, cũng theo ID tăng dần
        ]
        assert m.rank_ids(b, k=5)[0].tolist() == [
            "1409.0473", "2001.0003", "1810.04805", "1412.6980", "hep-th/9901001"
        ]

    def test_result_independent_of_pool_row_order(self):
        base = FeatureRankBaseline().rank_ids(
            make_batch([self.pool()], names=("title_lev_sim",)), k=7)
        rng = np.random.default_rng(0)
        for _ in range(10):
            perm = rng.permutation(len(self.IDS)).tolist()
            got = FeatureRankBaseline().rank_ids(
                make_batch([self.pool(perm)], names=("title_lev_sim",)), k=7)
            assert got.tolist() == base.tolist()

    def test_position_tie_break_depends_on_order(self):
        b = make_batch([[("z", 0.5), ("a", 0.5)]], names=("title_lev_sim",))
        m = FeatureRankBaseline()
        assert m.rank_ids(b, k=2, tie_break="position")[0].tolist() == ["z", "a"]
        assert m.rank_ids(b, k=2)[0].tolist() == ["a", "z"]  # auto = theo ID

    def test_short_and_empty_pools_give_none(self):
        b = make_batch([[("b", 0.5), ("a", 0.7)], [], [("c", NAN)]], names=("title_lev_sim",))
        out = FeatureRankBaseline().rank_ids(b, k=5)
        assert out.shape == (3, 5)
        assert out[0].tolist() == ["a", "b", None, None, None]
        assert out[1].tolist() == [None] * 5  # pool rỗng: cả năm cột null
        assert out[2].tolist() == ["c", None, None, None, None]

    def test_no_gaps_and_no_duplicates(self):
        b = make_batch(
            [[("a", 0.5), ("a", 0.9), ("b", NAN), ("b", NAN), ("c", 0.1)]],
            names=("title_lev_sim",),
        )
        row = FeatureRankBaseline().rank_ids(b, k=5)[0].tolist()
        assert row == ["a", "c", "b", None, None]
        filled = [x for x in row if x is not None]
        assert len(set(filled)) == len(filled)
        assert row[: len(filled)] == filled  # null chỉ ở cuối

    def test_dedup_picks_best_scored_copy(self):
        b = make_batch([[("a", 0.2), ("a", 0.9), ("b", 0.5)]], names=("title_lev_sim",))
        m = FeatureRankBaseline()
        idx = m.rank(b, k=3)
        assert idx[0].tolist() == [1, 2, PAD]  # bản a có điểm 0.9 (hàng 1) được giữ

    def test_column_lookup_variants(self):
        b = make_batch([[("a", 1.0, 9.0), ("b", 2.0, 3.0)]], names=("p", "q"))
        assert FeatureRankBaseline("q").rank_ids(b, k=2)[0].tolist() == ["a", "b"]
        assert FeatureRankBaseline(1).rank_ids(b, k=2)[0].tolist() == ["a", "b"]
        assert FeatureRankBaseline("q", higher_is_better=False).rank_ids(b, k=2)[0].tolist() == [
            "b", "a"]
        with pytest.raises(KeyError):
            FeatureRankBaseline("missing").rank_ids(b)
        with pytest.raises(IndexError):
            FeatureRankBaseline(5).rank_ids(b)
        assert FeatureRankBaseline("q", feature_names=("p", "q")).rank(
            np.array([[[1.0, 0.0], [0.0, 1.0]]]), k=2).tolist() == [[1, 0]]

    def test_feature_importance(self):
        b = make_batch([[("a", 1.0, 9.0)]], names=("p", "q"))
        m = FeatureRankBaseline("q")
        assert m.feature_importance() is None
        m.rank_ids(b)
        assert m.feature_importance() == {"p": 0.0, "q": 1.0}

    def test_rank_ids_requires_ids(self):
        with pytest.raises(ValueError):
            SumModel().rank_ids(RankingBatch(np.ones((1, 1)), [1]))
        with pytest.raises(ValueError):
            SumModel().rank_ids(np.ones((1, 2, 1)))
        with pytest.raises(ValueError):
            SumModel().rank(RankingBatch(np.ones((1, 1)), [1]), tie_break="id")

    def test_return_scores(self):
        b = make_batch([[("a", 0.5), ("b", NAN)]], names=("title_lev_sim",))
        ids, top = FeatureRankBaseline().rank_ids(b, k=3, return_scores=True)
        assert ids[0].tolist() == ["a", "b", None]
        assert top[0, 0] == 0.5 and np.isnan(top[0, 1]) and np.isneginf(top[0, 2])

    def test_mrr_matches_spec_example(self):
        # Mục 7: A -> cand_2 (0.5), B -> cand_1 (1), C ngoài top-5 (0), D no-match (0)
        ranked = [["x", "A", "y", "z", "w"], ["B", "x", "y", "z", "w"],
                  ["p", "q", "r", "s", "t"], ["p", "q", "r", "s", "t"]]
        truth = ["A", "B", "C", None]
        assert mrr_at_k(ranked, truth) == pytest.approx(0.375)


# ================================================================== chính sách huấn luyện mục 3.5
class TestTrainingPolicy:
    def test_unresolved_dropped_auto_downweighted_rule_feature_masked(self):
        # 4 entry: manual matched, manual no_match, auto matched (arxiv_in_url), auto unresolved
        names = ("title_lev_sim", "arxiv_id_in_entry")
        batch = make_batch(
            [
                [("r1", 0.9, 0.0), ("r2", 0.1, 0.0)],
                [("r1", 0.2, 0.0), ("r2", 0.3, 0.0)],
                [("r1", 0.4, 1.0), ("r2", 0.2, 0.0)],
                [("r1", 0.5, 0.0), ("r2", 0.6, 0.0)],
            ],
            names=names,
            y=[1, 0, 0, 0, 1, 0, 0, 0],
            qids=[("p1", "k0"), ("p1", "k1"), ("p2", "k2"), ("p2", "k3")],
        )
        status = np.array(["matched", "no_match_confirmed", "matched", "unresolved"])
        source = np.array(["manual", "manual", "auto", "auto"])
        rule = np.array([None, None, "arxiv_in_url", None], dtype=object)

        keep = (status != "unresolved") & (status != "no_match_confirmed")  # cấu hình A/B
        w_auto = 0.3
        train = (
            batch.select_queries(keep)
            .with_query_weights(np.where(source[keep] == "auto", w_auto, 1.0))
        )
        train = train.with_features_masked(rule[keep] == "arxiv_in_url", ["arxiv_id_in_entry"])

        m = RecordingModel().fit(train)
        X, y, groups, w = m.seen
        assert groups.tolist() == [2, 2]
        assert w.tolist() == [1.0, 1.0, 0.3, 0.3]
        assert np.isnan(X[2:, 1]).all()  # hàng auto khớp bởi arxiv_in_url bị che F09
        assert not np.isnan(X[:2, 1]).any()  # hàng manual giữ nguyên
        assert not np.isnan(batch.X).any()  # dữ liệu gốc (dùng khi đánh giá) không bị che
        assert train.query_ids.tolist() == [("p1", "k0"), ("p2", "k2")]

        # Đánh giá vẫn dùng đủ cả bốn entry: unresolved/no-match vẫn nằm trong |Q|
        ranked = m.rank_ids(batch, k=2)
        assert ranked.shape == (4, 2)


# ================================================================== adapter pointwise
def synth_papers(n_papers, seed, m=12, n=15, p_nomatch=0.2):
    """Bài giả: feature 0 ~ độ giống tiêu đề (có NaN), feature 1 ~ |lệch năm|."""
    rng = np.random.default_rng(seed)
    batches, truths = [], []
    for p in range(n_papers):
        ref_ids = [f"{2000 + p}.{j:05d}" for j in range(n)]
        rows, y, qids, truth = [], [], [], []
        for i in range(m):
            t = None if rng.random() < p_nomatch else int(rng.integers(n))
            for j in range(n):
                pos = j == t
                f0 = np.clip(rng.normal(0.8 if pos else 0.35, 0.12), 0, 1)
                if rng.random() < 0.1:
                    f0 = NAN
                f1 = abs(rng.normal(0.4, 0.4)) if pos else rng.uniform(0, 6)
                rows.append((ref_ids[j], f0, f1))
                y.append(float(pos))
            qids.append((f"paper{p}", f"key{i}"))
            truth.append(None if t is None else ref_ids[t])
        batches.append(
            make_batch([rows[k * n:(k + 1) * n] for k in range(m)],
                       names=("title_lev_sim", "year_diff_abs"), y=y, qids=qids)
        )
        truths.append(truth)
    return batches, truths


class TestPointwiseClassifierModel:
    def test_beats_naive_baseline_end_to_end(self):
        from sklearn.linear_model import LogisticRegression

        train_b, _ = synth_papers(8, seed=1)
        test_b, test_t = synth_papers(2, seed=2)
        train = RankingBatch.concat(train_b)
        test = RankingBatch.concat(test_b)
        truth = [t for ts in test_t for t in ts]

        naive = mrr_at_k(FeatureRankBaseline().rank_ids(test, k=5), truth)
        lr = PointwiseClassifierModel(LogisticRegression(class_weight="balanced"))
        lr.fit(train)
        lr_mrr = mrr_at_k(lr.rank_ids(test, k=5), truth)
        assert lr_mrr > naive
        assert lr_mrr > 0.7  # entry no-match (kỳ vọng ~20%) vẫn nằm trong mẫu số, điểm 0
        imp = lr.feature_importance()
        assert set(imp) == {"title_lev_sim", "year_diff_abs"}

    def test_imputer_fit_on_train_only(self):
        from sklearn.linear_model import LogisticRegression

        train = make_batch(
            [[("a", 0.2), ("b", NAN), ("c", 0.6)], [("a", 0.4), ("b", 0.8), ("c", 0.0)]],
            names=("f",), y=[1, 0, 0, 0, 0, 1],
        )
        m = PointwiseClassifierModel(LogisticRegression()).fit(train)
        med = np.nanmedian(train.X, axis=0)
        assert np.allclose(m.imputer_.fill_, med)
        test = make_batch([[("a", 100.0), ("b", NAN), ("c", 200.0)]], names=("f",))
        before = m.imputer_.fill_.copy()
        s = m.score(test)
        assert np.isfinite(s).all()  # NaN lúc dự đoán được điền bằng median của Train
        assert np.allclose(m.imputer_.fill_, before)  # Valid/Test không làm đổi imputer

    def test_all_nan_column_keeps_feature_count(self):
        from sklearn.linear_model import LogisticRegression

        b = make_batch([[("a", 0.9, NAN), ("b", 0.1, NAN)], [("a", 0.2, NAN), ("b", 0.8, NAN)]],
                       names=("f", "g"), y=[1, 0, 0, 1])
        m = PointwiseClassifierModel(LogisticRegression()).fit(b)
        assert m.score(b).shape == (4,)
        assert m.imputer_.fill_.tolist()[1] == 0.0

    def test_one_class_rejected(self):
        from sklearn.linear_model import LogisticRegression

        b = make_batch([[("a", 0.9), ("b", 0.1)]], names=("f",), y=[0, 0])
        with pytest.raises(ValueError, match="dương và"):
            PointwiseClassifierModel(LogisticRegression()).fit(b)

    def test_zero_weight_rows_equal_dropping_them(self):
        from sklearn.linear_model import LogisticRegression

        batches, _ = synth_papers(3, seed=5)
        clean = RankingBatch.concat(batches[:2])
        dirty = RankingBatch.concat(batches)
        w = np.where(np.arange(dirty.n_queries) < clean.n_queries, 1.0, 0.0)
        a = PointwiseClassifierModel(LogisticRegression(max_iter=1000)).fit(clean)
        b = PointwiseClassifierModel(LogisticRegression(max_iter=1000)).fit(
            dirty.with_query_weights(w))
        # trọng số 0 chỉ làm median của imputer đổi nhẹ: so sánh qua điểm trên dữ liệu sạch
        sa, sb = a.score(clean), b.score(clean)
        assert np.corrcoef(sa, sb)[0, 1] > 0.999

    def test_weights_change_the_model(self):
        from sklearn.linear_model import LogisticRegression

        batches, _ = synth_papers(2, seed=6)
        b = RankingBatch.concat(batches)
        w1 = PointwiseClassifierModel(LogisticRegression()).fit(b)
        w2 = PointwiseClassifierModel(LogisticRegression()).fit(
            b.with_query_weights(np.linspace(0.01, 5, b.n_queries)))
        assert not np.allclose(w1.estimator_.coef_, w2.estimator_.coef_)

    def test_estimator_is_cloned_not_mutated(self):
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression()
        batches, _ = synth_papers(1, seed=7)
        PointwiseClassifierModel(est).fit(batches[0])
        assert not hasattr(est, "coef_")

    def test_gradient_boosting_handles_nan_without_imputer(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        train_b, _ = synth_papers(6, seed=8)
        test_b, test_t = synth_papers(2, seed=9)
        m = PointwiseClassifierModel(
            HistGradientBoostingClassifier(max_iter=30, random_state=0), impute=False
        ).fit(RankingBatch.concat(train_b))
        assert m.imputer_ is None
        test = RankingBatch.concat(test_b)
        assert np.isfinite(m.score(test)).all()
        truth = [t for ts in test_t for t in ts]
        assert mrr_at_k(m.rank_ids(test), truth) > 0.6
        assert m.feature_importance() is None or set(m.feature_importance()) == set(
            m.feature_names)

    def test_decision_function_scores(self):
        from sklearn.linear_model import LogisticRegression

        b = RankingBatch.concat(synth_papers(2, seed=10)[0])
        m = PointwiseClassifierModel(LogisticRegression(), score_method="decision_function")
        m.fit(b)
        s = m.score(b)
        assert (s < 0).any() and (s > 0).any()  # logit, không phải xác suất
        with pytest.raises(ValueError):
            PointwiseClassifierModel(LogisticRegression(), score_method="nope")

    def test_save_load_roundtrip(self, tmp_path):
        from sklearn.linear_model import LogisticRegression

        b = RankingBatch.concat(synth_papers(2, seed=11)[0])
        m = PointwiseClassifierModel(LogisticRegression()).fit(b)
        path = tmp_path / "model.pkl"
        m.save(path)
        assert [p.name for p in tmp_path.iterdir()] == ["model.pkl"]  # không để lại file tạm
        loaded = MatchingModel.load(path)
        assert np.allclose(loaded.score(b), m.score(b))
        assert loaded.feature_names == m.feature_names and loaded.is_fitted_
        PointwiseClassifierModel.load(path)
        with pytest.raises(TypeError):
            FeatureRankBaseline.load(path)


class TestPersistence:
    def test_rule_model_roundtrip_and_failed_save_cleans_tmp(self, tmp_path):
        m = SumModel()
        p = tmp_path / "sum.pkl"
        m.save(p)
        assert isinstance(MatchingModel.load(p), SumModel)

        class Unpicklable(SumModel):
            def __reduce__(self):
                raise RuntimeError("không pickle được")

        with pytest.raises(RuntimeError):
            Unpicklable().save(tmp_path / "bad.pkl")
        assert sorted(x.name for x in tmp_path.iterdir()) == ["sum.pkl"]
