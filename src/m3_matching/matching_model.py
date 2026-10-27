"""Interface chung cho các mô hình xếp hạng reference (Milestone 3), phiên bản 2.

File này định nghĩa hợp đồng (contract) mà mọi mô hình phải tuân theo, kèm hai mô hình
tham chiếu nhỏ: ``FeatureRankBaseline`` (Naive baseline theo F01, không huấn luyện) và
``PointwiseClassifierModel`` (bọc LogisticRegression/GBM theo kiểu pointwise). Huấn
luyện và ablation nằm ở ``train.py``; MRR@k, xuất ``predictions.parquet`` nằm ở
``evaluate.py``; dựng đặc trưng nằm ở ``features.py``.

Quy ước dữ liệu
---------------
* Một query là một BibTeX entry; ứng viên là các reference của *cùng bài báo*.
  Mỗi query có thể có 0 ứng viên (pool rỗng).
* ``RankingBatch.X``: ``(n_pairs, n_features)``; các cặp của cùng query nằm liền nhau,
  ``groups[q]`` là số ứng viên của query ``q``.
* Nhãn ``y``: 0/1 theo cặp. Hàng toàn 0 nghĩa là entry không có reference đúng.
  Hàng ``unresolved`` của bài auto không được đưa vào batch huấn luyện (mục 3.4).
* Thứ tự cột feature cố định; lưu ở ``feature_names``.

Quy ước đầu ra
--------------
* ``rank`` luôn trả đúng ``k`` cột. Ô trống là ``PAD = -1`` và chỉ xuất hiện ở cuối hàng.
* Chỉ số trả về là chỉ số *trong danh sách ứng viên của chính query đó* (dạng batch) hoặc
  *chỉ số cột của tensor dense* (dạng dense).
* ``rank_ids`` trả mảng ID (``None`` cho ô trống), sẵn sàng ghi ``cand_1..cand_5``.
"""
from __future__ import annotations

import copy
import os
import pickle
import tempfile
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "PAD",
    "NotFittedError",
    "RankingBatch",
    "DenseView",
    "rank_groups",
    "indices_to_ids",
    "MatchingModel",
    "FeatureRankBaseline",
    "PointwiseClassifierModel",
]

PAD = -1


class NotFittedError(RuntimeError):
    """score/predict_scores/rank được gọi trước fit."""


# --------------------------------------------------------------------------- helpers
def _as_tensor(X: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 2:
        return arr[None, ...], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(
        "X phải có shape (candidates, features) hoặc (queries, candidates, features)"
    )


def _as_mask(mask: ArrayLike | None, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    if mask is None:
        return np.ones(shape, dtype=bool)
    M = np.asarray(mask, dtype=bool)
    if M.ndim == 1 and len(shape) == 2 and shape[0] == 1:
        M = M[None, :]
    if M.shape != shape:
        raise ValueError(f"mask phải có shape {shape}, nhận được {M.shape}")
    return M


def _object_array(values: Sequence[Any], n: int, what: str) -> NDArray[np.object_]:
    """Mảng object 1-D an toàn với phần tử là tuple (np.asarray sẽ làm phẳng tuple)."""
    values = list(values)
    if len(values) != n:
        raise ValueError(f"{what} phải có {n} phần tử, nhận được {len(values)}")
    out = np.empty(n, dtype=object)
    for i, v in enumerate(values):
        out[i] = v
    return out


def _id_tie_ranks(ids: NDArray[np.object_]) -> NDArray[np.int64]:
    """Hạng tăng dần theo chuỗi ID (dùng làm khóa tie-break)."""
    return np.unique(np.asarray(ids).astype(str), return_inverse=True)[1].astype(np.int64)


# --------------------------------------------------------------------------- batch
class DenseView(NamedTuple):
    """Dạng đệm của một RankingBatch (ô đệm của X là NaN, của y là 0)."""

    X: NDArray[np.float64]  # (n_queries, max_group, n_features)
    mask: NDArray[np.bool_]  # (n_queries, max_group), ô thật là True, đệm bên phải
    y: NDArray[np.float64] | None
    sample_weight: NDArray[np.float64] | None


@dataclass(frozen=True, eq=False)
class RankingBatch:
    """Dữ liệu xếp hạng dạng phẳng (ragged); không đệm, không giới hạn kích thước pool.

    Các mảng không được sửa tại chỗ sau khi tạo. Dùng ``select_queries``,
    ``with_query_weights``, ``with_features_masked`` để lấy bản mới.
    """

    X: NDArray[np.float64]
    groups: NDArray[np.intp]
    y: NDArray[np.float64] | None = None
    sample_weight: NDArray[np.float64] | None = None
    candidate_ids: NDArray[np.object_] | None = None
    query_ids: NDArray[np.object_] | None = None
    feature_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X phải có shape (n_pairs, n_features), nhận được {X.shape}")
        if np.isinf(X).any():
            raise ValueError("X chứa ±inf; chỉ cho phép NaN (thiếu dữ liệu)")

        raw_groups = np.asarray(self.groups)
        if raw_groups.ndim != 1 or (raw_groups.size and raw_groups.dtype.kind not in "iu"):
            raise ValueError("groups phải là mảng 1-D số nguyên")
        groups = raw_groups.astype(np.intp)
        if (groups < 0).any():
            raise ValueError("groups không được âm")
        n = X.shape[0]
        if int(groups.sum()) != n:
            raise ValueError(f"tổng groups ({int(groups.sum())}) phải bằng số cặp ({n})")

        y = None
        if self.y is not None:
            y = np.asarray(self.y, dtype=np.float64)
            if y.shape != (n,):
                raise ValueError(f"y phải có shape ({n},), nhận được {y.shape}")
            if not np.isin(y, (0.0, 1.0)).all():
                raise ValueError("y chỉ được nhận giá trị 0 hoặc 1 (không NaN)")

        w = None
        if self.sample_weight is not None:
            w = np.asarray(self.sample_weight, dtype=np.float64)
            if w.shape != (n,):
                raise ValueError(f"sample_weight phải có shape ({n},), nhận được {w.shape}")
            if not np.isfinite(w).all() or (w < 0).any():
                raise ValueError("sample_weight phải hữu hạn và không âm")

        ids = None
        if self.candidate_ids is not None:
            ids = _object_array(self.candidate_ids, n, "candidate_ids")
            if any(v is None for v in ids):
                raise ValueError("candidate_ids không được chứa None")

        qids = None
        if self.query_ids is not None:
            qids = _object_array(self.query_ids, groups.size, "query_ids")

        names = None
        if self.feature_names is not None:
            names = tuple(self.feature_names)
            if len(names) != X.shape[1]:
                raise ValueError("feature_names không khớp số cột feature")
            if len(set(names)) != len(names):
                raise ValueError("feature_names có tên trùng nhau")

        for key, val in (
            ("X", X), ("groups", groups), ("y", y), ("sample_weight", w),
            ("candidate_ids", ids), ("query_ids", qids), ("feature_names", names),
        ):
            object.__setattr__(self, key, val)

    # ------------------------------------------------------------ thuộc tính
    @property
    def n_pairs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_queries(self) -> int:
        return int(self.groups.size)

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def offsets(self) -> NDArray[np.intp]:
        """Vị trí đầu của từng query trong X."""
        return np.cumsum(self.groups) - self.groups

    @property
    def group_index(self) -> NDArray[np.intp]:
        """Query của từng cặp, shape (n_pairs,)."""
        return np.repeat(np.arange(self.n_queries), self.groups)

    # ------------------------------------------------------------ kiểm tra
    def require_labels(self, max_positives: int | None = 1) -> None:
        """Có nhãn, và mỗi query có không quá ``max_positives`` ô dương."""
        if self.y is None:
            raise ValueError("batch không có nhãn y")
        if max_positives is not None and self.n_pairs:
            pos = np.bincount(self.group_index, weights=self.y, minlength=self.n_queries)
            bad = np.flatnonzero(pos > max_positives)
            if bad.size:
                raise ValueError(
                    f"{bad.size} query có hơn {max_positives} ô dương "
                    f"(ví dụ query #{int(bad[0])}); mỗi entry chỉ có tối đa một đáp án"
                )

    # ------------------------------------------------------------ biến đổi
    def select_queries(self, keep: ArrayLike) -> "RankingBatch":
        """Giữ các query có ``keep`` True (mảng bool độ dài n_queries).

        Dùng để bỏ hàng ``unresolved`` của bài auto hoặc hàng ``no_match_confirmed``.
        """
        keep = np.asarray(keep)
        if keep.dtype != bool or keep.shape != (self.n_queries,):
            raise ValueError(f"keep phải là mảng bool shape ({self.n_queries},)")
        pk = np.repeat(keep, self.groups)
        return replace(
            self,
            X=self.X[pk],
            groups=self.groups[keep],
            y=None if self.y is None else self.y[pk],
            sample_weight=None if self.sample_weight is None else self.sample_weight[pk],
            candidate_ids=None if self.candidate_ids is None else self.candidate_ids[pk],
            query_ids=None if self.query_ids is None else self.query_ids[keep],
        )

    def with_query_weights(self, weights: ArrayLike) -> "RankingBatch":
        """Gán trọng số theo query (ví dụ ``w_auto`` cho hàng auto); thay trọng số cũ."""
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (self.n_queries,):
            raise ValueError(f"weights phải có shape ({self.n_queries},)")
        return replace(self, sample_weight=np.repeat(w, self.groups))

    def with_features_masked(
        self, queries: ArrayLike, columns: Sequence[int | str]
    ) -> "RankingBatch":
        """Đặt NaN cho ``columns`` ở các query được chọn (che feature trùng luật, mục 3.5).

        Chỉ nên áp dụng cho batch huấn luyện, không áp dụng khi đánh giá.
        """
        q = np.asarray(queries)
        if q.dtype != bool or q.shape != (self.n_queries,):
            raise ValueError(f"queries phải là mảng bool shape ({self.n_queries},)")
        cols = []
        for c in columns:
            if isinstance(c, str):
                if self.feature_names is None or c not in self.feature_names:
                    raise KeyError(f"không tìm thấy feature {c!r}")
                cols.append(self.feature_names.index(c))
            else:
                if not 0 <= int(c) < self.n_features:
                    raise IndexError(f"cột {c} ngoài phạm vi")
                cols.append(int(c))
        X = self.X.copy()
        rows = np.flatnonzero(np.repeat(q, self.groups))
        if rows.size and cols:
            X[np.ix_(rows, cols)] = np.nan
        return replace(self, X=X)

    @classmethod
    def concat(cls, batches: Sequence["RankingBatch"]) -> "RankingBatch":
        """Ghép nhiều batch (mỗi bài một batch). Các trường tùy chọn phải đồng nhất."""
        batches = list(batches)
        if not batches:
            raise ValueError("concat cần ít nhất một batch")
        if len({b.feature_names for b in batches}) > 1:
            raise ValueError("các batch có feature_names khác nhau")
        if len({b.n_features for b in batches}) > 1:
            raise ValueError("các batch có số feature khác nhau")
        for attr in ("y", "sample_weight", "candidate_ids", "query_ids"):
            if len({getattr(b, attr) is None for b in batches}) > 1:
                raise ValueError(f"trường {attr} có ở một số batch nhưng thiếu ở batch khác")

        def cat(attr: str):
            if getattr(batches[0], attr) is None:
                return None
            return np.concatenate([getattr(b, attr) for b in batches])

        return cls(
            X=np.concatenate([b.X for b in batches]),
            groups=np.concatenate([b.groups for b in batches]),
            y=cat("y"),
            sample_weight=cat("sample_weight"),
            candidate_ids=cat("candidate_ids"),
            query_ids=cat("query_ids"),
            feature_names=batches[0].feature_names,
        )

    # ------------------------------------------------------------ dense <-> ragged
    @classmethod
    def from_dense(
        cls,
        X: ArrayLike,
        mask: ArrayLike | None = None,
        *,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        candidate_ids: ArrayLike | None = None,
        query_ids: Sequence[Any] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "RankingBatch":
        """Dựng batch từ tensor đệm. Ô ``mask`` False bị bỏ.

        ``sample_weight`` nhận ``(n_queries, n_candidates)`` hoặc ``(n_queries,)``.
        ``candidate_ids`` nhận ``(n_queries, n_candidates)`` hoặc ``(n_candidates,)``
        (pool dùng chung).
        """
        X3, squeeze = _as_tensor(X)
        shape = X3.shape[:-1]
        n_q, n_c = shape
        M = _as_mask(mask, shape)

        Y = None
        if y is not None:
            Y = np.asarray(y, dtype=np.float64)
            if Y.ndim == 1 and n_q == 1:
                Y = Y[None, :]
            if Y.shape != shape:
                raise ValueError(f"y phải có shape {shape}, nhận được {Y.shape}")

        W = None
        if sample_weight is not None:
            W = np.asarray(sample_weight, dtype=np.float64)
            if W.ndim == 1:
                if not squeeze and W.size == n_q:
                    W = np.broadcast_to(W[:, None], shape)
                elif W.size == n_c and n_q == 1:
                    W = W[None, :]
            if W.shape != shape:
                raise ValueError(
                    f"sample_weight phải có shape {shape} hoặc ({n_q},), nhận được {W.shape}"
                )

        ids = None
        if candidate_ids is not None:
            raw = np.asarray(candidate_ids, dtype=object)
            if raw.ndim == 1:
                raw = np.broadcast_to(raw, shape)
            if raw.shape != shape:
                raise ValueError(f"candidate_ids phải có shape {shape} hoặc ({n_c},)")
            ids = raw[M]

        return cls(
            X=X3[M],
            groups=M.sum(axis=1),
            y=None if Y is None else Y[M],
            sample_weight=None if W is None else W[M],
            candidate_ids=ids,
            query_ids=query_ids,
            feature_names=None if feature_names is None else tuple(feature_names),
        )

    def to_dense(self) -> DenseView:
        """Đệm về tensor ``(n_queries, max_group, n_features)``; ô đệm bên phải."""
        n_c = int(self.groups.max()) if self.n_queries else 0
        M = np.arange(n_c)[None, :] < self.groups[:, None]
        X3 = np.full((self.n_queries, n_c, self.n_features), np.nan)
        X3[M] = self.X
        Y = W = None
        if self.y is not None:
            Y = np.zeros((self.n_queries, n_c))
            Y[M] = self.y
        if self.sample_weight is not None:
            W = np.zeros((self.n_queries, n_c))
            W[M] = self.sample_weight
        return DenseView(X3, M, Y, W)

    # ------------------------------------------------------------ ID
    def ids_at(self, idx: ArrayLike) -> NDArray[np.object_]:
        """Đổi chỉ số trong-query ``(n_queries, k)`` sang ID; PAD thành None."""
        if self.candidate_ids is None:
            raise ValueError("batch không có candidate_ids")
        idx = np.asarray(idx)
        if idx.ndim != 2 or idx.shape[0] != self.n_queries:
            raise ValueError(f"idx phải có shape ({self.n_queries}, k)")
        valid = idx >= 0
        if (idx >= self.groups[:, None]).any():
            raise IndexError("idx vượt quá số ứng viên của query")
        if self.n_pairs == 0:
            return np.full(idx.shape, None, dtype=object)
        glob = np.where(valid, self.offsets[:, None] + idx, 0)
        out = self.candidate_ids[glob]  # fancy indexing: luôn là bản sao
        out[~valid] = None
        return out


def indices_to_ids(order: ArrayLike, ids: ArrayLike) -> NDArray[np.object_]:
    """Đổi chỉ số xếp hạng (có PAD) sang ID, ô PAD thành ``None``.

    ``order``: ``(k,)`` hoặc ``(n_queries, k)``. ``ids``: ``(n_candidates,)`` dùng chung
    hoặc ``(n_queries, n_candidates)``. Dùng hàm này thay vì tự tra ``ids[order]`` vì
    ``-1`` là chỉ số hợp lệ của Python và sẽ âm thầm trả ứng viên cuối.
    """
    O = np.asarray(order)
    one = O.ndim == 1
    O = O[None, :] if one else O
    if O.ndim != 2:
        raise ValueError("order phải là mảng 1-D hoặc 2-D")
    I = np.asarray(ids, dtype=object)
    if I.ndim == 1:
        I = np.broadcast_to(I, (O.shape[0], I.size))
    if I.ndim != 2 or I.shape[0] != O.shape[0]:
        raise ValueError("ids phải là (n_candidates,) hoặc (n_queries, n_candidates)")
    valid = O >= 0
    if (O >= I.shape[1]).any():
        raise IndexError("order vượt quá số ứng viên")
    if I.shape[1] == 0:
        out = np.full(O.shape, None, dtype=object)
    else:
        out = np.take_along_axis(I, np.where(valid, O, 0), axis=1).astype(object)
        out[~valid] = None
    return out[0] if one else out


# --------------------------------------------------------------------------- xếp hạng
def rank_groups(
    scores: ArrayLike,
    groups: ArrayLike,
    k: int = 5,
    *,
    tie_break: ArrayLike | None = None,
    candidate_ids: ArrayLike | None = None,
    min_score: float | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Top-k theo từng nhóm, vector hóa hoàn toàn.

    Thứ tự: điểm thật giảm dần; điểm NaN xếp sau mọi điểm thật nhưng vẫn được trả về;
    hòa điểm thì khóa ``tie_break`` nhỏ hơn thắng, rồi vị trí đứng trước thắng.
    ``candidate_ids``: nếu có, mỗi ID chỉ giữ lần xuất hiện tốt nhất trong nhóm.
    ``min_score``: loại ứng viên có điểm ``< min_score``; điểm NaN cũng bị loại vì
    không có điểm để so với ngưỡng. Lưu ý với MRR@k việc loại ứng viên không bao giờ
    làm điểm tăng, chỉ dùng khi cần độ chính xác thay vì thứ hạng.

    Returns:
        ``(idx, top)`` shape ``(n_queries, k)``. ``idx`` là chỉ số trong nhóm, ô trống là
        ``PAD`` và chỉ ở cuối hàng; ``top`` là điểm tương ứng (ô trống là -inf).
    """
    if k < 1:
        raise ValueError("k phải lớn hơn hoặc bằng 1")
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.intp)
    n_q, n = groups.size, scores.size
    if scores.ndim != 1 or n != int(groups.sum()):
        raise ValueError("scores phải là mảng 1-D có độ dài bằng tổng groups")

    out_idx = np.full((n_q, k), PAD, dtype=np.intp)
    out_top = np.full((n_q, k), -np.inf)
    if n == 0:
        return out_idx, out_top

    gid = np.repeat(np.arange(n_q), groups)
    offsets = np.cumsum(groups) - groups
    is_nan = np.isnan(scores)
    key = np.where(is_nan, 0.0, scores)
    position = np.arange(n)
    keys: list[Any] = [position]
    if tie_break is not None:
        tb = np.asarray(tie_break)
        if tb.shape != (n,):
            raise ValueError(f"tie_break phải có shape ({n},), nhận được {tb.shape}")
        keys.append(tb)
    keys += [-key, is_nan, gid]  # khóa cuối là khóa chính
    order = np.lexsort(keys)
    g = gid[order]

    if candidate_ids is not None:
        ids = np.asarray(candidate_ids)
        if ids.shape != (n,):
            raise ValueError(f"candidate_ids phải có shape ({n},)")
        codes = np.unique(ids.astype(str), return_inverse=True)[1].astype(np.int64)
        combined = g.astype(np.int64) * (int(codes.max()) + 1) + codes[order]
        _, first = np.unique(combined, return_index=True)  # lần đầu = điểm tốt nhất
        keep = np.zeros(order.size, dtype=bool)
        keep[first] = True
        order, g = order[keep], g[keep]

    sc = scores[order]
    if min_score is not None:
        ok = sc >= min_score  # NaN so sánh False nên bị loại
        order, g, sc = order[ok], g[ok], sc[ok]

    count = np.bincount(g, minlength=n_q)
    start = np.cumsum(count) - count
    pos = np.arange(order.size) - start[g]
    sel = pos < k
    out_idx[g[sel], pos[sel]] = order[sel] - offsets[g[sel]]
    out_top[g[sel], pos[sel]] = sc[sel]
    return out_idx, out_top


class _DenseLayout:
    """Ghi nhớ cách ánh xạ kết quả ragged về tensor dense của người gọi."""

    def __init__(self, mask: NDArray[np.bool_], squeeze: bool) -> None:
        self.mask = mask
        self.squeeze = squeeze
        self.cols = np.nonzero(mask)[1]  # cột dense của từng cặp (thứ tự hàng trước)

    def scatter(self, values: NDArray[np.float64], fill: float) -> NDArray[np.float64]:
        out = np.full(self.mask.shape, fill)
        out[self.mask] = values
        return out[0] if self.squeeze else out

    def to_columns(self, idx: NDArray[np.intp], offsets: NDArray[np.intp]) -> NDArray[np.intp]:
        if self.cols.size == 0:
            return idx
        glob = np.where(idx >= 0, offsets[:, None] + idx, 0)
        return np.where(idx >= 0, self.cols[glob], PAD)


# --------------------------------------------------------------------------- hợp đồng
class MatchingModel(ABC):
    """Contract cho model chấm điểm và xếp hạng ứng viên.

    Subclass cài ``_score`` (bắt buộc) và ``_fit`` (nếu có huấn luyện). Lớp cha lo
    đổi dạng dense/ragged, kiểm tra nhãn/shape/feature, NaN, khử trùng, tie-break, đệm.

    Có hai cách gọi, cùng tên phương thức:

    * ``RankingBatch`` (khuyến nghị): ``model.fit(batch)``, ``model.rank(batch)``.
    * Dense như bản 1: ``model.fit(X, y, mask=...)``, ``model.rank(X, mask=...)``.
    """

    #: False cho mô hình luật không cần huấn luyện.
    requires_fit: bool = True
    #: Số ô dương tối đa mỗi query khi fit (đề: tối đa một đáp án). None = không kiểm tra.
    max_positives_per_query: int | None = 1
    #: Tên các cột feature theo đúng thứ tự (feature importance, poster).
    feature_names: tuple[str, ...] | None = None
    n_features_in_: int | None = None
    is_fitted_: bool = False

    # ------------------------------------------------------------------ API
    def fit(
        self,
        X: RankingBatch | ArrayLike,
        y: ArrayLike | None = None,
        *,
        mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "MatchingModel":
        """Huấn luyện từ ``RankingBatch`` hoặc từ ``(X, y, mask)`` dạng dense.

        Chỉ đưa vào các hàng dùng để học (bỏ hàng unresolved, theo mục 3.5). Hàng không
        có ô thật nào bị bỏ khỏi ``groups`` truyền cho ``_fit``.
        """
        if isinstance(X, RankingBatch):
            if y is not None or mask is not None or sample_weight is not None:
                raise ValueError("với RankingBatch, y/mask/sample_weight nằm trong batch")
            batch = X
            if feature_names is not None:
                batch = replace(batch, feature_names=tuple(feature_names))
        else:
            if y is None:
                raise ValueError("thiếu y")
            batch = RankingBatch.from_dense(
                X, mask, y=y, sample_weight=sample_weight, feature_names=feature_names
            )
        batch.require_labels(self.max_positives_per_query)
        if batch.n_pairs == 0:
            raise ValueError("không có cặp huấn luyện nào (mọi hàng đều bị bỏ?)")

        self.is_fitted_ = False  # fit lỗi giữa chừng không để lại mô hình "đã fit"
        if batch.feature_names is not None:
            self.feature_names = batch.feature_names
        elif self.feature_names is not None and len(self.feature_names) != batch.n_features:
            raise ValueError("feature_names đã lưu không khớp số cột feature")
        self.n_features_in_ = batch.n_features
        self._fit(batch.X, batch.y, batch.groups[batch.groups > 0], batch.sample_weight)
        self.is_fitted_ = True
        return self

    def score(self, batch: RankingBatch) -> NDArray[np.float64]:
        """Điểm từng cặp, shape ``(n_pairs,)``. NaN nghĩa là mô hình không chấm được."""
        if not isinstance(batch, RankingBatch):
            raise TypeError("score nhận RankingBatch; dùng predict_scores cho tensor dense")
        self._check_fitted()
        self._check_features(batch)
        if batch.n_pairs == 0:
            return np.empty(0, dtype=np.float64)
        s = np.asarray(self._score(batch.X), dtype=np.float64)
        if s.shape != (batch.n_pairs,):
            raise ValueError(f"_score() phải trả shape ({batch.n_pairs},), nhận được {s.shape}")
        if np.isinf(s).any():
            raise ValueError("_score() không được trả ±inf (dành cho quy ước đệm); dùng NaN")
        return s

    def predict_scores(
        self,
        X: RankingBatch | ArrayLike,
        mask: ArrayLike | None = None,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> NDArray[np.float64]:
        """Dense: shape ``X.shape[:-1]``, ô đệm là -inf, ô NaN giữ NaN.
        Với ``RankingBatch``: tương đương ``score``."""
        batch, layout = self._prepare(X, mask, feature_names)
        s = self.score(batch)
        return s if layout is None else layout.scatter(s, -np.inf)

    def rank(
        self,
        X: RankingBatch | ArrayLike,
        k: int = 5,
        *,
        mask: ArrayLike | None = None,
        feature_names: Sequence[str] | None = None,
        tie_break: ArrayLike | str = "auto",
        min_score: float | None = None,
        return_scores: bool = False,
    ):
        """Top-k chỉ số ứng viên theo điểm giảm dần.

        Args:
            tie_break: ``"auto"`` (mặc định) = theo ID tăng dần nếu có ``candidate_ids``,
                nếu không thì theo vị trí; ``"id"``; ``"position"``; hoặc mảng khóa
                (nhỏ hơn thắng) cùng shape với điểm. Hòa cuối cùng: vị trí đứng trước thắng.
            min_score: ngưỡng loại ứng viên (xem ``rank_groups``); chọn trên Valid.
            return_scores: trả thêm mảng điểm cùng shape (ô trống là -inf).

        Returns:
            Dạng batch: ``(n_queries, k)`` chỉ số trong danh sách ứng viên của query.
            Dạng dense: ``(k,)`` hoặc ``(n_queries, k)`` chỉ số cột của tensor.
            Ô trống là ``PAD`` (chỉ ở cuối). Batch có ``candidate_ids`` được khử trùng ID.
        """
        if k < 1:
            raise ValueError("k phải lớn hơn hoặc bằng 1")
        batch, layout = self._prepare(X, mask, feature_names)
        scores = self.score(batch)
        tb = self._resolve_tie_break(tie_break, batch, layout)
        idx, top = rank_groups(
            scores, batch.groups, k, tie_break=tb,
            candidate_ids=batch.candidate_ids, min_score=min_score,
        )
        if layout is not None:
            idx = layout.to_columns(idx, batch.offsets)
            if layout.squeeze:
                idx, top = idx[0], top[0]
        return (idx, top) if return_scores else idx

    def rank_ids(
        self,
        batch: RankingBatch,
        k: int = 5,
        *,
        tie_break: ArrayLike | str = "auto",
        min_score: float | None = None,
        return_scores: bool = False,
    ):
        """Top-k ID ứng viên ``(n_queries, k)``: sẵn sàng ghi ``cand_1..cand_k``.

        Ô trống là ``None`` (null trong Parquet), không trùng ID, không có khoảng trống
        (ràng buộc 6.1). Pool rỗng cho hàng toàn ``None``.
        """
        if not isinstance(batch, RankingBatch) or batch.candidate_ids is None:
            raise ValueError("rank_ids cần RankingBatch có candidate_ids")
        idx, top = self.rank(
            batch, k, tie_break=tie_break, min_score=min_score, return_scores=True
        )
        ids = batch.ids_at(idx)
        return (ids, top) if return_scores else ids

    def feature_importance(self) -> dict[str, float] | None:
        """Độ quan trọng theo tên feature nếu mô hình hỗ trợ; ngược lại None."""
        return None

    # ------------------------------------------------------------ hooks
    def _fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        groups: NDArray[np.intp],
        sample_weight: NDArray[np.float64] | None,
    ) -> None:
        """Mặc định không làm gì (mô hình luật).

        ``X``: ``(n_pairs, n_features)`` có thể chứa NaN; ``y``: 0/1; ``groups``: số ứng
        viên (> 0) của từng query theo thứ tự trong ``X`` (đúng định dạng ``group`` của
        LightGBM/XGBoost ranker); ``sample_weight``: theo cặp hoặc None.
        """

    @abstractmethod
    def _score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Điểm cho ma trận ``(n_pairs, n_features)``; trả về ``(n_pairs,)``.

        Trả NaN cho cặp không chấm được (sẽ xếp sau cùng); không trả ±inf.
        """

    # --------------------------------------------------------- tiện ích
    def save(self, path: str | Path) -> None:
        """Ghi pickle nguyên tử (ghi file tạm rồi đổi tên)."""
        path = Path(path)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: str | Path) -> "MatchingModel":
        """Nạp mô hình. Pickle có thể chạy mã tùy ý: chỉ nạp file do chính nhóm tạo."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"file chứa {type(obj).__name__}, không phải {cls.__name__}")
        return obj

    def _check_fitted(self) -> None:
        if self.requires_fit and not self.is_fitted_:
            raise NotFittedError(f"{type(self).__name__} chưa được fit()")

    def _check_features(self, batch: RankingBatch) -> None:
        if self.n_features_in_ is not None and batch.n_features != self.n_features_in_:
            raise ValueError(
                f"mô hình học với {self.n_features_in_} feature, nhận {batch.n_features}"
            )
        if batch.feature_names is None:
            return
        if self.feature_names is None:
            if not self.requires_fit:  # mô hình luật nhận tên từ dữ liệu đầu tiên
                self.feature_names = batch.feature_names
        elif batch.feature_names != self.feature_names:
            diff = [
                f"{i}: {a!r} != {b!r}"
                for i, (a, b) in enumerate(zip(self.feature_names, batch.feature_names))
                if a != b
            ] or ["độ dài khác nhau"]
            raise ValueError("thứ tự/tên feature không khớp lúc fit: " + "; ".join(diff[:5]))

    def _prepare(
        self,
        X: RankingBatch | ArrayLike,
        mask: ArrayLike | None,
        feature_names: Sequence[str] | None,
    ) -> tuple[RankingBatch, _DenseLayout | None]:
        if isinstance(X, RankingBatch):
            if mask is not None:
                raise ValueError("với RankingBatch không dùng mask")
            if feature_names is not None:
                X = replace(X, feature_names=tuple(feature_names))
            return X, None
        X3, squeeze = _as_tensor(X)
        M = _as_mask(mask, X3.shape[:-1])
        batch = RankingBatch.from_dense(X3, M, feature_names=feature_names)
        return batch, _DenseLayout(M, squeeze)

    @staticmethod
    def _resolve_tie_break(
        tie_break: ArrayLike | str, batch: RankingBatch, layout: _DenseLayout | None
    ) -> NDArray | None:
        if isinstance(tie_break, str):
            if tie_break == "position":
                return None
            if tie_break not in ("auto", "id"):
                raise ValueError("tie_break phải là 'auto', 'id', 'position' hoặc mảng khóa")
            if batch.candidate_ids is None:
                if tie_break == "id":
                    raise ValueError("tie_break='id' cần candidate_ids")
                return None
            return _id_tie_ranks(batch.candidate_ids)
        tb = np.asarray(tie_break)
        if layout is not None:  # dense: (n_c,) dùng chung hoặc (n_q, n_c)
            shape = layout.mask.shape
            if tb.ndim == 1 and tb.size == shape[1]:
                tb = np.broadcast_to(tb, shape)
            if tb.shape != shape:
                raise ValueError(f"tie_break phải có shape {shape} hoặc ({shape[1]},)")
            tb = tb[layout.mask]
        return tb


# --------------------------------------------------------------------------- mô hình tham chiếu
class FeatureRankBaseline(MatchingModel):
    """Naive baseline (mục 2.2): xếp ứng viên theo một cột feature, không huấn luyện.

    Mặc định ``title_lev_sim`` (F01) giảm dần. Cặp có feature NaN xếp sau cùng; hòa điểm
    theo ``ref_arxiv_id`` tăng dần khi batch có ``candidate_ids`` (``rank_ids``). Cột
    được tìm theo tên qua ``feature_names`` của batch/mô hình hoặc truyền chỉ số cột.
    """

    requires_fit = False

    def __init__(
        self,
        column: int | str = "title_lev_sim",
        *,
        higher_is_better: bool = True,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.column = column
        self.higher_is_better = higher_is_better
        if feature_names is not None:
            self.feature_names = tuple(feature_names)

    def _column_index(self) -> int:
        if isinstance(self.column, int):
            return self.column
        if self.feature_names is None or self.column not in self.feature_names:
            raise KeyError(
                f"không tìm thấy feature {self.column!r}; truyền feature_names khi tạo "
                "mô hình hoặc dùng batch có feature_names"
            )
        return self.feature_names.index(self.column)

    def _score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        col = self._column_index()
        if not 0 <= col < X.shape[1]:
            raise IndexError(f"cột {col} ngoài phạm vi {X.shape[1]}")
        v = X[:, col]
        return v if self.higher_is_better else -v

    def feature_importance(self) -> dict[str, float] | None:
        if self.feature_names is None:
            return None
        return {n: float(i == self._column_index()) for i, n in enumerate(self.feature_names)}


class _NanMedianImputer:
    """Điền median từng cột; fit chỉ trên dữ liệu được đưa vào ``fit`` (Train)."""

    def fit(self, X: NDArray[np.float64]) -> "_NanMedianImputer":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # cột toàn NaN
            med = np.nanmedian(X, axis=0)
        self.fill_ = np.where(np.isnan(med), 0.0, med)  # giữ nguyên số cột
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = X.copy()
        r, c = np.nonzero(np.isnan(X))
        X[r, c] = self.fill_[c]
        return X


class PointwiseClassifierModel(MatchingModel):
    """Bọc một estimator kiểu scikit-learn thành mô hình pointwise (mục 2.2).

    * ``impute=True`` (Logistic Regression): điền median fit trên dữ liệu ``fit``, dùng
      lại cho Valid/Test; cờ missing đã là feature F12 nên không thêm cột.
      ``impute=False`` cho mô hình chịu được NaN (HistGradientBoosting, LightGBM, XGBoost).
    * Điểm là xác suất lớp dương (``predict_proba``) hoặc ``decision_function``.
    * ``sample_weight`` (w_auto) được chuyển cho ``estimator.fit``.
    * Cần ít nhất một ô dương và một ô âm trong dữ liệu huấn luyện.
    """

    def __init__(self, estimator: Any, *, impute: bool = True, score_method: str = "auto") -> None:
        if score_method not in ("auto", "predict_proba", "decision_function"):
            raise ValueError("score_method không hợp lệ")
        self.estimator = estimator
        self.impute = impute
        self.score_method = score_method

    def _fit(self, X, y, groups, sample_weight) -> None:
        yb = (y > 0.5).astype(int)
        if np.unique(yb).size < 2:
            raise ValueError("cần cả ô dương và ô âm trong dữ liệu huấn luyện")
        try:
            from sklearn.base import clone

            est = clone(self.estimator)
        except ImportError:
            est = copy.deepcopy(self.estimator)
        self.imputer_ = _NanMedianImputer().fit(X) if self.impute else None
        Xt = X if self.imputer_ is None else self.imputer_.transform(X)
        if sample_weight is None:
            est.fit(Xt, yb)
        else:
            est.fit(Xt, yb, sample_weight=sample_weight)
        self.estimator_ = est

    def _score(self, X):
        Xt = X if self.imputer_ is None else self.imputer_.transform(X)
        est = self.estimator_
        method = self.score_method
        if method == "auto":
            method = "predict_proba" if hasattr(est, "predict_proba") else "decision_function"
        if method == "predict_proba":
            col = list(est.classes_).index(1)
            return est.predict_proba(Xt)[:, col]
        return est.decision_function(Xt)

    def feature_importance(self) -> dict[str, float] | None:
        est = getattr(self, "estimator_", None)
        if est is None or self.feature_names is None:
            return None
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
        elif hasattr(est, "coef_"):
            vals = np.abs(np.ravel(est.coef_))  # theo thang gốc của feature, không chuẩn hóa
        else:
            return None
        vals = np.asarray(vals, dtype=np.float64).ravel()
        if vals.size != len(self.feature_names):
            return None
        return {n: float(v) for n, v in zip(self.feature_names, vals)}