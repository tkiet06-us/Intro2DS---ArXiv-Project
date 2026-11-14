"""P4-01 (Owner: Kiệt) – Đánh giá MRR@5, so sánh với Naive baseline, xuất predictions.parquet.

Quy ước đánh giá (đề M3 và ``ml_design.md`` mục 7, 9)
-----------------------------------------------------
* |Q| là **mọi entry đã gán nhãn** của tập đánh giá. Entry không có đáp án
  (``no_match_confirmed``, hoặc ``unresolved`` của bài auto) vẫn nằm trong |Q| và nhận
  0 điểm; đáp án ngoài top-5 cũng 0 điểm. Entry chưa gán nhãn **không** thuộc |Q|.
* Báo cáo MRR@5 chung, rồi riêng bài ``manual`` và ``auto``. Nhóm con rỗng là ``nan``
  (N/A), không chia cho 0. MRR trên bài auto chủ yếu phản ánh mức đồng thuận với luật gán
  nhãn, nên kết luận chính dựa trên bài manual.
* Mô hình ML chỉ có giá trị nếu thắng Naive baseline (:func:`compare_with_baseline`).
* Không dùng Test để chọn mô hình hay ngưỡng; file này chỉ *đo*, không chọn.

Dữ liệu vào là danh sách ID dự đoán (đầu ra ``MatchingModel.rank_ids``) và ID đúng (``None`` khi
không có đáp án), nên không phụ thuộc cách dựng đặc trưng hay loại mô hình.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "K", "PARTITIONS", "SOURCES", "PREDICTIONS_SCHEMA", "PredictionRow",
    "BaselineNotBeatenError", "reciprocal_ranks", "mrr_at_k", "mrr", "mrr_report",
    "evaluate_ranking", "compare_with_baseline", "format_results_table",
    "predictions_table", "validate_predictions", "write_predictions", "read_predictions",
    "is_valid_arxiv_id",
]

K = 5
PARTITIONS: tuple[str, ...] = ("train", "valid", "test")
SOURCES: tuple[str, ...] = ("manual", "auto")

# ID chuẩn: dạng mới 2310.12345 hoặc cũ hep-th/9901001 (không hậu tố version).
_NEW_ID = re.compile(r"^\d{4}\.\d{4,5}$")
_OLD_ID = re.compile(r"^[a-z][a-z\-]*(\.[A-Za-z]{2})?/\d{7}$")


def is_valid_arxiv_id(value: object) -> bool:
    """True nếu ``value`` là arXiv ID chuẩn (không có tiền tố ``arXiv:`` hay hậu tố ``vN``)."""
    return isinstance(value, str) and bool(_NEW_ID.match(value) or _OLD_ID.match(value))


# =============================================================================
# MRR@k
# =============================================================================
def reciprocal_ranks(
    predicted: Sequence[Sequence[str | None]], truth: Sequence[str | None], k: int = K
) -> np.ndarray:
    """Reciprocal rank của từng entry: ``1/hạng`` trong top-k, ngược lại 0.

    ``truth[i] is None`` (không có đáp án) luôn cho 0, kể cả khi ``predicted[i]`` có ô ``None``
    (hai ``None`` không được coi là khớp nhau).
    """
    if len(predicted) != len(truth):
        raise ValueError(f"predicted ({len(predicted)}) và truth ({len(truth)}) khác độ dài")
    if k < 1:
        raise ValueError("k phải >= 1")
    out = np.zeros(len(truth), dtype=np.float64)
    for i, (cands, gold) in enumerate(zip(predicted, truth)):
        if gold is None:
            continue
        for rank, cand in enumerate(list(cands)[:k], start=1):
            if cand is not None and str(cand) == str(gold):
                out[i] = 1.0 / rank
                break
    return out


def mrr_at_k(
    predicted: Sequence[Sequence[str | None]],
    truth: Sequence[str | None],
    k: int = K,
    select: Sequence[bool] | None = None,
) -> float:
    """MRR@k trên các entry được chọn; ``nan`` (N/A) nếu không có entry nào."""
    rr = reciprocal_ranks(predicted, truth, k)
    if select is not None:
        sel = np.asarray(select, dtype=bool)
        if sel.shape != rr.shape:
            raise ValueError("select phải có một phần tử cho mỗi entry")
        rr = rr[sel]
    return float(rr.mean()) if rr.size else float("nan")


def mrr(ranked: Sequence[Sequence[str | None]], truth: Sequence[str | None], k: int = K) -> float:
    """Tương thích bản cũ: ``mrr(ranked, truth)``. Entry ``None`` vẫn nằm trong mẫu số."""
    return mrr_at_k(ranked, truth, k)


def _summary(rr: np.ndarray, answerable: np.ndarray, k: int) -> dict[str, Any]:
    n = int(rr.size)
    n_ans = int(answerable.sum())
    nan = float("nan")
    return {
        "mrr": float(rr.mean()) if n else nan,
        "n": n,
        "n_answerable": n_ans,
        # MRR tối đa đạt được: entry không có đáp án luôn 0 điểm (trần của bài toán).
        "mrr_ceiling": n_ans / n if n else nan,
        # Chỉ trên entry có đáp án: chất lượng xếp hạng không bị pha loãng bởi no-match.
        "mrr_answerable": float(rr[answerable].mean()) if n_ans else nan,
        f"hit@{k}": float((rr[answerable] > 0).mean()) if n_ans else nan,
        "hit@1": float((rr[answerable] == 1.0).mean()) if n_ans else nan,
    }


def mrr_report(
    predicted: Sequence[Sequence[str | None]],
    truth: Sequence[str | None],
    *,
    sources: Sequence[str] | None = None,
    arxiv_ids: Sequence[str] | None = None,
    k: int = K,
) -> dict[str, Any]:
    """Báo cáo MRR@k: chung, theo nguồn nhãn (manual/auto) và theo từng bài.

    Args:
        predicted: Top-k ID dự đoán của từng entry đã gán nhãn.
        truth: ID đúng của từng entry, ``None`` nếu không có đáp án.
        sources: ``"manual"``/``"auto"`` cho từng entry (bỏ trống thì chỉ có nhóm ``overall``).
        arxiv_ids: Bài chứa từng entry (bỏ trống thì không có ``by_publication``).

    Returns:
        ``{"k", "overall", "manual", "auto", "by_publication"}``; mỗi nhóm là dict gồm ``mrr``,
        ``n`` (=|Q|), ``n_answerable``, ``mrr_ceiling``, ``mrr_answerable``, ``hit@k``, ``hit@1``.
    """
    rr = reciprocal_ranks(predicted, truth, k)
    answerable = np.array([t is not None for t in truth], dtype=bool)
    report: dict[str, Any] = {"k": k, "overall": _summary(rr, answerable, k)}

    if sources is not None:
        src = np.asarray(list(sources))
        if src.shape != rr.shape:
            raise ValueError("sources phải có một phần tử cho mỗi entry")
        unknown = set(src.tolist()) - set(SOURCES)
        if unknown:
            raise ValueError(f"sources chỉ nhận {SOURCES}, nhận được {sorted(unknown)}")
        for name in SOURCES:
            sel = src == name
            report[name] = _summary(rr[sel], answerable[sel], k)

    if arxiv_ids is not None:
        ids = np.asarray(list(arxiv_ids), dtype=object)
        if ids.shape != rr.shape:
            raise ValueError("arxiv_ids phải có một phần tử cho mỗi entry")
        by_pub: dict[str, Any] = {}
        for pid in dict.fromkeys(ids.tolist()):
            sel = ids == pid
            row = _summary(rr[sel], answerable[sel], k)
            if sources is not None:
                row["source"] = str(np.asarray(list(sources))[sel][0])
            by_pub[str(pid)] = row
        report["by_publication"] = by_pub
    return report


def evaluate_ranking(
    model: Any,
    batch: Any,
    truth: Sequence[str | None],
    *,
    sources: Sequence[str] | None = None,
    arxiv_ids: Sequence[str] | None = None,
    k: int = K,
) -> dict[str, Any]:
    """Xếp hạng ``batch`` bằng ``model.rank_ids`` rồi tính :func:`mrr_report`.

    ``batch`` là ``RankingBatch`` có ``candidate_ids`` (mỗi query là một entry đã gán nhãn,
    pool là toàn bộ reference của bài; **không** lấy mẫu). ``truth`` cùng thứ tự query.
    """
    if batch.n_queries != len(truth):
        raise ValueError(f"batch có {batch.n_queries} query nhưng truth có {len(truth)}")
    predicted = [list(row) for row in model.rank_ids(batch, k)]
    return mrr_report(predicted, truth, sources=sources, arxiv_ids=arxiv_ids, k=k)


# =============================================================================
# So sánh với Naive baseline
# =============================================================================
class BaselineNotBeatenError(RuntimeError):
    """Mô hình không thắng Naive baseline trên nhóm chính."""


def compare_with_baseline(
    reports: Mapping[str, Mapping[str, Any]],
    baseline: str = "baseline",
    *,
    primary: str = "manual",
    margin: float = 0.0,
    raise_if_not: bool = False,
) -> dict[str, dict[str, Any]]:
    """Mỗi mô hình có thắng Naive baseline không? (``ml_design.md`` mục 2.2)

    Mô hình ML chỉ có giá trị nếu MRR@k của nó **lớn hơn** baseline (hơn ít nhất ``margin``)
    trên nhóm ``primary`` (mặc định bài manual). Chỉ gọi trên báo cáo Valid hoặc
    cross-validation khi *chọn* mô hình; báo cáo Test chỉ để công bố kết quả.

    Args:
        reports: ``{tên mô hình: mrr_report}``; phải có khóa ``baseline``.

    Returns:
        ``{tên mô hình: {"mrr", "baseline_mrr", "delta", "beats_baseline", "verdict"}}``.
        ``beats_baseline`` là ``None`` khi nhóm ``primary`` là N/A (chưa kết luận được).

    Raises:
        BaselineNotBeatenError: Nếu ``raise_if_not`` và có mô hình ML không thắng.
    """
    if baseline not in reports:
        raise KeyError(f"thiếu báo cáo của baseline '{baseline}'")
    base = reports[baseline].get(primary, {}).get("mrr", float("nan"))
    out: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for name, rep in reports.items():
        if name == baseline:
            continue
        cur = rep.get(primary, {}).get("mrr", float("nan"))
        row: dict[str, Any] = {"mrr": cur, "baseline_mrr": base}
        if math.isnan(cur) or math.isnan(base):
            row.update(delta=float("nan"), beats_baseline=None,
                       verdict=f"{name}: chưa kết luận (nhóm '{primary}' là N/A)")
        else:
            beats = bool(cur > base + margin)
            row.update(
                delta=cur - base, beats_baseline=beats,
                verdict=(f"{name}: {'THẮNG' if beats else 'KHÔNG thắng'} Naive baseline trên "
                         f"'{primary}' (MRR {cur:.4f} so với {base:.4f}, chênh {cur - base:+.4f})"),
            )
            if not beats:
                failed.append(row["verdict"])
        out[name] = row
    if raise_if_not and failed:
        raise BaselineNotBeatenError("; ".join(failed))
    return out


def _fmt(x: float) -> str:
    return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def format_results_table(
    reports: Mapping[str, Mapping[str, Any]],
    groups: Sequence[str] = ("overall", "manual", "auto"),
    metric: str = "mrr",
) -> str:
    """Bảng markdown: mỗi hàng một mô hình, mỗi cột một nhóm (kèm |Q|)."""
    first = next(iter(reports.values()), {})
    groups = [g for g in groups if g in first]
    head = "| Mô hình | " + " | ".join(f"{g} (|Q|={first[g]['n']})" for g in groups) + " |"
    sep = "|---|" + "---|" * len(groups)
    rows = [f"| {name} | " + " | ".join(_fmt(rep[g][metric]) for g in groups) + " |"
            for name, rep in reports.items()]
    return "\n".join([head, sep, *rows])


# =============================================================================
# predictions.parquet (9 cột, ml_design.md mục 6)
# =============================================================================
PREDICTIONS_SCHEMA = pa.schema(
    [
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("partition", pa.dictionary(pa.int8(), pa.string()), nullable=False),
        pa.field("citation_key", pa.string(), nullable=False),
        pa.field("groundtruth_arxiv_id", pa.string(), nullable=True),
    ]
    + [pa.field(f"cand_{k}", pa.string(), nullable=True) for k in range(1, K + 1)]
)


@dataclass(frozen=True)
class PredictionRow:
    """Một dòng của bảng predictions (một BibTeX entry)."""

    arxiv_id: str
    partition: str
    citation_key: str
    groundtruth_arxiv_id: str | None
    candidates: tuple[str | None, ...]  # tối đa K phần tử; thiếu thì đệm None


def predictions_table(rows: Sequence[PredictionRow]) -> pa.Table:
    """Dựng bảng Arrow đúng schema 9 cột (partition là dictionary int8)."""
    for r in rows:
        if r.partition not in PARTITIONS:
            raise ValueError(f"partition '{r.partition}' không hợp lệ ({PARTITIONS})")
        if len(r.candidates) > K:
            raise ValueError(f"{r.arxiv_id}/{r.citation_key}: {len(r.candidates)} ứng viên > {K}")

    def col(values: Sequence[str | None]) -> pa.Array:
        return pa.array([None if v is None else str(v) for v in values], type=pa.string())

    part_idx = pa.array([PARTITIONS.index(r.partition) for r in rows], type=pa.int8())
    arrays: list[pa.Array] = [
        col([r.arxiv_id for r in rows]),
        pa.DictionaryArray.from_arrays(part_idx, pa.array(PARTITIONS, type=pa.string())),
        col([r.citation_key for r in rows]),
        col([r.groundtruth_arxiv_id for r in rows]),
    ]
    for k in range(K):
        arrays.append(col([r.candidates[k] if k < len(r.candidates) else None for r in rows]))
    return pa.Table.from_arrays(arrays, schema=PREDICTIONS_SCHEMA)


def validate_predictions(
    table: pa.Table,
    pools: Mapping[str, Sequence[str]],
    split: Mapping[str, str] | None = None,
) -> list[str]:
    """Kiểm tra ràng buộc nội dung 6.1 của ``ml_design.md``; trả về danh sách lỗi (rỗng = đạt).

    Args:
        pools: ``{arxiv_id: các ref_arxiv_id của bài}`` (đầy đủ, không lấy mẫu).
        split: ``{arxiv_id: partition}`` từ ``split.json``; có thì kiểm tra partition khớp.
    """
    errors: list[str] = []
    if table.schema.names != PREDICTIONS_SCHEMA.names:
        return [f"tên cột {table.schema.names} khác {PREDICTIONS_SCHEMA.names}"]
    for f_actual, f_expected in zip(table.schema, PREDICTIONS_SCHEMA):
        if f_actual.type != f_expected.type:
            errors.append(f"cột {f_actual.name}: kiểu {f_actual.type}, cần {f_expected.type}")
        if f_actual.nullable != f_expected.nullable:
            errors.append(f"cột {f_actual.name}: nullable={f_actual.nullable}, "
                          f"cần {f_expected.nullable} (6.1.6)")

    data = table.to_pydict()
    for name in ("arxiv_id", "partition", "citation_key"):
        if any(v is None for v in data[name]):
            errors.append(f"cột {name} có giá trị null (6.1.6)")

    seen: set[tuple[str, str]] = set()
    paper_part: dict[str, set[str]] = {}
    pool_sets = {k: set(v) for k, v in pools.items()}
    for i in range(table.num_rows):
        aid, key = data["arxiv_id"][i], data["citation_key"][i]
        where = f"({aid}, {key})"
        if (aid, key) in seen:
            errors.append(f"{where}: trùng khóa (6.1.1)")
        seen.add((aid, key))
        paper_part.setdefault(aid, set()).add(data["partition"][i])

        pool = pool_sets.get(aid)
        if pool is None:
            errors.append(f"{where}: không có pool reference của bài")
            continue
        gt = data["groundtruth_arxiv_id"][i]
        if gt is not None and (not is_valid_arxiv_id(gt) or gt not in pool):
            errors.append(f"{where}: groundtruth '{gt}' không hợp lệ hoặc ngoài pool (6.1.2)")

        cands = [data[f"cand_{k}"][i] for k in range(1, K + 1)]
        filled = [c for c in cands if c is not None]
        for c in filled:
            if not is_valid_arxiv_id(c) or c not in pool:
                errors.append(f"{where}: ứng viên '{c}' không hợp lệ hoặc ngoài pool (6.1.2)")
        if len(set(filled)) != len(filled):
            errors.append(f"{where}: ứng viên trùng nhau (6.1.3)")
        if any(c is None for c in cands[: len(filled)]) or cands[: len(filled)] != filled:
            errors.append(f"{where}: có khoảng trống giữa các cand_k (6.1.3)")
        if len(filled) != min(K, len(pool)):
            errors.append(f"{where}: {len(filled)} ứng viên, cần {min(K, len(pool))} "
                          f"(pool {len(pool)}; 6.1.4)")

    for aid, parts in paper_part.items():
        if len(parts) > 1:
            errors.append(f"bài {aid}: nhiều partition {sorted(parts)} (6.1.5)")
        elif split is not None and aid in split and split[aid] not in parts:
            errors.append(f"bài {aid}: partition {sorted(parts)} khác split.json '{split[aid]}' (6.1.5)")
    return errors


def write_predictions(
    path: str | Path,
    rows: Sequence[PredictionRow],
    pools: Mapping[str, Sequence[str]],
    split: Mapping[str, str] | None = None,
) -> pa.Table:
    """Kiểm tra rồi ghi ``predictions.parquet``; đọc lại để xác nhận schema và nullability.

    Raises:
        ValueError: Nếu vi phạm ràng buộc 6.1 (không ghi file) hoặc schema đọc lại lệch.
    """
    table = predictions_table(rows)
    errors = validate_predictions(table, pools, split)
    if errors:
        shown = "\n  - ".join(errors[:20])
        raise ValueError(f"predictions vi phạm {len(errors)} ràng buộc:\n  - {shown}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    back = pq.read_table(path)
    errors = validate_predictions(back, pools, split)
    if errors:
        raise ValueError("file đọc lại không khớp schema: " + "; ".join(errors[:5]))
    return table


def read_predictions(path: str | Path) -> pa.Table:
    return pq.read_table(path)