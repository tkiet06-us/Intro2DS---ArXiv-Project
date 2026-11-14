"""P4-01 (Owner: Kiệt) – Huấn luyện, ablation chính sách nhãn A–D, so với Naive baseline, xuất predictions.

Luồng chạy (``ml_design.md`` mục 2, 3, 8)::

    bibtex + references + data/labels/  ->  check_labels  ->  ma trận đặc trưng từng bài
        -> với mỗi mô hình x chính sách huấn luyện (A, B, C, D): fit trên Train, đo Valid + LOPO
        -> chọn chính sách theo MRR@5 trên Valid (bài manual ưu tiên), KHÔNG dùng Test
        -> fit lại trên Train với chính sách đã chọn, đo Valid và Test
        -> mô hình ML phải thắng Naive baseline (F01) trên Valid-manual mới được dùng
        -> ghi predictions.parquet (mọi entry của mọi bài Train/Valid/Test) + báo cáo

Chính sách nhãn khi train (mục 3.5; hàng = một BibTeX entry):

=====================================  ===============================  =================
Loại hàng                              Dùng khi train?                  Trọng số
=====================================  ===============================  =================
manual · matched                       có                               1.0
manual · no_match_confirmed            không (chỉ cấu hình C)           1.0
auto · matched                         có, che feature trùng luật       ``w_auto`` (<1)
auto · unresolved                      không (mask, KHÔNG là mẫu âm)    –
=====================================  ===============================  =================

A = chỉ manual matched · B = A + auto matched · C = B + manual no-match làm hàng âm ·
D = đối chứng (A + toàn bộ auto, ``unresolved`` làm mẫu âm), **không bao giờ được chọn**.

Đánh giá: mọi entry đã gán nhãn nằm trong |Q|, kể cả no-match và ``unresolved`` (0 điểm).
Chạy::

    python -m src.m3_matching.train --bibtex data/processed/bibtex.parquet \\
        --references data/processed/references.parquet --labels data/labels --out-dir models \\
        --reports-dir reports --predictions data/predictions/predictions.parquet
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.m3_matching import features as feat
from src.m3_matching.evaluate import (
    K, PARTITIONS, SOURCES, PredictionRow, compare_with_baseline, evaluate_ranking,
    format_results_table, is_valid_arxiv_id, mrr_at_k, mrr_report, write_predictions,
)
from src.m3_matching.matching_model import (
    FeatureRankBaseline, MatchingModel, PointwiseClassifierModel, RankingBatch,
)
from src.schemas import Bibtex, Reference

log = logging.getLogger("m3.train")

MATCHED, NO_MATCH, UNRESOLVED = "matched", "no_match_confirmed", "unresolved"
MANUAL, AUTO = "manual", "auto"

#: Luật gán nhãn tự động -> feature định nghĩa trùng luật, che (NaN) khi train (mục 3.5).
#: Luật theo tiêu đề (``title_exact``, ``title_sim>=0.92``...) KHÔNG che: che F01–F04 sẽ mất gần
#: hết tín hiệu; ở đó dựa vào ``w_auto`` thấp và kiểm chứng trên bài manual.
RULE_MASK_FEATURES: dict[str, list[str]] = {"arxiv_in_url": ["arxiv_id_in_fields"]}

DEFAULT_W_AUTO = (0.1, 0.3, 0.5)
NAIVE_NAME = "baseline"


class LabelError(ValueError):
    """Nhãn hoặc split vi phạm quy tắc check_labels (mục 3.3)."""


# =============================================================================
# Dữ liệu
# =============================================================================
@dataclass(frozen=True)
class LabelRecord:
    citation_key: str
    groundtruth: str | None
    status: str
    rule: str | None = None
    annotator: str | None = None


@dataclass
class Paper:
    """Một bài báo: toàn bộ entry sau dedup, toàn bộ pool reference, nhãn và split."""

    arxiv_id: str
    partition: str
    source: str  # manual | auto (theo split.json)
    entries: list[Bibtex]
    candidates: list[Reference]  # đã khử trùng ID, sắp tăng dần theo ref_arxiv_id
    label_records: list[LabelRecord] = field(default_factory=list)
    file_source: str | None = None  # label_source ghi trong file nhãn, để kiểm tra
    raw_texts: list[str | None] | None = None
    tensor: np.ndarray | None = None  # (m, n, F) thô, còn NaN

    @property
    def candidate_ids(self) -> list[str]:
        return [c.ref_arxiv_id for c in self.candidates]

    @property
    def keys(self) -> list[str]:
        return [e.citation_key for e in self.entries]

    @property
    def labels(self) -> dict[str, LabelRecord]:
        return {r.citation_key: r for r in self.label_records}


# ----------------------------------------------------------------------- đọc bảng
def _none(value: Any) -> Any:
    """NaN/NA của pandas -> None; mảng numpy -> list."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if isinstance(value, np.generic) else value


def _field_names(cls: type) -> set[str]:
    if hasattr(cls, "model_fields"):  # pydantic v2
        return set(cls.model_fields)
    if hasattr(cls, "__fields__"):  # pydantic v1
        return set(cls.__fields__)
    return {f.name for f in dataclasses.fields(cls)}


def _make(cls: type, row: Mapping[str, Any], list_fields: Sequence[str]) -> Any:
    names = _field_names(cls)
    data = {k: _none(v) for k, v in row.items() if k in names}
    for name in list_fields:
        v = data.get(name)
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                v = [v]
        data[name] = [str(x) for x in v] if v else []
    if data.get("year") is not None:
        try:
            data["year"] = int(data["year"])
        except (TypeError, ValueError):
            data["year"] = None
    return cls(**data)


def read_table(path: str | Path):
    """Đọc bảng .parquet (hoặc thư mục shard), .csv, .jsonl thành DataFrame."""
    import pandas as pd

    path = Path(path)
    if path.is_dir() or path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=True)
    raise ValueError(f"không đọc được định dạng {path.suffix}")


def load_labels(labels_dir: str | Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    """Đọc ``split.json`` và mọi file nhãn ``<id>.json`` trong ``labels_dir``.

    Returns:
        ``(split, label_files)``: ``split[arxiv_id] = {"partition", "label_source"}``;
        ``label_files[arxiv_id] = {"label_source", "records": list[LabelRecord]}``.
        Tên file không được dùng; bài xác định bằng trường ``arxiv_id`` trong file.
    """
    labels_dir = Path(labels_dir)
    split_path = labels_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"thiếu {split_path} (nguồn sự thật duy nhất cho việc chia tập)")
    split = json.loads(split_path.read_text(encoding="utf-8"))

    files: dict[str, dict[str, Any]] = {}
    for p in sorted(labels_dir.glob("*.json")):
        if p.name == "split.json":
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        aid = doc["arxiv_id"]
        source = doc.get("label_source")
        records: list[LabelRecord] = []
        for r in doc.get("labels", []):
            gt = r.get("groundtruth_arxiv_id")
            status = r.get("label_status") or (
                MATCHED if gt is not None else (NO_MATCH if source == MANUAL else UNRESOLVED)
            )
            records.append(LabelRecord(r["citation_key"], gt, status, r.get("rule"), r.get("annotator")))
        if aid in files:
            raise LabelError(f"bài {aid} có nhiều hơn một file nhãn ({p.name})")
        files[aid] = {"label_source": source, "records": records}
    return split, files


def load_papers(
    bibtex_df: Any,
    references_df: Any,
    split: Mapping[str, Mapping[str, str]],
    label_files: Mapping[str, Mapping[str, Any]],
    raw_text_column: str = "raw_text",
) -> list[Paper]:
    """Gộp bảng ``bibtex`` và ``references`` theo bài, chỉ giữ các bài có trong ``split``.

    Khử trùng: entry theo ``citation_key`` (giữ dòng đầu); reference theo ``ref_arxiv_id`` (ưu
    tiên dòng có tiêu đề). Pool sắp tăng dần theo ``ref_arxiv_id`` để tie-break xác định.
    """
    bib_by = {k: g for k, g in bibtex_df.groupby("arxiv_id")}
    ref_by = {k: g for k, g in references_df.groupby("arxiv_id")}
    papers: list[Paper] = []
    for aid in sorted(split):
        bib = bib_by.get(aid)
        ref = ref_by.get(aid)
        entries: list[Bibtex] = []
        raws: list[str | None] = []
        if bib is not None:
            seen: set[str] = set()
            for row in bib.to_dict("records"):
                key = row["citation_key"]
                if key in seen:
                    continue
                seen.add(key)
                entries.append(_make(Bibtex, row, ["authors"]))
                raws.append(_none(row.get(raw_text_column)))
        cands: list[Reference] = []
        if ref is not None:
            best: dict[str, Reference] = {}
            for row in ref.to_dict("records"):
                r = _make(Reference, row, ["ref_authors"])
                rid = str(r.ref_arxiv_id).strip()
                old = best.get(rid)
                if old is None or (not old.ref_title and r.ref_title):
                    best[rid] = r
            cands = [best[k] for k in sorted(best)]
        lf = label_files.get(aid, {})
        papers.append(Paper(
            arxiv_id=aid, partition=split[aid]["partition"], source=split[aid]["label_source"],
            entries=entries, candidates=cands, label_records=list(lf.get("records", [])),
            file_source=lf.get("label_source"),
            raw_texts=raws if any(r is not None for r in raws) else None,
        ))
    return papers


# =============================================================================
# check_labels (ml_design 3.3)
# =============================================================================
#: Ngưỡng tối thiểu của đề (Data Labelling), tính THEO TỪNG THÀNH VIÊN rồi nhân với ``members``.
MANUAL_PUBS_PER_MEMBER = 5
MANUAL_ENTRIES_PER_MEMBER = 100
AUTO_PUBS_PER_MEMBER = 2
AUTO_MIN_FRACTION = 0.10  # tỉ lệ tối thiểu số bài auto trên số bài NGOÀI nhóm manual


def check_quota(
    papers: Sequence[Paper], *, members: int = 5, n_publications: int | None = None
) -> list[str]:
    """Kiểm tra ngưỡng gán nhãn TOÀN NHÓM của đề; trả về danh sách lỗi (rỗng = đạt).

    Đề (Data Labelling): nhóm ``k`` người cần ít nhất ``5k`` bài manual với tổng ít nhất
    ``100k`` entry đã gán; bài auto phải chiếm ít nhất 10% số bài NGOÀI nhóm manual và không
    dưới ``2k`` bài. Ngưỡng ≥ 20 entry cho từng bài manual được kiểm ở :func:`check_labels`.

    Args:
        papers: Mọi bài có trong ``split.json``.
        members: Số thành viên ``k``.
        n_publications: Tổng số bài của toàn bộ corpus (kể cả bài không gán nhãn), cần để tính
            mốc 10%. ``None`` thì bỏ qua mốc 10% (vẫn kiểm mốc ``2k``).

    Returns:
        Danh sách lỗi, mỗi lỗi bắt đầu bằng ``"quota:"``.
    """
    errors: list[str] = []
    manual = [p for p in papers if p.source == MANUAL]
    auto = [p for p in papers if p.source == AUTO]
    n_manual_entries = sum(len(p.label_records) for p in manual)

    need_manual = MANUAL_PUBS_PER_MEMBER * members
    if len(manual) < need_manual:
        errors.append(f"quota: {len(manual)} bài manual, cần >= {need_manual}")
    need_entries = MANUAL_ENTRIES_PER_MEMBER * members
    if n_manual_entries < need_entries:
        errors.append(f"quota: {n_manual_entries} entry manual đã gán, cần >= {need_entries}")

    need_auto = AUTO_PUBS_PER_MEMBER * members
    if n_publications is not None:
        outside_manual = n_publications - len(manual)
        if outside_manual < 0:
            errors.append(f"quota: n_publications={n_publications} nhỏ hơn số bài manual "
                          f"({len(manual)})")
        else:
            # Làm tròn LÊN: 10% của 2.475 bài là 247,5 -> cần 248 bài.
            need_auto = max(need_auto, math.ceil(AUTO_MIN_FRACTION * outside_manual))
    if len(auto) < need_auto:
        errors.append(f"quota: {len(auto)} bài auto, cần >= {need_auto}")
    return errors


def check_labels(
    papers: Sequence[Paper],
    *,
    members: int = 5,
    min_manual_entries: int = 20,
    label_files: Mapping[str, Any] | None = None,
    n_publications: int | None = None,
    enforce_quota: bool = True,
) -> list[str]:
    """Kiểm tra nhãn trước khi train; trả về danh sách lỗi (rỗng = đạt).

    ``members`` là số thành viên: đề quy định Test và Valid mỗi tập có ``members`` bài manual
    và ``members`` bài auto (đại lượng theo từng thành viên). ``ml_design.md`` mục 2.3 ghi
    "1 manual + 1 auto" tương ứng ``members=1``; nhóm 5 người dùng ``members=5``.

    ``enforce_quota`` bật thêm :func:`check_quota` (ngưỡng toàn nhóm, cần ``n_publications``
    cho mốc 10%). Tắt khi chỉ muốn kiểm tính hợp lệ của một phần nhãn (ví dụ đang gán dở).
    """
    errors: list[str] = []
    for p in papers:
        tag = f"bài {p.arxiv_id}"
        if p.partition not in PARTITIONS:
            errors.append(f"{tag}: partition '{p.partition}' không hợp lệ")
        if p.source not in SOURCES:
            errors.append(f"{tag}: label_source '{p.source}' không hợp lệ")
        if p.file_source is None and not p.label_records:
            errors.append(f"{tag}: có trong split.json nhưng không có file nhãn")
        elif p.file_source is not None and p.file_source != p.source:
            errors.append(f"{tag}: label_source file ({p.file_source}) khác split.json ({p.source})")

        pool = set(p.candidate_ids)
        bad_pool = [i for i in pool if not is_valid_arxiv_id(i)]
        if bad_pool:
            errors.append(f"{tag}: ref_arxiv_id sai dạng chuẩn: {bad_pool[:3]}")
        keys = set(p.keys)

        seen: set[str] = set()
        for r in p.label_records:
            where = f"{tag}/{r.citation_key}"
            if r.citation_key in seen:                                               # 3.3.1
                errors.append(f"{where}: citation_key trùng trong file nhãn")
            seen.add(r.citation_key)
            if r.citation_key not in keys:                                           # 3.3.1
                errors.append(f"{where}: không có trong bảng bibtex của bài")
            if r.groundtruth is not None:
                if not is_valid_arxiv_id(r.groundtruth):                             # 3.3.3
                    errors.append(f"{where}: groundtruth '{r.groundtruth}' sai dạng chuẩn")
                elif r.groundtruth not in pool:                                      # 3.3.2
                    errors.append(f"{where}: groundtruth '{r.groundtruth}' không nằm trong pool "
                                  "(đáp án ngoài pool phải ghi null)")
            if r.status not in (MATCHED, NO_MATCH, UNRESOLVED):                      # 3.3.6
                errors.append(f"{where}: label_status '{r.status}' không hợp lệ")
            elif (r.groundtruth is not None) != (r.status == MATCHED):
                errors.append(f"{where}: groundtruth null/khác null không khớp status '{r.status}'")
            elif r.status == NO_MATCH and p.source != MANUAL:
                errors.append(f"{where}: no_match_confirmed chỉ dành cho bài manual")
            elif r.status == UNRESOLVED and p.source != AUTO:
                errors.append(f"{where}: bài manual không có unresolved")
            if p.source == AUTO and r.status == MATCHED and not r.rule:
                errors.append(f"{where}: record auto matched phải có rule")

        if p.source == AUTO:                                                         # 3.3.4
            missing = keys - seen
            if missing:
                errors.append(f"{tag}: bài auto thiếu nhãn cho {len(missing)} entry "
                              f"(ví dụ {sorted(missing)[:3]})")
        elif p.source == MANUAL and len(p.label_records) < min_manual_entries:
            errors.append(f"{tag}: bài manual chỉ có {len(p.label_records)} entry "
                          f"(cần >= {min_manual_entries})")

    counts: dict[tuple[str, str], int] = {}
    for p in papers:
        counts[(p.partition, p.source)] = counts.get((p.partition, p.source), 0) + 1  # 3.3.5
    for part in ("valid", "test"):
        for src in SOURCES:
            if counts.get((part, src), 0) != members:
                errors.append(f"split: {part} có {counts.get((part, src), 0)} bài {src}, "
                              f"cần đúng {members}")
    if counts.get(("train", MANUAL), 0) == 0:
        errors.append("split: tập train không có bài manual")
    ids = [p.arxiv_id for p in papers]
    if len(ids) != len(set(ids)):
        errors.append("split: có bài xuất hiện nhiều lần")
    if enforce_quota:
        errors += check_quota(papers, members=members, n_publications=n_publications)
    return errors


def assert_labels_ok(papers: Sequence[Paper], **kwargs: Any) -> None:
    errors = check_labels(papers, **kwargs)
    if errors:
        shown = "\n  - ".join(errors[:30])
        more = f"\n  ... và {len(errors) - 30} lỗi khác" if len(errors) > 30 else ""
        raise LabelError(f"check_labels thấy {len(errors)} lỗi:\n  - {shown}{more}")


# =============================================================================
# Ma trận đặc trưng
# =============================================================================
def _features_fingerprint() -> str:
    return hashlib.sha1(Path(feat.__file__).read_bytes()).hexdigest()[:12]


def _tensor_job(job: tuple[str, list[Bibtex], list[Reference], list[str | None] | None]):
    aid, entries, cands, raws = job
    return aid, feat.build_feature_matrix(entries, cands, raws, fill_missing=False)


def build_tensors(
    papers: Sequence[Paper], workers: int = 1, cache_dir: str | Path | None = None
) -> None:
    """Điền ``paper.tensor`` = ``build_feature_matrix`` trên TOÀN BỘ entry và pool của bài.

    Hạng theo cột (``composite_col_rank_in_paper``) so entry với mọi entry khác của bài, nên phải
    dùng cả entry chưa gán nhãn (``ml_design`` 5.2). Cache theo nội dung + mã nguồn features.py,
    nên đổi features.py tự vô hiệu cache cũ.
    """
    fp = _features_fingerprint()
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    def key_of(p: Paper) -> str:
        blob = repr((fp, feat.FEATURE_NAMES, p.entries, p.candidates, p.raw_texts)).encode()
        return hashlib.sha1(blob).hexdigest()

    todo: list[Paper] = []
    for p in papers:
        f = cache / f"{key_of(p)}.npy" if cache else None
        if f is not None and f.exists():
            p.tensor = np.load(f)
        else:
            todo.append(p)
    jobs = [(p.arxiv_id, p.entries, p.candidates, p.raw_texts) for p in todo]
    by_id = {p.arxiv_id: p for p in todo}
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_tensor_job, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
            for aid, t in results:
                by_id[aid].tensor = t
    else:
        for job in jobs:
            aid, t = _tensor_job(job)
            by_id[aid].tensor = t
    if cache:
        for p in todo:
            np.save(cache / f"{key_of(p)}.npy", p.tensor)
    log.info("đã dựng ma trận đặc trưng cho %d bài (%d tính mới)", len(papers), len(todo))


def _view(paper: Paper, filled: bool) -> np.ndarray:
    if paper.tensor is None:
        raise RuntimeError(f"bài {paper.arxiv_id} chưa có tensor; gọi build_tensors trước")
    return feat.fill_missing_features(paper.tensor) if filled else paper.tensor


def _rows_batch(
    paper: Paper,
    rows: Sequence[int],
    filled: bool,
    y_gold: Sequence[str | None] | None = None,
    weights: Sequence[float] | None = None,
) -> RankingBatch:
    """Batch của các hàng entry ``rows`` trong bài: mỗi hàng một query, pool là cả bài."""
    ids = paper.candidate_ids
    n, r = len(ids), len(rows)
    T = _view(paper, filled)[list(rows)]
    X = T.reshape(r * n, T.shape[-1])
    y = None
    if y_gold is not None:
        pool = np.array(ids, dtype=object)
        y = np.concatenate([(pool == g).astype(float) if g is not None else np.zeros(n)
                            for g in y_gold]) if r else np.zeros(0)
    w = None if weights is None else np.repeat(np.asarray(weights, dtype=float), n)
    return RankingBatch(
        X=X, groups=np.full(r, n, dtype=np.intp), y=y, sample_weight=w,
        candidate_ids=np.tile(np.array(ids, dtype=object), r),
        query_ids=[(paper.arxiv_id, paper.keys[i]) for i in rows],
        feature_names=feat.FEATURE_NAMES,
    )


# ----------------------------------------------------------------------- tập đánh giá
@dataclass
class EvalSet:
    """Mọi entry đã gán nhãn của một nhóm bài (|Q| theo đề). ``truth`` None = không có đáp án."""

    batch: RankingBatch | None
    truth: list[str | None]
    sources: list[str]
    arxiv_ids: list[str]


def build_eval_set(papers: Sequence[Paper], filled: bool) -> EvalSet:
    batches: list[RankingBatch] = []
    truth: list[str | None] = []
    sources: list[str] = []
    ids: list[str] = []
    for p in papers:
        index = {k: i for i, k in enumerate(p.keys)}
        rows, gold = [], []
        for rec in p.label_records:
            if rec.citation_key in index:
                rows.append(index[rec.citation_key])
                gold.append(rec.groundtruth if rec.status == MATCHED else None)
        if not rows:
            continue
        batches.append(_rows_batch(p, rows, filled, y_gold=gold))
        truth += gold
        sources += [p.source] * len(rows)
        ids += [p.arxiv_id] * len(rows)
    return EvalSet(RankingBatch.concat(batches) if batches else None, truth, sources, ids)


class EvalCache:
    """Dựng EvalSet theo (partition, view) một lần."""

    def __init__(self, papers: Sequence[Paper]) -> None:
        self.papers = list(papers)
        self._cache: dict[tuple[str, bool], EvalSet] = {}

    def get(self, partition: str, filled: bool) -> EvalSet:
        key = (partition, filled)
        if key not in self._cache:
            self._cache[key] = build_eval_set([p for p in self.papers if p.partition == partition], filled)
        return self._cache[key]


def evaluate_partition(model: MatchingModel, filled: bool, cache: EvalCache,
                       partition: str, k: int = K) -> dict[str, Any]:
    es = cache.get(partition, filled)
    if es.batch is None:
        return mrr_report([], [], sources=[], arxiv_ids=[], k=k)
    return evaluate_ranking(model, es.batch, es.truth, sources=es.sources,
                            arxiv_ids=es.arxiv_ids, k=k)


# =============================================================================
# Chính sách huấn luyện (ml_design 3.5)
# =============================================================================
@dataclass(frozen=True)
class TrainPolicy:
    name: str  # A | B | C | D
    include_auto_matched: bool = False
    w_auto: float = 1.0
    mask_rule_features: bool = False
    include_manual_no_match: bool = False
    auto_unresolved_as_negative: bool = False  # chỉ cấu hình D (đối chứng)

    @property
    def label(self) -> str:
        return self.name if self.name in ("A", "D") else f"{self.name}(w={self.w_auto:g})"

    @property
    def selectable(self) -> bool:
        return self.name != "D"


def policy_grid(w_auto: Sequence[float] = DEFAULT_W_AUTO) -> list[TrainPolicy]:
    grid = [TrainPolicy("A")]
    for w in w_auto:
        grid.append(TrainPolicy("B", include_auto_matched=True, w_auto=w, mask_rule_features=True))
    for w in w_auto:
        grid.append(TrainPolicy("C", include_auto_matched=True, w_auto=w, mask_rule_features=True,
                                include_manual_no_match=True))
    grid.append(TrainPolicy("D", include_auto_matched=True, w_auto=1.0,
                            auto_unresolved_as_negative=True))
    return grid


def build_train_batch(papers: Sequence[Paper], policy: TrainPolicy, filled: bool) -> RankingBatch | None:
    """Batch huấn luyện theo bảng 9.1 của ``ml_design``; ``None`` nếu không có hàng nào."""
    batches: list[RankingBatch] = []
    rules: list[str | None] = []
    for p in papers:
        if p.partition != "train":  # chốt chặn rò rỉ: Valid/Test không bao giờ vào huấn luyện
            continue
        if not p.candidates:  # pool rỗng: không có cặp nào để học
            continue
        index = {k: i for i, k in enumerate(p.keys)}
        rows, gold, weights, row_rules = [], [], [], []
        for rec in p.label_records:
            i = index.get(rec.citation_key)
            if i is None:
                continue
            if p.source == MANUAL:
                if rec.status == MATCHED:
                    w = 1.0
                elif rec.status == NO_MATCH and policy.include_manual_no_match:
                    w = 1.0
                else:
                    continue
                rule = None
            else:
                if not policy.include_auto_matched:
                    continue
                if rec.status == MATCHED:
                    w = policy.w_auto
                elif rec.status == UNRESOLVED and policy.auto_unresolved_as_negative:
                    w = 1.0  # đối chứng D: cố ý coi null là âm cứng
                else:
                    continue  # unresolved: MASK khỏi loss, không phải mẫu âm
                rule = rec.rule if rec.status == MATCHED else None
            rows.append(i)
            gold.append(rec.groundtruth if rec.status == MATCHED else None)
            weights.append(w)
            row_rules.append(rule)
        if rows:
            batches.append(_rows_batch(p, rows, filled, y_gold=gold, weights=weights))
            rules += row_rules
    if not batches:
        return None
    batch = RankingBatch.concat(batches)
    if policy.mask_rule_features:
        arr = np.array(rules, dtype=object)
        for rule, cols in RULE_MASK_FEATURES.items():
            sel = arr == rule
            if sel.any():
                batch = batch.with_features_masked(sel, feat.columns_to_mask(cols))
    return batch


# =============================================================================
# Mô hình
# =============================================================================
class LGBMRankerModel(MatchingModel):
    """LightGBM LambdaRank: ``group`` = một entry, các cặp của entry là các ứng viên.

    Chấp nhận NaN. Chỉ nên thử khi pointwise cho MRR thấp (``ml_design`` mục 2.2, 9.4).
    """

    def __init__(self, **params: Any) -> None:
        self.params = params

    def _fit(self, X, y, groups, sample_weight) -> None:
        from lightgbm import LGBMRanker

        self.ranker_ = LGBMRanker(objective="lambdarank", **self.params)
        self.ranker_.fit(X, (y > 0.5).astype(int), group=groups, sample_weight=sample_weight)

    def _score(self, X):
        return self.ranker_.predict(X)

    def feature_importance(self) -> dict[str, float] | None:
        r = getattr(self, "ranker_", None)
        if r is None or self.feature_names is None:
            return None
        return {n: float(v) for n, v in zip(self.feature_names, r.feature_importances_)}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], MatchingModel]
    filled: bool  # True: tensor đã fill_missing_features; False: giữ NaN thô
    trainable: bool = True


def make_specs(seed: int = 0, n_estimators: int = 300, which: Sequence[str] | None = None
               ) -> dict[str, ModelSpec]:
    """Naive baseline, Logistic Regression, LightGBM pointwise, LightGBM LambdaRank."""
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression

    gbm = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=15, min_child_samples=10,
               reg_lambda=1.0, random_state=seed, n_jobs=1, verbose=-1,
               deterministic=True, force_row_wise=True)
    specs = {
        NAIVE_NAME: ModelSpec(NAIVE_NAME, lambda: FeatureRankBaseline("title_lev_sim"),
                              filled=False, trainable=False),  # NaN F01 xếp sau cùng đúng mục 2.2
        "logreg": ModelSpec(
            "logreg",
            lambda: PointwiseClassifierModel(
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000), impute=True),
            filled=True),
        "lgbm": ModelSpec(
            "lgbm",
            lambda: PointwiseClassifierModel(
                LGBMClassifier(class_weight="balanced", **gbm), impute=False),
            filled=False),
        "lgbm_rank": ModelSpec("lgbm_rank", lambda: LGBMRankerModel(**gbm), filled=False),
    }
    if which is not None:
        unknown = set(which) - set(specs)
        if unknown:
            raise ValueError(f"mô hình không có: {sorted(unknown)}; có {sorted(specs)}")
        specs = {k: v for k, v in specs.items() if k in which or k == NAIVE_NAME}
    return specs


def fit_policy(spec: ModelSpec, papers: Sequence[Paper], policy: TrainPolicy) -> MatchingModel | None:
    """Fit ``spec`` trên ``papers`` với ``policy``; ``None`` nếu không đủ dữ liệu (thiếu dương/âm)."""
    batch = build_train_batch(papers, policy, spec.filled)
    if batch is None:
        return None
    model = spec.factory()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(batch)
    except ValueError as exc:  # ví dụ chỉ có một lớp nhãn
        log.warning("%s/%s: không fit được (%s)", spec.name, policy.label, exc)
        return None
    return model


# =============================================================================
# Ablation, LOPO, chọn chính sách
# =============================================================================
def _nz(x: float, default: float = float("-inf")) -> float:
    return default if x is None or math.isnan(x) else x


def lopo_manual(spec: ModelSpec, policy: TrainPolicy, train_papers: Sequence[Paper],
                k: int = K) -> float:
    """Leave-one-publication-out trên các bài manual của Train: MRR@k gộp mọi entry bị giữ lại.

    Mỗi lần bỏ một bài manual ra khỏi dữ liệu train (bài auto giữ nguyên), fit, rồi đo chính bài
    đó trên toàn bộ entry đã gán nhãn. Không dùng Valid hay Test.
    """
    predicted: list[list[str | None]] = []
    truth: list[str | None] = []
    for held in [p for p in train_papers if p.source == MANUAL]:
        es = build_eval_set([held], spec.filled)
        if es.batch is None:
            continue
        model = fit_policy(spec, [p for p in train_papers if p is not held], policy)
        if model is None:
            continue
        predicted += [list(r) for r in model.rank_ids(es.batch, k)]
        truth += es.truth
    return mrr_at_k(predicted, truth, k) if truth else float("nan")


def run_ablation(
    specs: Mapping[str, ModelSpec],
    papers: Sequence[Paper],
    policies: Sequence[TrainPolicy],
    *,
    lopo: bool = True,
    k: int = K,
) -> list[dict[str, Any]]:
    """Mỗi (mô hình, chính sách): fit trên Train, đo Valid; tùy chọn LOPO. Không đụng Test."""
    train = [p for p in papers if p.partition == "train"]
    cache = EvalCache(papers)
    rows: list[dict[str, Any]] = []
    for spec in specs.values():
        if not spec.trainable:
            continue
        for pol in policies:
            model = fit_policy(spec, train, pol)
            row: dict[str, Any] = {"model": spec.name, "policy": pol.name, "w_auto": pol.w_auto,
                                   "label": pol.label, "selectable": pol.selectable}
            if model is None:
                row.update(valid_overall=float("nan"), valid_manual=float("nan"),
                           valid_auto=float("nan"), lopo_manual=float("nan"), fitted=False)
            else:
                rep = evaluate_partition(model, spec.filled, cache, "valid", k)
                row.update(valid_overall=rep["overall"]["mrr"], valid_manual=rep["manual"]["mrr"],
                           valid_auto=rep["auto"]["mrr"], fitted=True,
                           lopo_manual=lopo_manual(spec, pol, train, k) if lopo else float("nan"))
            rows.append(row)
            log.info("ablation %-9s %-10s valid_manual=%s lopo=%s", spec.name, pol.label,
                     row["valid_manual"], row["lopo_manual"])
    return rows


def select_policy(rows: Sequence[Mapping[str, Any]], model_name: str,
                  policies: Sequence[TrainPolicy]) -> TrainPolicy | None:
    """Chọn chính sách của một mô hình: MRR Valid-manual cao nhất (nếu N/A thì Valid chung),
    hòa thì LOPO-manual, hòa nữa thì cấu hình đứng trước (đơn giản hơn). Loại D (đối chứng).
    """
    by_label = {p.label: p for p in policies}
    best, best_key = None, None
    for i, r in enumerate(rows):
        if r["model"] != model_name or not r["selectable"] or not r["fitted"]:
            continue
        primary = _nz(r["valid_manual"], _nz(r["valid_overall"]))
        key = (primary, _nz(r["lopo_manual"]), -i)
        if best_key is None or key > best_key:
            best, best_key = r, key
    return by_label[best["label"]] if best else None


# =============================================================================
# Dự đoán cho bảng predictions
# =============================================================================
def predict_rows(model: MatchingModel, filled: bool, papers: Sequence[Paper],
                 k: int = K) -> list[PredictionRow]:
    """Một dòng cho MỌI entry của mỗi bài (kể cả entry chưa gán nhãn, groundtruth null)."""
    out: list[PredictionRow] = []
    for p in papers:
        if not p.entries:
            continue
        labels = p.labels
        batch = _rows_batch(p, range(len(p.entries)), filled)
        ids = model.rank_ids(batch, k)
        for i, entry in enumerate(p.entries):
            rec = labels.get(entry.citation_key)
            gt = rec.groundtruth if rec is not None and rec.status == MATCHED else None
            cands = tuple(None if c is None else str(c) for c in ids[i])
            out.append(PredictionRow(p.arxiv_id, p.partition, entry.citation_key, gt, cands))
    return out


# =============================================================================
# Toàn bộ pipeline
# =============================================================================
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (np.floating, np.integer)):
        return _jsonable(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_pipeline(
    papers: Sequence[Paper],
    *,
    models: Sequence[str] | None = None,
    w_auto: Sequence[float] = DEFAULT_W_AUTO,
    lopo: bool = True,
    seed: int = 0,
    n_estimators: int = 300,
    k: int = K,
    require_beat: bool = False,
    out_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ablation -> chọn trên Valid -> fit lại -> đo Valid/Test -> chọn mô hình cuối -> xuất.

    Mô hình cuối là mô hình ML có MRR Valid-manual cao nhất *và thắng baseline* trên Valid-manual;
    nếu không có mô hình nào thắng thì dùng chính Naive baseline (cảnh báo to) hoặc ném lỗi khi
    ``require_beat``. Test chỉ được đo sau khi mọi lựa chọn đã chốt và không ảnh hưởng lựa chọn.
    """
    specs = make_specs(seed, n_estimators, models)
    policies = policy_grid(w_auto)
    train = [p for p in papers if p.partition == "train"]
    cache = EvalCache(papers)

    ablation = run_ablation(specs, papers, policies, lopo=lopo, k=k)

    chosen: dict[str, TrainPolicy] = {}
    fitted: dict[str, MatchingModel] = {}
    for name, spec in specs.items():
        if not spec.trainable:
            continue
        pol = select_policy(ablation, name, policies)
        model = fit_policy(spec, train, pol) if pol else None
        if model is None:
            log.warning("%s: không có chính sách nào fit được, bỏ qua", name)
            continue
        chosen[name], fitted[name] = pol, model
    base_model = specs[NAIVE_NAME].factory()
    fitted_all: dict[str, MatchingModel] = {NAIVE_NAME: base_model, **fitted}

    reports: dict[str, dict[str, Any]] = {"valid": {}, "test": {}}
    for part in ("valid", "test"):
        for name, model in fitted_all.items():
            reports[part][name] = evaluate_partition(model, specs[name].filled, cache, part, k)

    cmp_valid = compare_with_baseline(reports["valid"], NAIVE_NAME, primary=MANUAL)
    cmp_test = compare_with_baseline(reports["test"], NAIVE_NAME, primary=MANUAL)

    winners = [n for n, c in cmp_valid.items() if c["beats_baseline"]]
    if winners:
        final_name = max(winners, key=lambda n: _nz(reports["valid"][n][MANUAL]["mrr"]))
    else:
        final_name = NAIVE_NAME
        msg = "KHÔNG có mô hình ML nào thắng Naive baseline trên Valid-manual"
        if require_beat:
            raise RuntimeError(msg)
        log.warning("%s: dùng Naive baseline làm đầu ra cuối. Xem lại feature và chính sách nhãn.", msg)
    final_model = fitted_all[final_name]

    result: dict[str, Any] = {
        "k": k, "seed": seed, "final_model": final_name,
        "chosen_policy": {n: p.label for n, p in chosen.items()},
        "ablation": ablation, "valid": reports["valid"], "test": reports["test"],
        "vs_baseline_valid": cmp_valid, "vs_baseline_test": cmp_test,
        "feature_importance": {n: m.feature_importance() for n, m in fitted_all.items()
                               if m.feature_importance() is not None},
        "n_papers": {part: sum(p.partition == part for p in papers) for part in PARTITIONS},
    }

    if predictions_path is not None:
        rows = predict_rows(final_model, specs[final_name].filled, papers, k)
        pools = {p.arxiv_id: p.candidate_ids for p in papers}
        write_predictions(predictions_path, rows, pools, {p.arxiv_id: p.partition for p in papers})
        result["predictions"] = {"path": str(predictions_path), "rows": len(rows)}
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        final_model.save(Path(out_dir) / f"{final_name}.pkl")  # *.pkl: KHÔNG commit (.gitignore)
    if reports_dir is not None:
        write_reports(result, reports_dir)
    return result


def write_reports(result: Mapping[str, Any], reports_dir: str | Path) -> None:
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "m3_results.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    abl = result["ablation"]
    if abl:
        with open(d / "m3_ablation.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(abl[0]))
            w.writeheader()
            w.writerows(_jsonable(abl))
    lines = [f"# Kết quả Milestone 3 (MRR@{result['k']})", "",
             f"Mô hình cuối: **{result['final_model']}** · chính sách: {result['chosen_policy']}", ""]
    for part in ("valid", "test"):
        lines += [f"## {part.capitalize()}", "", format_results_table(result[part]), ""]
        for n, c in result[f"vs_baseline_{part}"].items():
            lines.append(f"- {c['verdict']}")
        lines.append("")
    lines += ["## Ablation chính sách nhãn (Valid)", "",
              "| Mô hình | Chính sách | Valid manual | Valid auto | LOPO manual |", "|---|---|---|---|---|"]
    fmt = lambda x: "N/A" if x is None or math.isnan(x) else f"{x:.4f}"  # noqa: E731
    for r in abl:
        lines.append(f"| {r['model']} | {r['label']} | {fmt(r['valid_manual'])} | "
                     f"{fmt(r['valid_auto'])} | {fmt(r['lopo_manual'])} |")
    (d / "m3_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================
def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bibtex", required=True, help="bảng bibtex (.parquet/.csv/.jsonl)")
    ap.add_argument("--references", required=True, help="bảng references (.parquet/.csv/.jsonl)")
    ap.add_argument("--labels", default="data/labels", help="thư mục chứa split.json và <id>.json")
    ap.add_argument("--out-dir", default="models", help="nơi lưu mô hình (*.pkl, không commit)")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--predictions", default="data/predictions/predictions.parquet")
    ap.add_argument("--cache-dir", default=None, help="cache ma trận đặc trưng (theo nội dung)")
    ap.add_argument("--workers", type=int, default=1, help="số tiến trình dựng đặc trưng")
    ap.add_argument("--members", type=int, default=5, help="số thành viên (Valid/Test: mỗi tập "
                    "có đúng ngần ấy bài manual và auto)")
    ap.add_argument("--models", nargs="*", default=None, help="logreg lgbm lgbm_rank")
    ap.add_argument("--w-auto", nargs="*", type=float, default=list(DEFAULT_W_AUTO))
    ap.add_argument("--no-lopo", action="store_true")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-publications", type=int, default=None,
                    help="tổng số bài của corpus (vd 25000), để kiểm mốc auto >= 10%% số bài "
                    "ngoài nhóm manual; bỏ trống thì chỉ kiểm mốc 2k")
    ap.add_argument("--no-quota", action="store_true",
                    help="bỏ kiểm ngưỡng toàn nhóm (chỉ dùng khi đang gán nhãn dở)")
    ap.add_argument("--check-only", action="store_true", help="chỉ chạy check_labels")
    ap.add_argument("--require-beat", action="store_true",
                    help="lỗi nếu không mô hình ML nào thắng Naive baseline trên Valid-manual")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    split, label_files = load_labels(args.labels)
    papers = load_papers(read_table(args.bibtex), read_table(args.references), split, label_files)
    if args.n_publications is None and not args.no_quota:
        log.warning("thiếu --n-publications: KHÔNG kiểm được mốc auto >= 10%% số bài ngoài manual")
    assert_labels_ok(papers, members=args.members, n_publications=args.n_publications,
                     enforce_quota=not args.no_quota)
    log.info("check_labels đạt: %d bài (%s)", len(papers), {
        f"{part}/{src}": sum(p.partition == part and p.source == src for p in papers)
        for part in PARTITIONS for src in SOURCES})
    if args.check_only:
        return

    build_tensors(papers, workers=args.workers, cache_dir=args.cache_dir)
    result = run_pipeline(
        papers, models=args.models, w_auto=args.w_auto, lopo=not args.no_lopo, seed=args.seed,
        n_estimators=args.trees, require_beat=args.require_beat, out_dir=args.out_dir,
        reports_dir=args.reports_dir, predictions_path=args.predictions)
    print(format_results_table(result["test"]))
    for c in result["vs_baseline_valid"].values():
        print(c["verdict"])
    print(f"Mô hình cuối: {result['final_model']} · predictions: {result.get('predictions')}")


if __name__ == "__main__":
    main()