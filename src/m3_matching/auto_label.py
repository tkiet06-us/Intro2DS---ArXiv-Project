"""Gán nhãn tự động bằng luật cổ điển (regex, độ giống chuỗi) và đo precision/recall của luật (M3).

Đề (Data Labelling): gán nhãn tự động cho ít nhất 10% số bài ngoài nhóm manual, bằng "classic
tools such as regex or string similarity heuristics". Module này:

1. Tính các điểm giống nhau cho mọi cặp (mục BibTeX, ứng viên) của một bài (:func:`score_paper`).
2. Định nghĩa luật (:class:`Rule`): luật ID arXiv (regex) và luật tiêu đề có tham số (độ đo,
   ngưỡng, khoảng cách với ứng viên thứ hai, ràng buộc tác giả/năm).
3. Áp một chuỗi luật theo thứ tự ưu tiên (:func:`label_paper`) -> label record đúng định dạng
   ``ml_design.md`` mục 3 (``matched`` có ``rule``; không luật nào khớp thì ``unresolved``).
4. Đo precision/recall từng luật và cả chuỗi luật so với một bộ nhãn chuẩn
   (:func:`evaluate_rules`). Bộ nhãn chuẩn là nhãn tay (``data/labels``) hoặc, khi chưa có,
   "nhãn bạc" (:func:`silver_gold`) suy từ ID arXiv ghi ngay trong mục BibTeX.

Cách chạy (từ thư mục gốc repo)::

    # Khảo sát luật trên nhãn bạc (chưa cần nhãn tay), xuất bảng precision/recall
    python -m src.m3_matching.auto_label evaluate --raw-root data/raw/papers_kiet --limit 300

    # Khi đã có nhãn tay: đo lại trên nhãn tay
    python -m src.m3_matching.auto_label evaluate --raw-root data/raw/papers_kiet \\
        --gold-labels data/labels

    # Sinh file nhãn auto cho các bài trong danh sách
    python -m src.m3_matching.auto_label label --raw-root data/raw/papers_kiet \\
        --ids-file data/labels/auto_ids.txt --out-dir data/labels

Tham khảo: Levenshtein (Levenshtein, 1966) và token_set_ratio/partial_ratio của thư viện
rapidfuzz (https://github.com/rapidfuzz/RapidFuzz).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from src.common.ids import to_folder_name
from src.common.utils_io import write_json
from src.m3_matching import features as feat
from src.m3_matching.corpus import PaperRecord, iter_paper_dirs, load_paper
from src.schemas import Bibtex, Reference

logger = logging.getLogger(__name__)

MATCHED, UNRESOLVED, NO_MATCH = "matched", "unresolved", "no_match_confirmed"
ARXIV_RULE = "arxiv_in_url"  # tên luật mà train.RULE_MASK_FEATURES che feature tương ứng
METRICS = ("lev", "lev_compact", "token_set", "partial")
AUTHOR_MODES = ("none", "first", "any")

#: Chuỗi luật mặc định, chọn bằng khảo sát trên mẫu 297 bài ngẫu nhiên của dải 2410.13025-18024
#: (14/11/2026; 23.799 mục, 7.630 ứng viên). Trên nhãn bạc (3.488 mục), cả chuỗi đạt
#: precision 0,997: 4 dự đoán "sai" đều là mục mà TÁC GIẢ ghi nhầm ID arXiv (luật tiêu đề đúng).
#: Soát tay 40 cặp khớp mờ nhất (0,90 <= lev < 0,95) trên mục KHÔNG ghi ID: 2 cặp sai (bài
#: cùng chuỗi, ví dụ "One-/Two-dimensional Ising model"), tức ~0,06% số khớp. Luật lev < 0,85
#: sinh lỗi thật (ví dụ "Qwen2 Technical Report" -> "GPT-4 Technical Report"), nên không dùng.
#: Sửa ở đây khi khảo sát lại trên nhãn tay (lệnh ``evaluate --gold-labels``).
DEFAULT_RULE_NAMES: tuple[str, ...] = (
    ARXIV_RULE,
    "title_lev_compact>=1.00|any",
    "title_lev>=0.90|m0.05|any",
)
#: Mục tiêu precision khi chọn luật: nhãn auto vào train như nhãn thật (trọng số w_auto), nên
#: nhãn sai gây hại hơn nhãn thiếu (thiếu chỉ thành ``unresolved``, bị mask khi train).
TARGET_PRECISION = 0.98


# =============================================================================
# Điểm giống nhau của mọi cặp trong một bài
# =============================================================================
@dataclass
class PaperScores:
    """Ma trận điểm (m mục x n ứng viên) của một bài. NaN = thiếu dữ liệu để so."""

    entry_keys: list[str]
    candidate_ids: list[str]
    id_match: np.ndarray  # bool: ID arXiv của ứng viên xuất hiện trong url/journal/doi của mục
    entry_has_id: np.ndarray  # bool (m,): mục có ít nhất một ID arXiv
    lev: np.ndarray
    lev_compact: np.ndarray  # Levenshtein sau khi bỏ mọi khoảng trắng (xem _compact)
    token_set: np.ndarray
    partial: np.ndarray
    first_author: np.ndarray  # 1/0, NaN nếu một bên không có tác giả
    any_author: np.ndarray  # 1 nếu chung ít nhất một họ, NaN nếu thiếu tác giả
    year_diff: np.ndarray  # |năm ứng viên - năm mục|, NaN nếu thiếu năm
    short_len: np.ndarray  # độ dài tiêu đề NGẮN hơn của cặp (ký tự, sau chuẩn hóa)

    def metric(self, name: str) -> np.ndarray:
        return getattr(self, name)


def _title(text: str | None) -> str:
    # Tường minh remove_stopwords=False: luật so cả tiêu đề, không phụ thuộc mặc định preprocess.
    return feat.preprocess(text, remove_stopwords=False)


def _compact(title: str) -> str:
    """Bỏ khoảng trắng: M2 hiện tách chữ hoa trong ngoặc nhọn thành "V ariational A pproach"
    (từ ``{V}ariational {A}pproach``), làm Levenshtein thường bị phạt oan."""
    return title.replace(" ", "")


def _entry_ids(entry: Bibtex) -> set[str]:
    text = " ".join(f for f in (entry.url, entry.journal, entry.doi) if f)
    return feat.extract_arxiv_ids(text)


def _similarity(a: list[str], b: list[str], scorer, scale: float) -> np.ndarray:
    """``cdist`` của rapidfuzz, chia về 0..1; ô có chuỗi rỗng ở một bên là NaN."""
    out = np.full((len(a), len(b)), np.nan)
    rows = [i for i, s in enumerate(a) if s]
    cols = [j for j, s in enumerate(b) if s]
    if rows and cols:
        sub = process.cdist([a[i] for i in rows], [b[j] for j in cols], scorer=scorer,
                            dtype=np.float64, workers=1)
        out[np.ix_(rows, cols)] = sub / scale
    return out


def score_paper(entries: Sequence[Bibtex], candidates: Sequence[Reference]) -> PaperScores:
    """Tính mọi điểm cần cho luật trên ma trận m x n của một bài.

    Args:
        entries: Mục BibTeX của bài (sau dedup).
        candidates: Pool reference có arXiv ID của cùng bài.

    Returns:
        ``PaperScores`` với các ma trận (m, n).
    """
    m, n = len(entries), len(candidates)
    e_titles = [_title(e.title) for e in entries]
    c_titles = [_title(c.ref_title) for c in candidates]
    c_ids = [feat._normalize_arxiv_id(c.ref_arxiv_id) for c in candidates]
    e_ids = [_entry_ids(e) for e in entries]
    e_auth = [feat.author_keys(e.authors) for e in entries]
    c_auth = [feat.author_keys(c.ref_authors) for c in candidates]
    e_year = [e.year for e in entries]
    c_year = [feat.parse_year(c.ref_submission_date) for c in candidates]

    id_match = np.array([[cid in ids for cid in c_ids] for ids in e_ids], dtype=bool).reshape(m, n)
    first = np.full((m, n), np.nan)
    anya = np.full((m, n), np.nan)
    ydiff = np.full((m, n), np.nan)
    short = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if e_auth[i] and c_auth[j]:
                first[i, j] = float(e_auth[i][0][0] == c_auth[j][0][0])
                anya[i, j] = float(bool({a[0] for a in e_auth[i]} & {a[0] for a in c_auth[j]}))
            if e_year[i] is not None and c_year[j] is not None:
                ydiff[i, j] = abs(c_year[j] - e_year[i])
            short[i, j] = min(len(e_titles[i]), len(c_titles[j]))
    return PaperScores(
        entry_keys=[e.citation_key for e in entries],
        candidate_ids=[c.ref_arxiv_id for c in candidates],
        id_match=id_match,
        entry_has_id=np.array([bool(ids) for ids in e_ids], dtype=bool),
        lev=_similarity(e_titles, c_titles, Levenshtein.normalized_similarity, 1.0),
        lev_compact=_similarity([_compact(t) for t in e_titles], [_compact(t) for t in c_titles],
                                Levenshtein.normalized_similarity, 1.0),
        token_set=_similarity(e_titles, c_titles, fuzz.token_set_ratio, 100.0),
        partial=_similarity(e_titles, c_titles, fuzz.partial_ratio, 100.0),
        first_author=first,
        any_author=anya,
        year_diff=ydiff,
        short_len=short,
    )


# =============================================================================
# Luật
# =============================================================================
@dataclass(frozen=True)
class Rule:
    """Một luật gán nhãn. ``kind="arxiv_id"`` là luật regex; còn lại là luật tiêu đề.

    Luật tiêu đề chọn ứng viên có ``metric`` cao nhất của hàng và chỉ khớp khi:
    ``best >= threshold``, ``best - nhì >= margin`` (tránh hai ứng viên gần như nhau), tác giả
    thỏa ``author`` (``first``: trùng họ tác giả đầu; ``any``: chung ít nhất một họ; thiếu tác
    giả thì KHÔNG đạt), chênh năm ``<= max_year_diff`` (thiếu năm thì bỏ qua ràng buộc này) và
    tiêu đề ngắn hơn của cặp dài ít nhất ``min_title_len`` ký tự.
    """

    kind: str = "title"
    metric: str = "lev"
    threshold: float = 1.0
    margin: float = 0.0
    author: str = "none"
    max_year_diff: int | None = None
    min_title_len: int = 0

    @property
    def name(self) -> str:
        if self.kind == "arxiv_id":
            return ARXIV_RULE
        parts = [f"title_{self.metric}>={self.threshold:.2f}"]
        if self.margin:
            parts.append(f"m{self.margin:g}")
        if self.author != "none":
            parts.append(self.author)
        if self.max_year_diff is not None:
            parts.append(f"y{self.max_year_diff}")
        if self.min_title_len:
            parts.append(f"len{self.min_title_len}")
        return "|".join(parts)

    @classmethod
    def from_name(cls, name: str) -> Rule:
        """Dựng lại luật từ tên (ví dụ ``"title_lev>=0.90|m0.05|first|y1"``)."""
        if name == ARXIV_RULE:
            return cls(kind="arxiv_id")
        head, *opts = name.split("|")
        if not head.startswith("title_") or ">=" not in head:
            raise ValueError(f"tên luật không hợp lệ: {name!r}")
        metric, threshold = head[len("title_"):].split(">=")
        kwargs: dict = {"metric": metric, "threshold": float(threshold)}
        for opt in opts:
            if opt in AUTHOR_MODES:
                kwargs["author"] = opt
            elif opt.startswith("len"):
                kwargs["min_title_len"] = int(opt[3:])
            elif opt.startswith("m"):
                kwargs["margin"] = float(opt[1:])
            elif opt.startswith("y"):
                kwargs["max_year_diff"] = int(opt[1:])
            else:
                raise ValueError(f"tùy chọn luật không hợp lệ {opt!r} trong {name!r}")
        rule = cls(**kwargs)
        if rule.metric not in METRICS:
            raise ValueError(f"metric phải thuộc {METRICS}: {name!r}")
        return rule

    def apply(self, scores: PaperScores, i: int) -> tuple[int, float] | None:
        """Ứng viên (chỉ số cột, điểm) mà luật chọn cho mục ``i``, hoặc ``None``."""
        if not scores.candidate_ids:
            return None
        if self.kind == "arxiv_id":
            hits = np.flatnonzero(scores.id_match[i])
            # Mục ghi hai ID cùng nằm trong pool: không đoán, để luật sau hoặc unresolved.
            return (int(hits[0]), 1.0) if hits.size == 1 else None
        row = scores.metric(self.metric)[i]
        if np.all(np.isnan(row)):
            return None
        order = np.argsort(-np.nan_to_num(row, nan=-1.0), kind="stable")
        j = int(order[0])
        best = row[j]
        second = row[order[1]] if row.size > 1 and not np.isnan(row[order[1]]) else 0.0
        if best < self.threshold - 1e-9 or best - second < self.margin - 1e-9:
            return None
        if scores.short_len[i, j] < self.min_title_len:
            return None
        if self.author == "first" and scores.first_author[i, j] != 1.0:
            return None
        if self.author == "any" and scores.any_author[i, j] != 1.0:
            return None
        if self.max_year_diff is not None:
            diff = scores.year_diff[i, j]
            if not np.isnan(diff) and diff > self.max_year_diff:
                return None
        return j, float(best)


def default_rules() -> list[Rule]:
    """Chuỗi luật mặc định (thứ tự = ưu tiên)."""
    return [Rule.from_name(n) for n in DEFAULT_RULE_NAMES]


# =============================================================================
# Gán nhãn một bài
# =============================================================================
def predict_paper(
    scores: PaperScores, rules: Sequence[Rule], one_to_one: bool = False
) -> list[tuple[str | None, str | None]]:
    """Áp chuỗi luật: mỗi mục lấy kết quả của luật ĐẦU TIÊN khớp.

    Args:
        scores: Điểm của bài.
        rules: Luật theo thứ tự ưu tiên.
        one_to_one: True thì nếu nhiều mục cùng trỏ một ứng viên, chỉ giữ mục có luật ưu tiên
            hơn (rồi điểm cao hơn); các mục còn lại thành ``None``.

    Returns:
        Mỗi mục một cặp ``(ref_arxiv_id dự đoán hoặc None, tên luật hoặc None)``.
    """
    picks: list[tuple[int, int, float] | None] = []
    for i in range(len(scores.entry_keys)):
        chosen = None
        for r_idx, rule in enumerate(rules):
            hit = rule.apply(scores, i)
            if hit is not None:
                chosen = (hit[0], r_idx, hit[1])
                break
        picks.append(chosen)
    if one_to_one:
        best_for: dict[int, int] = {}
        for i, pick in enumerate(picks):
            if pick is None:
                continue
            other = best_for.get(pick[0])
            if other is None or (pick[1], -pick[2]) < (picks[other][1], -picks[other][2]):
                best_for[pick[0]] = i
        picks = [p if p is not None and best_for.get(p[0]) == i else None
                 for i, p in enumerate(picks)]
    return [(None, None) if p is None else (scores.candidate_ids[p[0]], rules[p[1]].name)
            for p in picks]


def label_paper(paper: PaperRecord, rules: Sequence[Rule], one_to_one: bool = False) -> dict:
    """Sinh nội dung file nhãn auto của một bài (định dạng ``ml_design.md`` mục 3).

    Mọi mục BibTeX đều có record (``check_labels`` 3.3.4): khớp thì ``matched`` + ``rule``,
    không luật nào khớp thì ``unresolved`` (KHÔNG phải no-match, xem ml_design 3.5).
    """
    preds = predict_paper(score_paper(paper.entries, paper.candidates), rules, one_to_one)
    labels = []
    for entry, (pred, rule) in zip(paper.entries, preds):
        labels.append({
            "citation_key": entry.citation_key,
            "groundtruth_arxiv_id": pred,
            "label_status": MATCHED if pred is not None else UNRESOLVED,
            "annotator": None,
            "rule": rule,
            "note": "",
        })
    return {"arxiv_id": paper.arxiv_id, "label_source": "auto", "labels": labels}


# =============================================================================
# Bộ nhãn chuẩn để đo
# =============================================================================
Gold = dict[tuple[str, str], str | None]  # (arxiv_id, citation_key) -> ID đúng hoặc None


def silver_gold(papers: Iterable[tuple[PaperRecord, PaperScores]]) -> Gold:
    """"Nhãn bạc": suy từ ID arXiv mà chính mục BibTeX ghi trong url/journal/doi.

    - Mục ghi đúng MỘT ID nằm trong pool -> đáp án là ID đó.
    - Mục có ID nhưng không ID nào nằm trong pool -> không có đáp án trong pool (``None``).
    - Mục không ghi ID -> không thuộc bộ nhãn bạc.

    Giới hạn: chỉ phủ các mục có ghi ID (thường là trích dẫn preprint), nên dùng để đo các luật
    KHÔNG dựa vào ID; luật ``arxiv_in_url`` đo trên nhãn bạc luôn đúng theo định nghĩa.
    """
    gold: Gold = {}
    for paper, sc in papers:
        for i, key in enumerate(sc.entry_keys):
            if not sc.entry_has_id[i]:
                continue
            hits = np.flatnonzero(sc.id_match[i])
            if hits.size == 1:
                gold[(paper.arxiv_id, key)] = sc.candidate_ids[int(hits[0])]
            elif hits.size == 0:
                gold[(paper.arxiv_id, key)] = None
    return gold


def gold_from_labels(labels_dir: Path, source: str = "manual") -> Gold:
    """Đọc nhãn tay (``matched`` và ``no_match_confirmed``) từ các file ``<id>.json``."""
    gold: Gold = {}
    for path in sorted(Path(labels_dir).glob("*.json")):
        if path.name == "split.json":
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("label_source") != source:
            continue
        for rec in doc.get("labels", []):
            status = rec.get("label_status")
            if status == MATCHED:
                gold[(doc["arxiv_id"], rec["citation_key"])] = rec.get("groundtruth_arxiv_id")
            elif status == NO_MATCH:
                gold[(doc["arxiv_id"], rec["citation_key"])] = None
    return gold


# =============================================================================
# Đo precision / recall
# =============================================================================
@dataclass
class RuleReport:
    """Kết quả của một luật (hoặc chuỗi luật) trên bộ nhãn chuẩn và trên toàn bộ mục."""

    rule: str
    n_gold: int  # số mục có trong bộ nhãn chuẩn
    n_answerable: int  # trong đó có đáp án nằm trong pool
    fired: int  # số mục (thuộc bộ chuẩn) mà luật đưa ra dự đoán
    correct: int
    wrong_ref: int  # đoán một ứng viên nhưng sai ứng viên
    false_on_none: int  # đoán một ứng viên trong khi thật ra không có đáp án
    precision: float
    recall: float
    f1: float
    coverage_all: float  # tỉ lệ mục (trên TẤT CẢ mục) mà luật gán được

    @staticmethod
    def build(rule: str, pairs: Sequence[tuple[str | None, str | None]],
              fired_all: int, n_all: int) -> RuleReport:
        """``pairs``: (dự đoán, đáp án) cho các mục thuộc bộ chuẩn."""
        fired = sum(p is not None for p, _ in pairs)
        correct = sum(p is not None and p == g for p, g in pairs)
        wrong_ref = sum(p is not None and g is not None and p != g for p, g in pairs)
        false_none = sum(p is not None and g is None for p, g in pairs)
        n_ans = sum(g is not None for _, g in pairs)
        prec = correct / fired if fired else math.nan
        rec = correct / n_ans if n_ans else math.nan
        f1 = 2 * prec * rec / (prec + rec) if fired and n_ans and prec + rec else math.nan
        return RuleReport(rule, len(pairs), n_ans, fired, correct, wrong_ref, false_none,
                          prec, rec, f1, fired_all / n_all if n_all else math.nan)


def evaluate_rules(
    scored: Sequence[tuple[PaperRecord, PaperScores]],
    gold: Gold,
    rules: Sequence[Rule],
    cascades: Mapping[str, Sequence[Rule]] | None = None,
    one_to_one: bool = False,
) -> list[RuleReport]:
    """Precision/recall từng luật đứng một mình, rồi của từng chuỗi luật trong ``cascades``.

    Precision = đúng / số mục luật đưa ra dự đoán; recall = đúng / số mục có đáp án trong pool
    (cả hai chỉ tính trên mục thuộc ``gold``). ``coverage_all`` tính trên mọi mục của mọi bài.
    """
    todo: list[tuple[str, Sequence[Rule]]] = [(r.name, [r]) for r in rules]
    todo += [(f"cascade:{name}", seq) for name, seq in (cascades or {}).items()]
    n_all = sum(len(sc.entry_keys) for _, sc in scored)
    reports = []
    for name, seq in todo:
        pairs: list[tuple[str | None, str | None]] = []
        fired_all = 0
        for paper, sc in scored:
            preds = predict_paper(sc, seq, one_to_one)
            for key, (pred, _) in zip(sc.entry_keys, preds):
                fired_all += pred is not None
                k = (paper.arxiv_id, key)
                if k in gold:
                    pairs.append((pred, gold[k]))
        reports.append(RuleReport.build(name, pairs, fired_all, n_all))
    return reports


def rule_grid() -> list[Rule]:
    """Lưới luật tiêu đề để khảo sát (cộng luật ID arXiv đứng đầu)."""
    rules = [Rule(kind="arxiv_id")]
    for metric in METRICS:
        thresholds = (1.0, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80)
        for t in thresholds:
            for author in ("none", "first", "any"):
                for year in (None, 1):
                    for margin in (0.0, 0.05):
                        # partial_ratio = 1 cho mọi tiêu đề là chuỗi con: bắt buộc độ dài tối thiểu.
                        min_len = 20 if metric == "partial" else 0
                        rules.append(Rule(metric=metric, threshold=t, margin=margin,
                                          author=author, max_year_diff=year,
                                          min_title_len=min_len))
    return rules


def select_rules(reports: Sequence[RuleReport], target: float = TARGET_PRECISION,
                 min_support: int = 30) -> list[RuleReport]:
    """Các luật đạt precision >= ``target`` với ít nhất ``min_support`` dự đoán, xếp theo recall."""
    ok = [r for r in reports if not r.rule.startswith("cascade:") and r.fired >= min_support
          and not math.isnan(r.precision) and r.precision >= target]
    return sorted(ok, key=lambda r: (-r.recall, -r.precision))


# =============================================================================
# CLI
# =============================================================================
def _read_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_scored(
    raw_root: Path, processed_root: Path | None, ids: set[str] | None, limit: int | None
) -> list[tuple[PaperRecord, PaperScores]]:
    """Đọc và tính điểm cho các bài; bỏ bài không có mục BibTeX hoặc pool rỗng."""
    out = []
    dirs = iter_paper_dirs(raw_root, ids)
    for k, d in enumerate(dirs[:limit] if limit else dirs, start=1):
        paper = load_paper(d, processed_root)
        if paper.entries and paper.candidates:
            out.append((paper, score_paper(paper.entries, paper.candidates)))
        if k % 100 == 0:
            logger.info("đã đọc %d bài", k)
    logger.info("%d bài có cả mục BibTeX và pool", len(out))
    return out


def write_report(reports: Sequence[RuleReport], path: Path) -> None:
    """Ghi bảng kết quả ra CSV (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(reports[0])))
        writer.writeheader()
        for r in reports:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in asdict(r).items()})


def _rule_counts(scored: Sequence[tuple[PaperRecord, PaperScores]],
                 rules: Sequence[Rule]) -> dict[str, int]:
    """Số mục (trên mọi bài) được gán bởi từng luật của chuỗi, và số mục ``unresolved``."""
    counts = {r.name: 0 for r in rules} | {UNRESOLVED: 0}
    for _, sc in scored:
        for _, rule in predict_paper(sc, rules):
            counts[rule or UNRESOLVED] += 1
    return counts


def _cmd_evaluate(args: argparse.Namespace) -> int:
    scored = load_scored(args.raw_root, args.processed_root, _read_ids(args.ids_file), args.limit)
    if args.gold_labels:
        gold = gold_from_labels(args.gold_labels)
        gold_name = f"nhãn tay ({args.gold_labels})"
    else:
        gold = silver_gold(scored)
        gold_name = "nhãn bạc (ID arXiv ghi trong mục)"
    cascades = {"default": default_rules()}
    reports = evaluate_rules(scored, gold, rule_grid(), cascades, args.one_to_one)
    write_report(reports, args.out)
    best = select_rules(reports)
    logger.info("Bộ nhãn chuẩn: %s, %d mục (%d có đáp án)", gold_name, len(gold),
                sum(v is not None for v in gold.values()))
    for r in best[:10]:
        logger.info("  %-40s P=%.3f R=%.3f fired=%d", r.rule, r.precision, r.recall, r.fired)
    cascade = next(r for r in reports if r.rule == "cascade:default")
    logger.info("Chuỗi luật mặc định: P=%.3f R=%.3f coverage=%.3f", cascade.precision,
                cascade.recall, cascade.coverage_all)
    if not args.gold_labels:
        logger.info("Lưu ý: trên nhãn bạc, recall của chuỗi luôn = 1 (luật %s định nghĩa nhãn "
                    "bạc) và precision chỉ đại diện cho mục CÓ ghi ID; soát tay thêm các cặp "
                    "khớp mờ của mục không ghi ID trước khi đổi ngưỡng.", ARXIV_RULE)
    for name, count in _rule_counts(scored, default_rules()).items():
        logger.info("  chuỗi mặc định gán bằng %-32s %d mục", name, count)
    logger.info("Đã ghi %s", args.out)
    return 0


def _cmd_label(args: argparse.Namespace) -> int:
    rules = [Rule.from_name(n) for n in args.rules] if args.rules else default_rules()
    ids = _read_ids(args.ids_file)
    n_written = 0
    for d in iter_paper_dirs(args.raw_root, ids):
        paper = load_paper(d, args.processed_root)
        if not paper.entries:
            logger.warning("%s: không có mục BibTeX, bỏ qua", paper.arxiv_id)
            continue
        doc = label_paper(paper, rules, args.one_to_one)
        write_json(args.out_dir / f"{to_folder_name(paper.arxiv_id)}.json", doc)
        n_written += 1
    logger.info("Đã ghi %d file nhãn auto vào %s", n_written, args.out_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Gán nhãn tự động và khảo sát luật (M3).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("evaluate", "label"):
        p = sub.add_parser(name)
        p.add_argument("--raw-root", type=Path, required=True, help="thư mục bài của M1")
        p.add_argument("--processed-root", type=Path, default=None,
                       help="thư mục đầu ra M2 (chứa <yymm-id>/refs.bib); bỏ trống = trích từ TeX")
        p.add_argument("--ids-file", type=Path, default=None, help="mỗi dòng một arXiv ID chuẩn")
        p.add_argument("--one-to-one", action="store_true",
                       help="mỗi ứng viên chỉ gán cho một mục (mục ưu tiên hơn thắng)")
    ev = sub.choices["evaluate"]
    ev.add_argument("--gold-labels", type=Path, default=None,
                    help="thư mục nhãn tay; bỏ trống thì dùng nhãn bạc")
    ev.add_argument("--limit", type=int, default=None, help="chỉ đọc N bài đầu")
    ev.add_argument("--out", type=Path, default=Path("reports/auto_label_rules.csv"))
    lb = sub.choices["label"]
    lb.add_argument("--out-dir", type=Path, default=Path("data/labels"))
    lb.add_argument("--rules", nargs="*", default=None,
                    help=f"tên luật theo thứ tự ưu tiên; mặc định {list(DEFAULT_RULE_NAMES)}")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return _cmd_evaluate(args) if args.cmd == "evaluate" else _cmd_label(args)


if __name__ == "__main__":
    raise SystemExit(main())
