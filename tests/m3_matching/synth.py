"""Dữ liệu giả lập cho M3: bài manual/auto có entry nhiễu, no-match, false-null và pool rỗng.

Chỉ dùng cho test. Không phản ánh chất lượng thật; mục đích là kiểm tra pipeline chạy đúng.
"""
from __future__ import annotations

import difflib
import json
import random
from pathlib import Path

import pandas as pd

SURNAMES = ["vaswani", "devlin", "brown", "he", "kingma", "bahdanau", "lecun", "hinton", "graves",
            "sutskever", "schmidt", "nguyen", "tran", "smith", "garcia", "mueller", "kim", "wang"]
GIVEN = ["Ashish", "Jacob", "Tom", "Kaiming", "Diederik", "Dzmitry", "Yann", "Geoffrey", "Alex"]
VOCAB = ("attention transformer neural network learning deep bidirectional language model "
         "pretraining optimization gradient stochastic adaptive residual convolution image "
         "recognition sequence translation generative adversarial variational inference graph "
         "embedding sparse efficient scalable robust fine tuning reinforcement policy diffusion "
         "contrastive representation self supervised few shot prompt reasoning planning memory "
         "kernel bayesian regularization normalization dropout batch layer").split()


def _title(rng: random.Random) -> str:
    return " ".join(rng.sample(VOCAB, rng.randint(4, 8))).capitalize()


def _authors(rng: random.Random) -> list[str]:
    return [f"{rng.choice(GIVEN)} {rng.choice(SURNAMES).capitalize()}" for _ in range(rng.randint(1, 4))]


def _perturb_title(rng: random.Random, title: str) -> str:
    words = title.split()
    kind = rng.random()
    if kind < 0.35:
        return title.lower()
    if kind < 0.55 and len(words) > 4:
        return " ".join(words[:-rng.randint(1, 2)])           # tiêu đề bị cắt
    if kind < 0.75:
        return "{" + words[0] + "} " + " ".join(words[1:]).lower()   # ngoặc LaTeX
    return title


def make_paper(rng: random.Random, arxiv_id: str, source: str, n_refs: int, n_extra: int,
               n_labeled: int | None = None):
    """Trả về (bibtex_rows, reference_rows, label_doc)."""
    ref_ids = sorted({f"{rng.randint(1501, 2310)}.{rng.randint(1000, 19999):05d}" for _ in range(n_refs)})
    refs = []
    for rid in ref_ids:
        refs.append(dict(arxiv_id=arxiv_id, ref_arxiv_id=rid, ref_title=_title(rng),
                         ref_authors=_authors(rng), ref_submission_date=f"{2000 + int(rid[:2])}-0{rng.randint(1, 9)}-15",
                         ref_s2_id=f"s2-{rid}"))
    bib, records = [], []
    n_entries = len(refs) + n_extra
    for i in range(n_entries):
        key = f"key{i:03d}"
        matched = i < len(refs) and rng.random() < 0.75
        if matched:
            r = refs[i]
            title = None if rng.random() < 0.05 else _perturb_title(rng, r["ref_title"])
            authors = [a.split()[-1] + ", " + a.split()[0][0] + "." for a in r["ref_authors"]][: rng.randint(1, 3)]
            year = None if rng.random() < 0.1 else int(r["ref_submission_date"][:4]) + rng.choice([0, 0, 0, 1, 2])
            url = f"https://arxiv.org/abs/{r['ref_arxiv_id']}" if rng.random() < 0.15 else None
            gt = r["ref_arxiv_id"]
        else:
            title, authors, year, url, gt = _title(rng), _authors(rng), rng.randint(1990, 2023), None, None
        bib.append(dict(arxiv_id=arxiv_id, citation_key=key, entry_type="article", title=title,
                        authors=authors, year=year, doi=None, journal=None, url=url))
        if source == "manual":
            records.append(dict(citation_key=key, groundtruth_arxiv_id=gt,
                                label_status="matched" if gt else "no_match_confirmed",
                                annotator="Kiet", rule=None, note=""))
        else:
            rule = None
            if gt and url:
                rule = "arxiv_in_url"
            elif gt and title and difflib.SequenceMatcher(
                    None, title.lower().strip("{}"), refs[i]["ref_title"].lower()).ratio() >= 0.92:
                rule = "title_sim>=0.92"
            records.append(dict(citation_key=key, groundtruth_arxiv_id=gt if rule else None,
                                label_status="matched" if rule else "unresolved",
                                annotator=None, rule=rule, note=""))
    if source == "manual":
        keep = n_labeled or 22
        records = records[:max(keep, 20)]
    return bib, refs, dict(arxiv_id=arxiv_id, label_source=source, labels=records)


def make_dataset(members: int = 2, n_train_manual: int = 6, n_train_auto: int = 6, seed: int = 0,
                 empty_pool_in_test: bool = False):
    """Trả về (bibtex_df, references_df, split, label_files).

    ``label_files[arxiv_id] = {"label_source", "records"}`` đã là LabelRecord (dùng cho load_papers).
    """
    from src.m3_matching.train import LabelRecord

    rng = random.Random(seed)
    plan = ([("train", "manual")] * n_train_manual + [("train", "auto")] * n_train_auto
            + [("valid", "manual")] * members + [("valid", "auto")] * members
            + [("test", "manual")] * members + [("test", "auto")] * members)
    bib_rows, ref_rows, split, files = [], [], {}, {}
    empty_done = False
    for idx, (part, src) in enumerate(plan):
        aid = f"hep-th/{9900000 + idx}" if idx % 7 == 3 else f"2301.{10000 + idx:05d}"  # có cả ID cũ
        n_refs = rng.randint(18, 26)
        b, r, doc = make_paper(rng, aid, src, n_refs, rng.randint(2, 6))
        if empty_pool_in_test and part == "test" and src == "auto" and not empty_done:
            r, empty_done = [], True
            doc["labels"] = [dict(x, groundtruth_arxiv_id=None, label_status="unresolved", rule=None)
                             for x in doc["labels"]]
        bib_rows += b
        ref_rows += r
        split[aid] = dict(partition=part, label_source=src)
        files[aid] = dict(label_source=src, records=[
            LabelRecord(x["citation_key"], x["groundtruth_arxiv_id"], x["label_status"],
                        x["rule"], x["annotator"]) for x in doc["labels"]])
        files[aid]["_doc"] = doc
    refs_df = pd.DataFrame(ref_rows, columns=["arxiv_id", "ref_arxiv_id", "ref_title", "ref_authors",
                                              "ref_submission_date", "ref_s2_id"])
    return pd.DataFrame(bib_rows), refs_df, split, files


def write_label_dir(path: Path, split, files) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "split.json").write_text(json.dumps(split), encoding="utf-8")
    for aid, f in files.items():
        (path / (aid.replace("/", "_") + ".json")).write_text(json.dumps(f["_doc"]), encoding="utf-8")