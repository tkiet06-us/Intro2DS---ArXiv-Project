"""Unit tests cho ``src.m2_parsing.dedup`` (dùng tests/mock_data)."""

import json
from pathlib import Path

import pytest

from src.m2_parsing.dedup import dedup_bibtex, dedup_fulltext, rewrite_citations

MOCK_DATA = Path(__file__).resolve().parents[1] / "mock_data"


# ---------------------------------------------------------------------------
# Dedup BibTeX
# ---------------------------------------------------------------------------


def _entry(**overrides):
    entry = {
        "citation_key": "cite",
        "entry_type": "misc",
        "title": None,
        "authors": [],
        "year": None,
        "doi": None,
        "journal": None,
        "url": None,
    }
    entry.update(overrides)
    return entry


def test_dedup_bibtex_merges_same_doi_and_unions_fields():
    entries = [
        _entry(
            citation_key="a",
            title="Paper",
            authors=["Alice"],
            year=2020,
            doi="https://doi.org/10.1234/XYZ",
        ),
        _entry(
            citation_key="b",
            entry_type="article",
            title="Paper: A Longer Study",
            authors=["Bob"],
            year=2020,
            doi="10.1234/xyz",
            journal="Journal",
            url="https://example.com",
        ),
    ]

    deduplicated, citation_key_map = dedup_bibtex(entries)

    assert len(deduplicated) == 1
    merged = deduplicated[0]
    assert merged["citation_key"] == "a"
    assert merged["entry_type"] == "article"
    assert merged["title"] == "Paper: A Longer Study"
    assert merged["authors"] == ["Alice", "Bob"]
    assert merged["year"] == 2020
    assert merged["doi"] == "10.1234/XYZ"
    assert merged["journal"] == "Journal"
    assert merged["url"] == "https://example.com"
    assert citation_key_map == {"b": "a"}


def test_dedup_bibtex_title_match_requires_compatible_authors():
    entries = [
        _entry(citation_key="first", title="Deep Learning", authors=["Yann LeCun"], year=2015),
        _entry(
            citation_key="second",
            title="Deep Learning",
            authors=["Yann LeCun"],
            year=2015,
            journal="Nature",
        ),
        _entry(citation_key="different", title="Deep Learning", authors=["Someone Else"], year=2015),
    ]

    deduplicated, _ = dedup_bibtex(entries)

    assert [entry["citation_key"] for entry in deduplicated] == ["first", "different"]
    assert deduplicated[0]["journal"] == "Nature"


def test_dedup_bibtex_fills_missing_year_and_authors():
    entries = [
        _entry(citation_key="a", title="Same Work", authors=[], year=None),
        _entry(citation_key="b", title="Same Work", authors=["A. Author"], year=2019),
    ]

    deduplicated, _ = dedup_bibtex(entries)

    assert len(deduplicated) == 1
    assert deduplicated[0]["authors"] == ["A. Author"]
    assert deduplicated[0]["year"] == 2019


def test_dedup_bibtex_renames_reused_key_for_different_works():
    entries = [
        _entry(citation_key="same", title="Work A", authors=["Alice"]),
        _entry(citation_key="same", title="Work B", authors=["Bob"]),
    ]

    deduplicated, citation_key_map = dedup_bibtex(entries)

    assert [entry["citation_key"] for entry in deduplicated] == ["same", "same_2"]
    # Không thể ánh xạ "same" toàn cục vì nó thuộc hai công trình; map theo lần xuất hiện.
    assert "same" not in citation_key_map
    assert citation_key_map["same#duplicate-2"] == "same_2"


def test_dedup_bibtex_does_not_mutate_input():
    entries = [_entry(citation_key="a", title="T", authors=["Alice"])]
    original = [dict(entry) for entry in entries]

    dedup_bibtex(entries)

    assert entries == original


def test_dedup_bibtex_empty_input():
    assert dedup_bibtex([]) == ([], {})


# ---------------------------------------------------------------------------
# Sửa lệnh \cite
# ---------------------------------------------------------------------------


def test_rewrite_citations_handles_multiple_commands_and_options():
    latex = (
        r"See \cite{a,b} and \citep[see][p.~2]{c}. "
        r"Also \citeauthor{a} and \nocite{*}."
    )

    rewritten = rewrite_citations(latex, {"a": "x", "b": "y", "c": "z"})

    assert rewritten == (
        r"See \cite{x,y} and \citep[see][p.~2]{z}. "
        r"Also \citeauthor{x} and \nocite{*}."
    )


def test_rewrite_citations_preserves_spacing_and_unknown_keys():
    assert rewrite_citations(r"\cite{a, b, unknown}", {"a": "x", "b": "y"}) == (
        r"\cite{x, y, unknown}"
    )


def test_rewrite_citations_empty_map_is_noop():
    latex = r"\cite{a}"
    assert rewrite_citations(latex, {}) == latex


# ---------------------------------------------------------------------------
# Full-text dedup
# ---------------------------------------------------------------------------


def _two_version_hierarchy():
    return {
        "elements": {
            "pub#root1": "Version 1 root",
            "pub#v1-h1": "Introduction",
            "pub#v1-s1": "Shared sentence.",
            "pub#v1-s2": "Only in version one.",
            "pub#root2": "Version 2 root",
            "pub#v2-h1": "Introduction",
            "pub#v2-s1": "Shared sentence.",
            "pub#v2-s2": "Only in version two.",
        },
        "hierarchy": {
            "1": {
                "pub#v1-h1": "pub#root1",
                "pub#v1-s1": "pub#v1-h1",
                "pub#v1-s2": "pub#v1-h1",
            },
            "2": {
                "pub#v2-h1": "pub#root2",
                "pub#v2-s1": "pub#v2-h1",
                "pub#v2-s2": "pub#v2-h1",
            },
        },
    }


def test_dedup_fulltext_remaps_ids_and_keeps_version_edges():
    result = dedup_fulltext(_two_version_hierarchy())

    assert result["elements"] == {
        "pub#root1": "Version 1 root",
        "pub#h1": "Introduction",
        "pub#s1": "Shared sentence.",
        "pub#s2": "Only in version one.",
        "pub#root2": "Version 2 root",
        "pub#s2-2": "Only in version two.",
    }
    assert result["hierarchy"] == {
        "1": {
            "pub#h1": "pub#root1",
            "pub#s1": "pub#h1",
            "pub#s2": "pub#h1",
        },
        "2": {
            "pub#h1": "pub#root2",
            "pub#s1": "pub#h1",
            "pub#s2-2": "pub#h1",
        },
    }
    # Không còn id mang tiền tố version sau dedup.
    assert not any("#v1-" in element_id or "#v2-" in element_id for element_id in result["elements"])


def test_dedup_fulltext_is_idempotent():
    once = dedup_fulltext(_two_version_hierarchy())
    twice = dedup_fulltext(once)
    assert twice == once


def test_dedup_fulltext_mock_hierarchy_is_already_canonical():
    data = json.loads((MOCK_DATA / "2310-12345" / "hierarchy.json").read_text(encoding="utf-8"))
    assert dedup_fulltext(data) == data


def test_dedup_fulltext_keeps_repeated_content_distinct_within_one_version():
    data = {
        "elements": {
            "pub#root1": "Version 1 root",
            "pub#v1-s5": "Repeated.",
            "pub#v1-s6": "Repeated.",
            "pub#root2": "Version 2 root",
            "pub#v2-s5": "Repeated.",
            "pub#v2-s6": "Repeated.",
        },
        "hierarchy": {
            "1": {"pub#v1-s5": "pub#root1", "pub#v1-s6": "pub#root1"},
            "2": {"pub#v2-s5": "pub#root2", "pub#v2-s6": "pub#root2"},
        },
    }

    result = dedup_fulltext(data)

    # Lần xuất hiện thứ nhất và thứ hai vẫn là hai element khác nhau, nhưng dùng
    # chung id giữa hai version (s5 và s6).
    assert set(result["elements"]) == {"pub#root1", "pub#s5", "pub#s6", "pub#root2"}
    assert result["hierarchy"]["1"] == {"pub#s5": "pub#root1", "pub#s6": "pub#root1"}
    assert result["hierarchy"]["2"] == {"pub#s5": "pub#root2", "pub#s6": "pub#root2"}


def test_dedup_fulltext_rewrites_citations_before_comparing():
    data = {
        "elements": {
            "pub#root1": "Version 1 root",
            "pub#v1-s1": r"See \cite{old}.",
            "pub#root2": "Version 2 root",
            "pub#v2-s1": r"See \cite{new}.",
        },
        "hierarchy": {
            "1": {"pub#v1-s1": "pub#root1"},
            "2": {"pub#v2-s1": "pub#root2"},
        },
    }

    result = dedup_fulltext(data, {"old": "new"})

    assert result["elements"]["pub#s1"] == r"See \cite{new}."
    assert len(result["elements"]) == 3  # root1, root2, s1


def test_dedup_fulltext_compares_content_after_cleanup():
    """Hai bản cùng nội dung nhưng một bản còn \\centering/[ht] vẫn phải gộp."""
    figure = "\\begin{figure}\n\\caption{Plot.}\n\\end{figure}"
    data = {
        "elements": {
            "pub#root1": "Version 1 root",
            "pub#v1-fig1": "\\begin{figure}[ht]\n\\centering\n\\caption{Plot.}\n\\end{figure}",
            "pub#root2": "Version 2 root",
            "pub#v2-fig1": figure,
        },
        "hierarchy": {
            "1": {"pub#v1-fig1": "pub#root1"},
            "2": {"pub#v2-fig1": "pub#root2"},
        },
    }

    result = dedup_fulltext(data)

    assert set(result["elements"]) == {"pub#root1", "pub#root2", "pub#fig1"}
    assert result["hierarchy"] == {
        "1": {"pub#fig1": "pub#root1"},
        "2": {"pub#fig1": "pub#root2"},
    }


def test_dedup_fulltext_falls_back_to_text_list():
    assert dedup_fulltext(["Same text.", "Same text.", "Other text."]) == [
        "Same text.",
        "Other text.",
    ]


def test_dedup_fulltext_rejects_malformed_input():
    with pytest.raises(ValueError):
        dedup_fulltext({"elements": []})
