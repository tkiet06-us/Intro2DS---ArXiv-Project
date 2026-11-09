"""Unit tests cho ``src.m2_parsing.bibtex_extractor`` (dùng tests/mock_data)."""

from pathlib import Path

import pytest

from src.m2_parsing.bibtex_extractor import (
    extract_bibitems,
    extract_bibtex,
    format_bibtex_entry,
    parse_bib_file,
    write_bibtex,
)
from src.m2_parsing.latex_parser import resolve_paper

MOCK_DATA = Path(__file__).resolve().parents[1] / "mock_data"


def test_extracts_bibitem_from_mock_version():
    """Nguồn thật của mock dùng \\begin{thebibliography} + \\bibitem."""
    latex = resolve_paper(MOCK_DATA / "2310-12345" / "v1")
    entries = extract_bibitems(latex)

    assert len(entries) == 1
    entry = entries[0]
    assert entry == {
        "citation_key": "foo",
        "entry_type": "misc",
        "title": "Foo Bar",
        "authors": ["A. Author"],
        "year": 2020,
        "doi": None,
        "journal": None,
        "url": None,
    }


def test_parses_bib_file_from_mock():
    """Nguồn thứ hai: file .bib đi kèm do \\bibliography{refs} chỉ định."""
    entries = parse_bib_file(MOCK_DATA / "2401-00001" / "v1" / "refs.bib")

    assert [entry["citation_key"] for entry in entries] == ["bert", "noyear"]
    bert = entries[0]
    assert bert["entry_type"] == "article"
    assert bert["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"
    assert bert["authors"] == ["J. Devlin", "M. Chang"]
    assert bert["year"] == 2019
    assert bert["journal"] == "NAACL"

    # Entry thiếu năm là chuyện thường gặp: phải là None, không được bịa 0.
    assert entries[1]["year"] is None
    assert entries[1]["authors"] == ["Unknown"]


def test_extract_bibtex_reads_bibliography_command():
    """\\bibliography{refs} phải được phân giải thành file refs.bib trong thư mục version."""
    version_dir = MOCK_DATA / "2401-00001" / "v1"
    latex = resolve_paper(version_dir)

    assert extract_bibitems(latex) == []
    entries = extract_bibtex(latex, version_dir)
    assert [entry["citation_key"] for entry in entries] == ["bert", "noyear"]


def test_extract_bibtex_without_version_dir_only_reads_bibitems():
    latex = resolve_paper(MOCK_DATA / "2310-12345" / "v1")
    assert [entry["citation_key"] for entry in extract_bibtex(latex)] == ["foo"]


def test_bibitem_with_doi_url_and_journal():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{smith2021}
    J. Smith and A. Doe, \emph{A Study of Things}, Journal of Tests, 2021.
    \url{https://example.com/paper} \doi{10.1234/ABC.567}
    \end{thebibliography}
    """
    entry = extract_bibitems(latex)[0]

    assert entry["citation_key"] == "smith2021"
    assert entry["title"] == "A Study of Things"
    assert entry["authors"] == ["J. Smith", "A. Doe"]
    assert entry["year"] == 2021
    assert entry["journal"] == "Journal of Tests"
    assert entry["doi"] == "10.1234/ABC.567"
    assert entry["url"] == "https://example.com/paper"


def test_year_is_not_taken_from_url_doi_or_arxiv_id():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{x} A. Author, \emph{Title}, 2024.
    \url{https://arxiv.org/abs/2023.12345} \doi{10.2022/abc.123}
    \end{thebibliography}
    """
    assert extract_bibitems(latex)[0]["year"] == 2024


def test_ieee_bibinfo_fields_and_nested_braces():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{ieee}
    \bibinfo{author}{A. B. C.} \bibinfo{title}{A {Nested} Title}
    \bibinfo{journal}{Conference on Tests} \bibinfo{year}{2018}
    \end{thebibliography}
    """
    entry = extract_bibitems(latex)[0]

    assert entry["authors"] == ["A. B. C."]
    assert entry["title"] == "A Nested Title"
    assert entry["journal"] == "Conference on Tests"
    assert entry["year"] == 2018


def test_apa_bibitem_with_textit_venue_keeps_title_and_authors():
    """APA ``\\bibitem`` dùng ``\\textit{}`` cho venue: title/authors phải đúng."""
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{lee2019} Lee, B. (2019). Graph nets. In \textit{NeurIPS}.
    \end{thebibliography}
    """
    entry = extract_bibitems(latex)[0]

    assert entry["title"] == "Graph nets"
    assert entry["authors"] == ["Lee, B."]
    assert entry["journal"] == "In NeurIPS"
    assert entry["year"] == 2019


def test_plain_bibitem_with_initials_after_surname_splits_author_list():
    """``Họ, X., Họ, Y. Title. Venue, năm.`` không được cắt title thành 'A'."""
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{vaswani2017} Vaswani, A., Shazeer, N. Attention is all you need. NeurIPS, 2017.
    \end{thebibliography}
    """
    entry = extract_bibitems(latex)[0]

    assert entry["title"] == "Attention is all you need"
    assert entry["authors"] == ["Vaswani, A.", "Shazeer, N."]
    assert entry["journal"] == "NeurIPS"
    assert entry["year"] == 2017


def test_multiple_bibitems_are_split_at_the_right_boundary():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{first} A. One, \emph{First Title}, 2001.
    \bibitem{second} B. Two, \emph{Second Title}, 2002.
    \end{thebibliography}
    """
    entries = extract_bibitems(latex)

    assert [entry["citation_key"] for entry in entries] == ["first", "second"]
    assert entries[0]["title"] == "First Title"
    assert entries[1]["title"] == "Second Title"
    assert entries[1]["year"] == 2002


def test_author_lists_with_and_ampersand_and_semicolon():
    latex = r"""
    \begin{thebibliography}{9}
    \bibitem{a} Alice A. and Bob B.; Carol C., \emph{Title}, 2020.
    \bibitem{b} Dan D. & Eve E., \emph{Other}, 2021.
    \end{thebibliography}
    """
    entries = extract_bibitems(latex)

    assert entries[0]["authors"] == ["Alice A.", "Bob B.", "Carol C."]
    assert entries[1]["authors"] == ["Dan D.", "Eve E."]


def test_parse_bib_file_skips_string_and_comment_entries(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        '@string{jan = "January"}\n'
        "@comment{not an entry}\n"
        "@article{real,\n"
        "  title = {A Real Title},\n"
        "  author = {Real Author},\n"
        "  year = {2020},\n"
        "}\n",
        encoding="utf-8",
    )

    entries = parse_bib_file(bib)
    assert [entry["citation_key"] for entry in entries] == ["real"]
    assert entries[0]["title"] == "A Real Title"
    assert entries[0]["authors"] == ["Real Author"]
    assert entries[0]["year"] == 2020


def test_write_bibtex_round_trip(tmp_path):
    entries = [
        {
            "citation_key": "k1",
            "entry_type": "article",
            "title": "A Title",
            "authors": ["Alice", "Bob"],
            "year": 2020,
            "doi": "10.1234/xyz",
            "journal": "Journal",
            "url": "https://example.com/a",
        }
    ]
    target = tmp_path / "nested" / "refs.bib"
    write_bibtex(entries, target)

    text = target.read_text(encoding="utf-8")
    assert text.startswith("@article{k1,")
    assert "author = {Alice and Bob}" in text
    assert parse_bib_file(target) == entries


def test_format_bibtex_entry_strips_braces_from_values():
    text = format_bibtex_entry(
        {
            "citation_key": "x",
            "entry_type": "misc",
            "title": "A {weird} title",
            "authors": [],
            "year": None,
            "doi": None,
            "journal": None,
            "url": None,
        }
    )

    assert "@misc{x," in text
    assert "title = {A weird title}" in text
    assert "year" not in text


@pytest.mark.parametrize(
    ("raw_authors", "expected"),
    [
        ("J. Devlin and M. Chang", ["J. Devlin", "M. Chang"]),
        ("Smith, J., Doe, A.", ["Smith, J.", "Doe, A."]),
        ("A. One; B. Two", ["A. One", "B. Two"]),
    ],
)
def test_author_parsing_variants(raw_authors, expected):
    latex = (
        "\\begin{thebibliography}{9}\n"
        f"\\bibitem{{k}} {raw_authors}, \\emph{{Title}}, 2020.\n"
        "\\end{thebibliography}\n"
    )
    assert extract_bibitems(latex)[0]["authors"] == expected
