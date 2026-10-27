"""Kiểm tra tiền xử lý và đặc trưng của P2-06 (chỉ dùng dữ liệu tự dựng, không gọi mạng)."""

import numpy as np
import pytest

from src.m3_matching.features import (
    FEATURE_NAMES,
    author_key,
    build_feature_matrix,
    build_features,
    char_ngrams,
    extract_arxiv_ids,
    jaccard,
    parse_year,
    preprocess,
    tfidf_cosine_matrix,
    title_levenshtein,
)
from src.schemas import Bibtex, Reference

PAPER_ID = "2310.12345"


def make_entry(**overrides) -> Bibtex:
    """Mục BibTeX hợp lệ tối thiểu, ghi đè các trường cần thử."""
    base = {"arxiv_id": PAPER_ID, "citation_key": "key", "entry_type": "article"}
    return Bibtex(**{**base, **overrides})


def make_candidate(**overrides) -> Reference:
    """Reference hợp lệ tối thiểu, ghi đè các trường cần thử."""
    base = {"arxiv_id": PAPER_ID, "ref_arxiv_id": "1706.03762"}
    return Reference(**{**base, **overrides})


def feature(vector: list[float], name: str) -> float:
    return vector[FEATURE_NAMES.index(name)]


# ---- preprocess ----


def test_preprocess_removes_latex_commands_and_accents():
    raw = r"\textbf{Attention} Is All You Need! M{\"u}ller"

    assert preprocess(raw) == "attention is all you need muller"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"Mart\'{\i}nez", "martinez"),
        (r"Erd\H{o}s", "erdos"),
        (r"Fran\c{c}ois", "francois"),
        ("Nguyễn Đình", "nguyen dinh"),
        ("Müller-Łukasiewicz", "muller lukasiewicz"),
    ],
)
def test_preprocess_folds_accents(raw, expected):
    assert preprocess(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"see \cite{abc}", "see abc"),
        (r"\url{http://x.org}", "http x org"),
        (r"\underline{Deep} nets", "deep nets"),
        (r"\vspace{2cm} A", "2cm a"),
        (r"\citep{abc}", "abc"),
    ],
)
def test_preprocess_keeps_content_of_commands_that_look_like_accents(raw, expected):
    # Các lệnh bắt đầu bằng \c, \u, \v, \r không được bị hiểu nhầm là dấu LaTeX.
    assert preprocess(raw) == expected


def test_preprocess_keeps_inline_math_content():
    # S2 trả tiêu đề dạng văn bản thường ("k-means"), nên phía BibTeX cũng phải giữ chữ "k".
    assert preprocess("Is $k$-means hard?") == "is k means hard"


def test_preprocess_does_not_swallow_cite_command_content():
    # "\c" là lệnh cedilla nhưng "\cite{a}" thì không được hiểu nhầm.
    assert preprocess(r"see \cite{abc}") == "see abc"


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_preprocess_handles_empty_input(empty):
    assert preprocess(empty) == ""


def test_preprocess_can_remove_stopwords():
    assert preprocess("The Art of War", remove_stopwords=True) == "art war"
    assert preprocess("The Art of War") == "the art of war"


# ---- tác giả và năm ----


@pytest.mark.parametrize("name", ["Kaiming He", "He, K.", "K. He", "He, Kaiming"])
def test_author_key_same_person_different_formats(name):
    assert author_key(name) == ("he", "k")


def test_author_key_keeps_last_surname_token():
    assert author_key("van der Waals, J.") == author_key("J. van der Waals") == ("waals", "j")


@pytest.mark.parametrize("name", ["et al.", "others", "", "  "])
def test_author_key_ignores_non_names(name):
    assert author_key(name) is None


def test_parse_year():
    assert parse_year("2017-06-12") == 2017
    assert parse_year(None) is None
    assert parse_year("không rõ") is None


# ---- độ giống (khớp với bài tập BT1-BT3) ----


def test_levenshtein_kitten_sitting():
    assert title_levenshtein("kitten", "sitting") == pytest.approx(1 - 3 / 7)


def test_char_bigram_jaccard_night_nacht():
    assert jaccard(char_ngrams("night", 2), char_ngrams("nacht", 2)) == pytest.approx(1 / 7)


def test_jaccard_empty_set_is_zero():
    assert jaccard(set(), {"a"}) == 0.0


# ---- build_features ----


def test_build_features_has_one_value_per_feature_name():
    vector = build_features(make_entry(), make_candidate())

    assert len(vector) == len(FEATURE_NAMES)


def test_identical_pair_scores_high_everywhere():
    entry = make_entry(title="Attention Is All You Need", authors=["A. Vaswani"], year=2017)
    candidate = make_candidate(
        ref_title="Attention is all you need",
        ref_authors=["Ashish Vaswani", "Noam Shazeer"],
        ref_submission_date="2017-06-12",
    )

    vector = build_features(entry, candidate)

    assert feature(vector, "title_lev") == 1.0
    assert feature(vector, "title_word_jaccard") == 1.0
    assert feature(vector, "title_token_set") == 1.0
    assert feature(vector, "first_author_match") == 1.0
    assert feature(vector, "year_match") == 1.0
    assert feature(vector, "year_diff") == 0.0
    assert [feature(vector, f) for f in ("has_title", "has_authors", "has_year")] == [1.0] * 3


def test_missing_data_gives_neutral_values_and_flags():
    vector = build_features(make_entry(), make_candidate())

    assert [feature(vector, f) for f in ("has_title", "has_authors", "has_year")] == [0.0] * 3
    assert feature(vector, "title_lev") == 0.0
    assert feature(vector, "first_author_match") == 0.0
    assert feature(vector, "year_diff") == 0.5


def test_year_diff_is_scaled_and_capped():
    candidate = make_candidate(ref_submission_date="2017-06-12")

    close = build_features(make_entry(year=2019), candidate)
    far = build_features(make_entry(year=2030), candidate)

    assert feature(close, "year_match") == 0.0
    assert feature(close, "year_diff") == pytest.approx(0.4)
    assert feature(far, "year_diff") == 1.0


def test_arxiv_id_found_in_journal_field():
    entry = make_entry(journal="arXiv preprint arXiv:1706.03762")

    assert feature(build_features(entry, make_candidate()), "arxiv_id_in_fields") == 1.0


def test_arxiv_id_in_url_with_version_suffix():
    entry = make_entry(url="https://arxiv.org/abs/1706.03762v5")

    assert feature(build_features(entry, make_candidate()), "arxiv_id_in_fields") == 1.0


def test_arxiv_id_does_not_match_inside_longer_id():
    entry = make_entry(journal="arXiv:2310.12345")
    candidate = make_candidate(ref_arxiv_id="2310.1234")

    assert feature(build_features(entry, candidate), "arxiv_id_in_fields") == 0.0


def test_different_papers_score_lower_than_same_paper():
    entry = make_entry(title="Attention Is All You Need", authors=["A. Vaswani"], year=2017)
    same = make_candidate(
        ref_title="Attention Is All You Need",
        ref_authors=["Ashish Vaswani"],
        ref_submission_date="2017-06-12",
    )
    other = make_candidate(
        ref_arxiv_id="1810.04805",
        ref_title="BERT: Pre-training of Deep Bidirectional Transformers",
        ref_authors=["Jacob Devlin"],
        ref_submission_date="2018-10-11",
    )

    same_vector, other_vector = build_features(entry, same), build_features(entry, other)

    for name in ("title_lev", "title_word_jaccard", "first_author_match", "year_match"):
        assert feature(same_vector, name) > feature(other_vector, name), name


# ---- build_feature_matrix và TF-IDF ----


def test_feature_matrix_shape_and_true_candidate_has_highest_tfidf():
    entries = [
        make_entry(citation_key="a", title="Attention Is All You Need"),
        make_entry(citation_key="b", title="Deep Residual Learning for Image Recognition"),
    ]
    candidates = [
        make_candidate(ref_arxiv_id="1810.04805", ref_title="BERT Pre-training Transformers"),
        make_candidate(ref_arxiv_id="1706.03762", ref_title="Attention is all you need"),
        make_candidate(
            ref_arxiv_id="1512.03385", ref_title="Deep Residual Learning for Image Recognition"
        ),
    ]

    tensor = build_feature_matrix(entries, candidates)

    column = FEATURE_NAMES.index("title_tfidf_cos")
    assert tensor.shape == (2, 3, len(FEATURE_NAMES))
    assert [int(np.argmax(tensor[i, :, column])) for i in range(2)] == [1, 2]


def test_feature_matrix_matches_single_pair_function():
    entry = make_entry(title="Attention Is All You Need", year=2017)
    candidate = make_candidate(ref_title="Attention is all you need")

    tensor = build_feature_matrix([entry], [candidate])
    single = build_features(
        entry, candidate, tfidf_cosine=tensor[0, 0, FEATURE_NAMES.index("title_tfidf_cos")]
    )

    assert tensor[0, 0].tolist() == pytest.approx(single)


@pytest.mark.parametrize(("m", "n"), [(0, 3), (3, 0), (0, 0)])
def test_feature_matrix_handles_empty_sides(m, n):
    entries = [make_entry()] * m
    candidates = [make_candidate()] * n

    assert build_feature_matrix(entries, candidates).shape == (m, n, len(FEATURE_NAMES))


def test_tfidf_returns_zeros_when_no_candidate_has_a_title():
    result = tfidf_cosine_matrix(["attention"], ["", ""])

    assert result.shape == (1, 2)
    assert not result.any()


# ---- extract_arxiv_ids ----


def test_extract_arxiv_ids_strips_version_and_keeps_old_style_ids():
    text = "arXiv:solv-int/9901001v2, 2310.12345v2 and math.GT/0309136"

    assert extract_arxiv_ids(text) == {"solv-int/9901001", "2310.12345", "math.gt/0309136"}


def test_extract_arxiv_ids_finds_every_id_not_only_the_first():
    assert extract_arxiv_ids("see 1810.04805 and 1706.03762") == {"1810.04805", "1706.03762"}


def test_extract_arxiv_ids_ignores_doi_and_longer_numbers():
    assert extract_arxiv_ids("doi 10.1145/3292500.3330701") == set()
    assert extract_arxiv_ids("2310.123456") == set()


def test_extract_arxiv_ids_handles_empty_input():
    assert extract_arxiv_ids(None) == set()
    assert extract_arxiv_ids("") == set()


# ---- raw_text ----


def test_raw_text_supplies_arxiv_id_when_fields_are_empty():
    raw = r"A. Vaswani et al. Attention is all you need. arXiv:1706.03762v5, 2017."

    vector = build_features(make_entry(), make_candidate(), raw_text=raw)

    assert feature(vector, "arxiv_id_in_fields") == 1.0


def test_raw_text_matches_the_second_id_when_several_are_present():
    raw = "compare with arXiv:1810.04805 and arXiv:1706.03762"

    assert (
        feature(build_features(make_entry(), make_candidate(), raw_text=raw), "arxiv_id_in_fields")
        == 1.0
    )


def test_raw_text_supplies_year_when_entry_has_none():
    candidate = make_candidate(ref_submission_date="2017-06-12")

    vector = build_features(make_entry(), candidate, raw_text="Foo. Bar. 2017.")

    assert feature(vector, "has_year") == 1.0
    assert feature(vector, "year_match") == 1.0


def test_feature_matrix_uses_raw_texts_and_checks_their_length():
    entries = [make_entry(citation_key="a"), make_entry(citation_key="b")]
    candidates = [make_candidate()]

    tensor = build_feature_matrix(entries, candidates, raw_texts=["arXiv:1706.03762", None])

    column = FEATURE_NAMES.index("arxiv_id_in_fields")
    assert tensor[:, 0, column].tolist() == [1.0, 0.0]
    with pytest.raises(ValueError):
        build_feature_matrix(entries, candidates, raw_texts=["only one"])


# ---- tác giả dạng list (lỗi cũ: nối list thành một chuỗi làm mất ranh giới tác giả) ----


def test_author_lists_in_different_formats_match_on_first_author_and_surnames():
    entry = make_entry(authors=["Vaswani, A.", "Shazeer, N."])
    candidate = make_candidate(ref_authors=["Ashish Vaswani", "Noam Shazeer"])

    vector = build_features(entry, candidate)

    assert feature(vector, "first_author_match") == 1.0
    assert feature(vector, "author_surname_jaccard") == 1.0


# ---- feature mới: year_signed_diff, title_containment ----


def test_year_signed_diff_keeps_direction_and_is_capped():
    candidate = make_candidate(ref_submission_date="2017-06-12")

    preprint_earlier = build_features(make_entry(year=2019), candidate)
    preprint_later = build_features(make_entry(year=2015), candidate)
    far = build_features(make_entry(year=2030), candidate)

    assert feature(preprint_earlier, "year_signed_diff") == pytest.approx(-0.4)
    assert feature(preprint_later, "year_signed_diff") == pytest.approx(0.4)
    assert feature(far, "year_signed_diff") == -1.0


def test_year_signed_diff_is_zero_when_year_is_missing():
    assert feature(build_features(make_entry(), make_candidate()), "year_signed_diff") == 0.0


def test_title_containment_is_high_for_truncated_title():
    entry = make_entry(title="Attention Is All You Need")
    full = make_candidate(ref_title="Attention Is All You Need: A Study of Transformers")
    unrelated = make_candidate(ref_title="Gradient Boosting for Tabular Data")

    assert feature(build_features(entry, full), "title_containment") == 1.0
    assert feature(build_features(entry, unrelated), "title_containment") < 0.7


def test_identical_short_titles_still_have_full_trigram_overlap():
    vector = build_features(make_entry(title="AI"), make_candidate(ref_title="AI"))

    assert feature(vector, "title_char3_jaccard") == 1.0