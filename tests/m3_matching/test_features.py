"""Kiểm tra tiền xử lý và đặc trưng của P2-06 (chỉ dùng dữ liệu tự dựng, không gọi mạng)."""

import math

import numpy as np
import pytest

from src.m3_matching.features import (
    COMPOSITE_WEIGHTS,
    FEATURE_NAMES,
    MISSING_FILL_VALUES,
    _dense_percentile_rank,
    _margin_by_row,
    _normalize_arxiv_id,
    author_key,
    build_feature_matrix,
    build_features,
    char_ngrams,
    composite_score_matrix,
    extract_arxiv_ids,
    fill_missing_features,
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


def column(tensor: np.ndarray, name: str) -> np.ndarray:
    """Lấy một feature (m, n) từ tensor (m, n, F)."""
    return tensor[..., FEATURE_NAMES.index(name)]


def is_nan(value: float) -> bool:
    return math.isnan(value)


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

def test_author_key_strips_latex_braces():
    assert author_key(r'M{\"u}ller, Hans') == ("muller", "h")


def test_author_key_strips_latex_braces_with_nested_closing_brace():
    # Erd{\H{o}}s -> ascii_fold ra "erd{o}s" (còn cả "{" và "}" sau khi fold).
    assert author_key(r'Erd{\H{o}}s, P.') == ("erdos", "p")


def test_author_key_handles_accents_without_braces():
    # Trường hợp \c{c} -> ascii_fold đã xử lý gọn, không còn ngoặc nào.
    assert author_key(r'Fran\c{c}ois, M.') == ("francois", "m")


@pytest.mark.parametrize("name", ["et al.", "others", "", "  "])
def test_author_key_ignores_non_names(name):
    assert author_key(name) is None


def test_parse_year():
    assert parse_year("2017-06-12") == 2017
    assert parse_year(None) is None
    assert parse_year("không rõ") is None

@pytest.mark.parametrize("raw", ["arXiv:1806.01234", "arXiv:2012.01234v2"])
def test_parse_year_ignores_arxiv_id(raw):
    assert parse_year(raw) is None


def test_parse_year_still_finds_year_next_to_arxiv_id():
    assert parse_year("Foo. arXiv:1806.01234, 2018.") == 2018

@pytest.mark.parametrize(
    "raw",
    [
        "arXiv:1806.01234",           # YYMM của 2018 trùng 18xx
        "arXiv:1903.04567v2",         # 2019 + version suffix
        "arXiv:2012.01234",           # 2020 trùng 20xx
        "https://arxiv.org/abs/1806.01234",
        "hep-th/9901001",             # style cũ, không match YEAR_PATTERN nhưng vẫn phải None
    ],
)
def test_parse_year_ignores_arxiv_id_extended(raw):
    assert parse_year(raw) is None

def test_parse_year_still_finds_real_year_next_to_arxiv_id():
    # Nếu bibitem có CẢ arXiv ID lẫn năm thật, phải trả năm thật — fix không được
    # xóa luôn năm.
    assert parse_year("A. Vaswani et al. arXiv:1706.03762. 2017.") == 2017
    assert parse_year("NeurIPS 2018, arXiv:1806.01234") == 2018


def test_parse_year_does_not_treat_generic_4_digit_numbers_as_arxiv_id():
    # "pp. 1806" (số trang) không phải arXiv ID -> vẫn giữ làm ứng viên năm,
    # dù đây là case hiếm và kết quả có thể sai — nhưng fix không được âm thầm
    # coi "1806" đứng riêng là arXiv ID.
    assert parse_year("pp. 1806") == 1806

@pytest.mark.parametrize("empty", [None, ""])
def test_parse_year_handles_empty_input(empty):
    assert parse_year(empty) is None

# ---- độ giống ----


def test_levenshtein_kitten_sitting():
    assert title_levenshtein("kitten", "sitting") == pytest.approx(1 - 3 / 7)


def test_char_bigram_jaccard_night_nacht():
    assert jaccard(char_ngrams("night", 2), char_ngrams("nacht", 2)) == pytest.approx(1 / 7)


def test_jaccard_empty_set_is_nan_not_zero():
    # Tập rỗng nghĩa là thiếu dữ liệu, không phải "không giống nhau".
    assert is_nan(jaccard(set(), {"a"}))
    assert is_nan(jaccard({"a"}, set()))
    assert jaccard({"a"}, {"b"}) == 0.0  # có dữ liệu nhưng không giống: 0.0 thật


def test_title_levenshtein_is_nan_when_a_title_is_empty():
    assert is_nan(title_levenshtein("", "abc"))
    assert is_nan(title_levenshtein("abc", ""))
    # rapidfuzz trả 1.0 cho hai chuỗi rỗng; ta không được để giá trị "giống hệt" giả lọt ra.
    assert is_nan(title_levenshtein("", ""))


# ---- build_features ----

def test_arxiv_id_in_raw_text_does_not_create_fake_year():
    # entry.year = None, raw_text chỉ có arXiv ID 2018 -> KHÔNG được tạo năm giả.
    raw = r"A. Vaswani et al. arXiv:1806.01234."
    vector = build_features(make_entry(), make_candidate(), raw_text=raw)

    assert feature(vector, "has_year") == 0.0
    assert is_nan(feature(vector, "year_match"))
    assert is_nan(feature(vector, "year_diff"))

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

    assert feature(vector, "title_lev_sim") == 1.0
    assert feature(vector, "title_word_jaccard") == 1.0
    assert feature(vector, "title_token_set") == 1.0
    assert feature(vector, "first_author_match") == 1.0
    assert feature(vector, "year_match") == 1.0
    assert feature(vector, "year_diff") == 0.0
    assert [feature(vector, f) for f in ("has_title", "has_authors", "has_year")] == [1.0] * 3


def test_missing_data_gives_nan_before_fill():
    # Tầng 1: build_features trả NaN thô cho feature thiếu đầu vào (để composite_score
    # biết bỏ tín hiệu thiếu). Cờ has_* vẫn 0/1.
    vector = build_features(make_entry(), make_candidate())

    assert [feature(vector, f) for f in ("has_title", "has_authors", "has_year")] == [0.0] * 3
    for name in (
        "title_lev_sim", "title_word_jaccard", "title_char3_jaccard",
        "title_token_set", "title_containment", "title_tfidf_cos",
        "first_author_match", "author_surname_jaccard",
        "year_match", "year_diff",
        "arxiv_id_in_fields",
    ):
        assert is_nan(feature(vector, name)), name


def test_missing_data_filled_with_neutral_midpoint():
    # Tầng 2: fill_missing_features điền NaN bằng neutral midpoint 0.5.
    vector = fill_missing_features(np.array(build_features(make_entry(), make_candidate())))

    for name in (
        "title_lev_sim", "title_word_jaccard", "title_char3_jaccard",
        "title_token_set", "title_containment", "title_tfidf_cos",
        "first_author_match", "author_surname_jaccard",
        "year_match", "year_diff",
        "arxiv_id_in_fields",
    ):
        assert feature(vector.tolist(), name) == 0.5, name
    # Meta flags không bị ảnh hưởng bởi fill
    for name in ("has_title", "has_authors", "has_year"):
        assert feature(vector.tolist(), name) == 0.0, name

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

def test_candidate_id_normalized_via_model_construct_bypass():
    # Defense-in-depth: nếu ai đó dùng model_construct (bypass Pydantic) hoặc load
    # parquet thuần, ref_arxiv_id có thể chứa version/prefix. _prepare_candidate phải
    # normalize để vẫn match entry (canonical).
    entry = make_entry(journal="arXiv preprint arXiv:1706.03762")
    candidate = Reference.model_construct(
        arxiv_id=PAPER_ID,
        ref_arxiv_id="1706.03762v5",
        ref_title="",
        ref_authors=[],
        ref_submission_date="",
        ref_s2_id="",
    )

    assert feature(build_features(entry, candidate), "arxiv_id_in_fields") == 1.0

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

    for name in ("title_lev_sim", "title_word_jaccard", "first_author_match", "year_match"):
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

    tfidf = column(tensor, "title_tfidf_cos")
    assert tensor.shape == (2, 3, len(FEATURE_NAMES))
    assert [int(np.argmax(tfidf[i])) for i in range(2)] == [1, 2]


def test_feature_matrix_matches_single_pair_function():
    entry = make_entry(title="Attention Is All You Need", year=2017)
    candidate = make_candidate(ref_title="Attention is all you need")

    tensor = build_feature_matrix([entry], [candidate])
    single = build_features(
        entry, candidate, tfidf_cosine=tensor[0, 0, FEATURE_NAMES.index("title_tfidf_cos")]
    )

    # Ba cột ngữ cảnh chỉ có ở ma trận (single-pair để NaN): so sánh các cột còn lại, rồi kiểm
    # tra riêng ma trận 1x1 cho 0.0 theo spec (pool chỉ 1 phần tử -> mặc định đứng đầu).
    context = [FEATURE_NAMES.index(n) for n in (
        "composite_rank_in_pool", "composite_margin", "composite_col_rank_in_paper")]
    rest = [i for i in range(len(FEATURE_NAMES)) if i not in context]
    # nan_ok: nhiều feature là NaN (thiếu tác giả/chuỗi thô); hai bên phải NaN đúng cùng chỗ.
    assert tensor[0, 0, rest].tolist() == pytest.approx([single[i] for i in rest], nan_ok=True)
    assert tensor[0, 0, context].tolist() == [0.0, 0.0, 0.0]

def test_feature_matrix_is_equivariant_to_candidate_order():
    """Đảo thứ tự candidates -> tensor đảo theo cùng hoán vị."""
    entries = [
        make_entry(citation_key="a", title="Attention Is All You Need", year=2017),
        make_entry(citation_key="b", title="Deep Residual Learning", year=2016),
    ]
    candidates = [
        make_candidate(ref_arxiv_id="1706.03762", ref_title="Attention is all you need"),
        make_candidate(
            ref_arxiv_id="1512.03385",
            ref_title="Deep Residual Learning for Image Recognition",
        ),
        make_candidate(ref_arxiv_id="1810.04805", ref_title="BERT Pre-training"),
    ]
    perm = [2, 0, 1]

    base = build_feature_matrix(entries, candidates)
    shuffled = build_feature_matrix(entries, [candidates[i] for i in perm])

    np.testing.assert_allclose(shuffled, base[:, perm, :])


@pytest.mark.parametrize(("m", "n"), [(0, 3), (3, 0), (0, 0)])
def test_feature_matrix_handles_empty_sides(m, n):
    entries = [make_entry()] * m
    candidates = [make_candidate()] * n

    assert build_feature_matrix(entries, candidates).shape == (m, n, len(FEATURE_NAMES))


def test_tfidf_returns_nan_when_no_candidate_has_a_title():
    result = tfidf_cosine_matrix(["attention"], ["", ""])

    assert result.shape == (1, 2)
    assert np.isnan(result).all()


def test_tfidf_is_nan_only_for_cells_with_a_missing_title():
    result = tfidf_cosine_matrix(["attention is all", ""], ["attention is all", "", "resnet"])

    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 2] == 0.0  # có tiêu đề nhưng không chung từ nào: 0.0 thật
    assert np.isnan(result[0, 1])  # ứng viên thiếu tiêu đề
    assert np.isnan(result[1]).all()  # mục thiếu tiêu đề


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

    values = column(tensor, "arxiv_id_in_fields")[:, 0]
    assert values[0] == 1.0
    assert is_nan(values[1])  # mục b không có trường nào để dò ID -> không kết luận được
    with pytest.raises(ValueError):
        build_feature_matrix(entries, candidates, raw_texts=["only one"])


# ---- tác giả dạng list (lỗi cũ: nối list thành một chuỗi làm mất ranh giới tác giả) ----


def test_author_lists_in_different_formats_match_on_first_author_and_surnames():
    entry = make_entry(authors=["Vaswani, A.", "Shazeer, N."])
    candidate = make_candidate(ref_authors=["Ashish Vaswani", "Noam Shazeer"])

    vector = build_features(entry, candidate)

    assert feature(vector, "first_author_match") == 1.0
    assert feature(vector, "author_surname_jaccard") == 1.0

def test_accented_authors_match_between_bibtex_and_s2():
    # BibTeX hay viết Müller bằng LaTeX "M{\"u}ller"; S2 trả "Müller" (Unicode).
    entry = make_entry(authors=[r'M{\"u}ller, K.', r'Erd{\H{o}}s, P.'])
    candidate = make_candidate(ref_authors=["Klaus Müller", "Paul Erdős"])

    vector = build_features(entry, candidate)

    assert feature(vector, "first_author_match") == 1.0
    assert feature(vector, "author_surname_jaccard") == 1.0


# ---- feature mới: year_signed_diff, title_containment ----


def test_title_containment_is_high_for_truncated_title():
    entry = make_entry(title="Attention Is All You Need")
    full = make_candidate(ref_title="Attention Is All You Need: A Study of Transformers")
    unrelated = make_candidate(ref_title="Gradient Boosting for Tabular Data")

    assert feature(build_features(entry, full), "title_containment") == 1.0
    assert feature(build_features(entry, unrelated), "title_containment") < 0.7


def test_identical_short_titles_still_have_full_trigram_overlap():
    vector = build_features(make_entry(title="AI"), make_candidate(ref_title="AI"))

    assert feature(vector, "title_char3_jaccard") == 1.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1706.03762", "1706.03762"),                  # đã chuẩn
        ("1706.03762v5", "1706.03762"),                # strip version
        ("arXiv:1706.03762", "1706.03762"),            # strip prefix
        ("ARXIV:1706.03762V2", "1706.03762"),          # case-insensitive
        ("hep-th/9901001v3", "hep-th/9901001"),        # style cũ + version
        ("math.GT/0309136", "math.gt/0309136"),        # lowercase cho style cũ
        ("", ""),                                       # rỗng
        (None, ""),                                     # None (defensive)
    ],
)
def test_normalize_arxiv_id(raw, expected):
    assert _normalize_arxiv_id(raw) == expected


# ---- chính sách NaN: thiếu dữ liệu khác với "không giống" ----


def test_disjoint_titles_give_zero_not_nan():
    vector = build_features(
        make_entry(title="alpha beta gamma"), make_candidate(ref_title="delta epsilon zeta")
    )

    assert feature(vector, "has_title") == 1.0
    assert feature(vector, "title_word_jaccard") == 0.0
    assert not is_nan(feature(vector, "title_lev_sim"))


def test_missing_title_only_makes_title_features_nan():
    entry = make_entry(authors=["A. Vaswani"], year=2017)
    candidate = make_candidate(ref_authors=["Ashish Vaswani"], ref_submission_date="2017-06-12")

    vector = build_features(entry, candidate)

    assert feature(vector, "has_title") == 0.0
    assert is_nan(feature(vector, "title_lev_sim"))
    assert is_nan(feature(vector, "title_containment"))
    assert feature(vector, "first_author_match") == 1.0
    assert feature(vector, "year_match") == 1.0


def test_missing_candidate_side_alone_is_enough_for_nan():
    entry = make_entry(title="Attention Is All You Need", authors=["A. Vaswani"], year=2017)

    vector = build_features(entry, make_candidate())  # ứng viên không có gì

    assert [feature(vector, f) for f in ("has_title", "has_authors", "has_year")] == [0.0] * 3
    assert is_nan(feature(vector, "title_lev_sim"))
    assert is_nan(feature(vector, "first_author_match"))
    assert is_nan(feature(vector, "year_diff"))


def test_arxiv_id_feature_is_zero_when_fields_exist_but_lack_the_id_and_nan_otherwise():
    with_fields = build_features(make_entry(journal="NeurIPS 2017"), make_candidate())
    no_fields = build_features(make_entry(), make_candidate())

    assert feature(with_fields, "arxiv_id_in_fields") == 0.0
    assert is_nan(feature(no_fields, "arxiv_id_in_fields"))


def test_flags_are_never_nan_even_for_empty_pair():
    vector = build_features(make_entry(), make_candidate())

    for name in ("has_title", "has_authors", "has_year"):
        assert feature(vector, name) in (0.0, 1.0), name


def test_single_pair_leaves_context_features_nan():
    entry = make_entry(title="Attention Is All You Need")
    candidate = make_candidate(ref_title="Attention Is All You Need")

    vector = build_features(entry, candidate)

    for name in ("composite_rank_in_pool", "composite_margin", "composite_col_rank_in_paper"):
        assert is_nan(feature(vector, name)), name


def test_empty_feature_matrix_has_full_shape():
    assert build_feature_matrix([], [make_candidate()]).shape == (0, 1, len(FEATURE_NAMES))


# ---- composite_score ----


def test_composite_weights_sum_to_one_and_use_existing_features():
    assert sum(COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(COMPOSITE_WEIGHTS) <= set(FEATURE_NAMES)


def test_composite_ignores_missing_signals_instead_of_penalising_them():
    # Cùng tiêu đề, nhưng cả hai bên không có tác giả/năm: điểm vẫn phải ~1.0, không bị
    # kéo xuống 0.75 chỉ vì thiếu tín hiệu phụ.
    entry = make_entry(title="Attention Is All You Need")
    candidate = make_candidate(ref_title="Attention Is All You Need")

    score = composite_score_matrix(build_feature_matrix([entry], [candidate]))

    assert score[0, 0] == pytest.approx(1.0)


def test_composite_is_nan_without_title():
    entry = make_entry(authors=["A. Vaswani"], year=2017)
    candidate = make_candidate(ref_authors=["Ashish Vaswani"], ref_submission_date="2017-06-12")

    assert np.isnan(composite_score_matrix(build_feature_matrix([entry], [candidate]))).all()


# ---- hạng dense chuẩn hóa ----


def test_dense_percentile_rank_best_is_zero_and_ties_share_rank():
    ranks = _dense_percentile_rank(np.array([0.9, 0.5, 0.5, 0.1]))

    np.testing.assert_allclose(ranks, [0.0, 1 / 3, 1 / 3, 2 / 3])


def test_dense_percentile_rank_nan_is_worst_and_denominator_is_pool_size():
    # n = 3: ô NaN -> 1.0 (tệ nhất); hai ô hợp lệ có dense rank 1, 2 -> 0/2 và 1/2.
    ranks = _dense_percentile_rank(np.array([0.9, np.nan, 0.1]))

    np.testing.assert_allclose(ranks, [0.0, 1.0, 0.5])


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.7], [0.0]),  # pool chỉ 1 phần tử: mặc định đứng đầu
        ([0.7, np.nan], [0.0, 1.0]),  # một điểm hợp lệ duy nhất vẫn đứng đầu
        ([np.nan, np.nan], [1.0, 1.0]),  # không tính được: tệ nhất, không phải NaN
        ([np.nan], [1.0]),
        ([], []),
    ],
)
def test_dense_percentile_rank_never_returns_nan(scores, expected):
    ranks = _dense_percentile_rank(np.array(scores, dtype=float))

    np.testing.assert_allclose(ranks, expected)


def test_dense_percentile_rank_treats_float_noise_as_tie():
    ranks = _dense_percentile_rank(np.array([0.3, 0.1 + 0.2, 0.1]))

    assert ranks[0] == ranks[1] == 0.0


# ---- margin theo hàng ----


def test_margin_is_positive_for_winner_and_negative_for_others():
    margin = _margin_by_row(np.array([[0.9, 0.5, 0.2]]))

    np.testing.assert_allclose(margin, [[0.4, -0.4, -0.7]])


def test_margin_is_zero_for_tied_winners():
    margin = _margin_by_row(np.array([[0.8, 0.8, 0.1]]))

    np.testing.assert_allclose(margin, [[0.0, 0.0, -0.7]])


def test_margin_is_zero_for_nan_cells_and_rows_without_competitors():
    margin = _margin_by_row(np.array([[0.9, np.nan, 0.5], [0.4, np.nan, np.nan]]))

    np.testing.assert_allclose(margin, [[0.4, 0.0, -0.4], [0.0] * 3])


# ---- ba feature ngữ cảnh trên một bài hoàn chỉnh ----


def make_small_paper():
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
    return entries, candidates


def test_context_features_put_true_candidate_first_in_row_and_column():
    entries, candidates = make_small_paper()

    tensor = build_feature_matrix(entries, candidates)

    row_rank = column(tensor, "composite_rank_in_pool")
    margin = column(tensor, "composite_margin")
    col_rank = column(tensor, "composite_col_rank_in_paper")
    assert [row_rank[0, 1], row_rank[1, 2]] == [0.0, 0.0]  # đúng cặp: hạng 0 theo hàng
    assert margin[0, 1] > 0 > margin[0, 0]
    assert margin[1, 2] > 0 > margin[1, 0]
    assert [col_rank[0, 1], col_rank[1, 2]] == [0.0, 0.0]  # và cũng hạng 0 theo cột
    assert col_rank[1, 1] == 1.0  # entry b là entry khớp tệ hơn cho ref "Attention"


def test_context_features_are_neutral_for_a_paper_with_one_entry_or_one_candidate():
    entries, candidates = make_small_paper()

    one_entry = build_feature_matrix(entries[:1], candidates)
    one_candidate = build_feature_matrix(entries, candidates[:1])

    # m = 1: hạng theo cột mặc định 0.0; n = 1: hạng theo hàng 0.0 và margin 0.0.
    assert (column(one_entry, "composite_col_rank_in_paper") == 0.0).all()
    assert not np.isnan(column(one_entry, "composite_rank_in_pool")).any()
    assert (column(one_candidate, "composite_rank_in_pool") == 0.0).all()
    assert (column(one_candidate, "composite_margin") == 0.0).all()


def test_context_features_for_entry_without_title_follow_kiet_spec():
    entries, candidates = make_small_paper()
    entries.append(make_entry(citation_key="c"))  # không có tiêu đề

    tensor = build_feature_matrix(entries, candidates)

    # Thiếu tiêu đề: hạng (hàng và cột) = 1.0 tệ nhất, margin = 0.0 trung tính, không NaN.
    assert (column(tensor, "composite_rank_in_pool")[2] == 1.0).all()
    assert (column(tensor, "composite_col_rank_in_paper")[2] == 1.0).all()
    assert (column(tensor, "composite_margin")[2] == 0.0).all()
    for name in ("composite_rank_in_pool", "composite_margin", "composite_col_rank_in_paper"):
        assert not np.isnan(column(tensor, name)).any(), name


def test_context_rank_ignores_candidates_without_title():
    entries, candidates = make_small_paper()
    candidates[0] = make_candidate(ref_arxiv_id="1810.04805")  # ứng viên thiếu tiêu đề

    tensor = build_feature_matrix(entries[:1], candidates)

    row_rank = column(tensor, "composite_rank_in_pool")[0]
    assert row_rank[0] == 1.0  # ứng viên thiếu tiêu đề: tệ nhất
    # n = 3 nên mẫu số là 2: ứng viên hợp lệ tốt nhất 0/2, kém hơn 1/2 (vẫn tốt hơn ô thiếu).
    assert row_rank[1] == 0.0 and row_rank[2] == 0.5


def test_identical_candidate_titles_get_identical_context_features():
    entries, candidates = make_small_paper()
    candidates.append(make_candidate(ref_arxiv_id="1706.03763", ref_title="Attention is all you need"))

    tensor = build_feature_matrix(entries[:1], candidates)

    row_rank = column(tensor, "composite_rank_in_pool")[0]
    assert row_rank[1] == row_rank[3] == 0.0
    assert column(tensor, "composite_margin")[0, 1] == 0.0  # hòa nhất: không "áp đảo"


# ---- điền giá trị thiếu theo ý nghĩa (fill_missing_features) ----


def make_paper_with_gaps():
    """Bài có đủ các kiểu thiếu: entry thiếu năm/tác giả, ứng viên thiếu tiêu đề."""
    entries = [
        make_entry(citation_key="full", title="Attention Is All You Need", year=2017,
                   authors=["Ashish Vaswani"]),
        make_entry(citation_key="bare", title="Deep Residual Learning"),
        make_entry(citation_key="notitle", year=2016),
    ]
    candidates = [
        make_candidate(ref_arxiv_id="1706.03762", ref_title="Attention Is All You Need",
                       ref_authors=["Ashish Vaswani"], ref_submission_date="2017-06-12"),
        make_candidate(
            ref_arxiv_id="1512.03385", ref_title="Deep Residual Learning for Image Recognition"
        ),
        make_candidate(ref_arxiv_id="1810.04805"),  # thiếu hết metadata
    ]
    return entries, candidates


def test_every_feature_has_a_fill_value():
    assert set(MISSING_FILL_VALUES) == set(FEATURE_NAMES)


def test_fill_missing_removes_all_nan():
    entries, candidates = make_paper_with_gaps()

    raw = build_feature_matrix(entries, candidates)
    filled = build_feature_matrix(entries, candidates, fill_missing=True)

    assert np.isnan(raw).any()  # mặc định vẫn giữ NaN thô
    assert not np.isnan(filled).any()


def test_fill_missing_only_changes_nan_cells():
    entries, candidates = make_paper_with_gaps()

    raw = build_feature_matrix(entries, candidates)
    filled = build_feature_matrix(entries, candidates, fill_missing=True)

    known = ~np.isnan(raw)
    np.testing.assert_array_equal(filled[known], raw[known])


def test_fill_uses_per_feature_meaning_not_zero_everywhere():
    entries, candidates = make_paper_with_gaps()

    filled = build_feature_matrix(entries, candidates, fill_missing=True)
    pair = filled[1, 1]  # entry "bare" không năm/tác giả vs ứng viên không năm/tác giả

    # year_diff là khoảng cách: thiếu năm KHÔNG được thành 0 (0 nghĩa là trùng năm).
    assert feature(pair.tolist(), "year_diff") == 0.5
    assert feature(pair.tolist(), "year_match") == 0.5
    assert feature(pair.tolist(), "first_author_match") == 0.5
    # cờ vẫn báo đúng là thiếu để mô hình phân biệt "thiếu" với "khác".
    assert feature(pair.tolist(), "has_year") == 0.0
    assert feature(pair.tolist(), "has_authors") == 0.0


def test_fill_happens_after_composite_so_scores_are_not_distorted():
    entries, candidates = make_paper_with_gaps()

    raw = build_feature_matrix(entries, candidates)
    filled = build_feature_matrix(entries, candidates, fill_missing=True)

    # Hạng/margin ở ô có giá trị phải y hệt: điền 0 trước composite sẽ làm lệch chúng.
    for name in ("composite_rank_in_pool", "composite_margin", "composite_col_rank_in_paper"):
        known = ~np.isnan(column(raw, name))
        np.testing.assert_allclose(column(filled, name)[known], column(raw, name)[known])


def test_single_pair_context_fill_matches_kiet_spec():
    filled = fill_missing_features(np.array(build_features(make_entry(), make_candidate())))

    # Đường single-pair không có ma trận: không tính được -> hạng 1.0, margin 0.0.
    assert feature(filled.tolist(), "composite_rank_in_pool") == 1.0
    assert feature(filled.tolist(), "composite_col_rank_in_paper") == 1.0
    assert feature(filled.tolist(), "composite_margin") == 0.0


def test_fill_missing_features_does_not_mutate_input():
    raw = build_feature_matrix(*make_paper_with_gaps())
    snapshot = raw.copy()

    fill_missing_features(raw)

    np.testing.assert_array_equal(raw, snapshot)  # array_equal coi NaN ở cùng chỗ là bằng nhau


def test_fill_missing_features_accepts_single_vector_and_flat_pairs():
    vector = build_features(make_entry(), make_candidate())

    assert not np.isnan(fill_missing_features(np.array(vector))).any()
    assert fill_missing_features(np.array([vector, vector])).shape == (2, len(FEATURE_NAMES))


def test_fill_missing_features_rejects_wrong_column_count():
    with pytest.raises(ValueError):
        fill_missing_features(np.zeros((2, 3, len(FEATURE_NAMES) - 1)))


def test_fill_missing_handles_empty_paper():
    assert build_feature_matrix([], [make_candidate()], fill_missing=True).shape[0] == 0