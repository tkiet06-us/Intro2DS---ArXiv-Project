"""Schema dữ liệu chung của dự án: 7 bảng Parquet của M3 (task P1-02).

Vì sao cần file này: theo nguyên tắc IDD (README mục 1.2), cả nhóm chốt "hình dạng" dữ liệu
ngay từ Tuần 1 để M1, M2, M3 code song song mà không phải chờ nhau.

"""

from enum import StrEnum
from typing import Annotated, Literal, Self

import pyarrow as pa
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from src.common.ids import validate_arxiv_id


def make_element_id(arxiv_id: str, local_id: str) -> str:
    """Ghép element_id toàn cục từ arxiv_id và id cục bộ trong bài.

    Args:
        arxiv_id: Mã arXiv chuẩn của bài.
        local_id: Id phân biệt các element trong cùng một bài, ví dụ "s1".

    Returns:
        Chuỗi dạng "2310.12345#s1".
    """
    return f"{arxiv_id}#{local_id}"


# Annotated gắn validator vào KIỂU, nên dùng lại cho mọi trường ID mà không phải lặp code.
# (Cách `_v = field_validator(...)` trong bản nháp sẽ bị Pydantic v2 bỏ qua vì tên bắt đầu
# bằng "_" được coi là thuộc tính private, validator không chạy.)
ArxivId = Annotated[str, AfterValidator(validate_arxiv_id)]
# Các khóa non-null của Parquet cũng không được là chuỗi rỗng.
NonEmptyStr = Annotated[str, Field(min_length=1)]
Version = Annotated[int, Field(ge=1)]


class Kind(StrEnum):
    """Loại element trong cây phân cấp (cột `kind` của bảng elements)."""

    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    ITEMIZE = "itemize"
    ITEM = "item"
    SENTENCE = "sentence"
    FORMULA = "formula"
    FIGURE = "figure"
    TABLE = "table"


class FlatModel(BaseModel):
    """Lớp cha cho mọi bảng: cấm trường lạ để schema không bị "trôi" âm thầm."""

    model_config = ConfigDict(extra="forbid")


class Paper(FlatModel):
    """Một dòng của bảng `papers` (cũng là nội dung metadata.json của M1)."""

    arxiv_id: ArxivId
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    submission_date: str | None = None  # ISO, ví dụ "2023-10-05"
    revised_dates: list[str] = Field(default_factory=list)
    venue: str | None = None


class Element(FlatModel):
    """Một dòng của bảng `elements` (đã khử trùng lặp giữa các version)."""

    element_id: NonEmptyStr
    arxiv_id: ArxivId
    kind: Kind
    content: str


class ElementParent(FlatModel):
    """Một dòng của bảng `element_parents`: cha của element trong từng version."""

    arxiv_id: ArxivId
    version: Version
    element_id: NonEmptyStr
    parent_id: NonEmptyStr


class BibtexEntry(FlatModel):
    """Một dòng của bảng `bibtex_entries` (các trường BibTeX đã chuẩn hóa, phẳng)."""

    arxiv_id: ArxivId
    citation_key: NonEmptyStr
    entry_type: str  # article, inproceedings, misc, ...
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None  # entry thiếu năm là chuyện thường gặp
    doi: str | None = None
    journal: str | None = None
    url: str | None = None


class PaperReference(FlatModel):
    """Một dòng của bảng `paper_references`: một cặp (bài trích dẫn, bài được trích dẫn)."""

    arxiv_id: ArxivId  # bài đang trích dẫn
    ref_arxiv_id: ArxivId  # bài được trích dẫn (ref không có arXiv ID thì bị loại từ M1)
    ref_title: str | None = None
    ref_authors: list[str] = Field(default_factory=list)
    ref_submission_date: str | None = None
    ref_s2_id: str | None = None


class Prediction(FlatModel):
    """Một dòng của bảng `predictions`: top-5 ứng viên cho một BibTeX entry."""

    arxiv_id: ArxivId
    partition: Literal["train", "valid", "test"]
    citation_key: NonEmptyStr
    groundtruth_arxiv_id: ArxivId | None = None
    cand_1: ArxivId | None = None
    cand_2: ArxivId | None = None
    cand_3: ArxivId | None = None
    cand_4: ArxivId | None = None
    cand_5: ArxivId | None = None

    @model_validator(mode="after")
    def candidates_are_ranked(self) -> Self:
        """Đảm bảo top-5 liền mạch từ hạng 1 và không có ứng viên trùng nhau.

        Raises:
            ValueError: Khi có cand null nằm giữa các cand có giá trị, hoặc có ứng viên trùng.
        """
        cands = [self.cand_1, self.cand_2, self.cand_3, self.cand_4, self.cand_5]
        filled = [cand for cand in cands if cand is not None]
        # Danh sách xếp hạng không có "lỗ hổng": cand_3 có giá trị thì cand_1, cand_2 cũng phải có.
        if cands[: len(filled)] != filled:
            raise ValueError("cand null nằm giữa: top-5 phải liền mạch từ cand_1")
        # Một bài xuất hiện hai lần sẽ làm hạng trong MRR khó diễn giải.
        if len(filled) != len(set(filled)):
            raise ValueError("ứng viên trùng nhau trong top-5")
        return self


class Source(FlatModel):
    """Một dòng của bảng `sources`: một file LaTeX/BibTeX gốc của một version."""

    arxiv_id: ArxivId
    version: Version
    filename: str
    content: str


# ---- Schema Parquet tường minh: nguồn sự thật cho kiểu cột khi ghi file .parquet ----
# Pydantic kiểm tra dữ liệu trong code, còn Parquet cần kiểu cột cụ thể (int32, dictionary...).
# Ta giữ cả hai và dùng tests/test_schemas.py để đảm bảo chúng không lệch nhau.
_STR = pa.string()
# Cột có ít giá trị lặp lại (kind, partition, entry_type) dùng dictionary để nén và để
# Hugging Face viewer hiển thị gọn, đúng yêu cầu "string, dictionary" của đề.
_DICT = pa.dictionary(pa.int8(), pa.string())
_STR_LIST = pa.list_(_STR)  # chỉ cho phép list phẳng một cấp, không lồng object

PARQUET_SCHEMAS: dict[str, pa.Schema] = {
    "papers": pa.schema(
        [
            pa.field("arxiv_id", _STR, nullable=False),
            pa.field("title", _STR),
            pa.field("authors", _STR_LIST),
            pa.field("submission_date", _STR),
            pa.field("revised_dates", _STR_LIST),
            pa.field("venue", _STR),
        ]
    ),
    "elements": pa.schema(
        [
            pa.field("element_id", _STR, nullable=False),
            pa.field("arxiv_id", _STR, nullable=False),
            pa.field("kind", _DICT),
            pa.field("content", _STR),
        ]
    ),
    "element_parents": pa.schema(
        [
            pa.field("arxiv_id", _STR),
            pa.field("version", pa.int32()),
            pa.field("element_id", _STR, nullable=False),
            pa.field("parent_id", _STR, nullable=False),
        ]
    ),
    "bibtex_entries": pa.schema(
        [
            pa.field("arxiv_id", _STR),
            pa.field("citation_key", _STR, nullable=False),
            pa.field("entry_type", _DICT),
            pa.field("title", _STR),
            pa.field("authors", _STR_LIST),
            pa.field("year", pa.int32()),
            pa.field("doi", _STR),
            pa.field("journal", _STR),
            pa.field("url", _STR),
        ]
    ),
    "paper_references": pa.schema(
        [
            pa.field("arxiv_id", _STR),
            pa.field("ref_arxiv_id", _STR, nullable=False),
            pa.field("ref_title", _STR),
            pa.field("ref_authors", _STR_LIST),
            pa.field("ref_submission_date", _STR),
            pa.field("ref_s2_id", _STR),
        ]
    ),
    "predictions": pa.schema(
        [
            pa.field("arxiv_id", _STR, nullable=False),
            pa.field("partition", _DICT),
            pa.field("citation_key", _STR, nullable=False),
            pa.field("groundtruth_arxiv_id", _STR),
        ]
        + [pa.field(f"cand_{rank}", _STR) for rank in range(1, 6)]
    ),
    "sources": pa.schema(
        [
            pa.field("arxiv_id", _STR),
            pa.field("version", pa.int32()),
            pa.field("filename", _STR),
            pa.field("content", _STR),
        ]
    ),
}

# Ánh xạ tên bảng -> model Pydantic, để code chung (mock, exporter) duyệt qua cả 7 bảng.
MODELS: dict[str, type[FlatModel]] = {
    "papers": Paper,
    "elements": Element,
    "element_parents": ElementParent,
    "bibtex_entries": BibtexEntry,
    "paper_references": PaperReference,
    "predictions": Prediction,
    "sources": Source,
}