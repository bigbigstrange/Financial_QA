from .docir import DocBlock, DocIR, DocPage
from .identity import ParseContext, make_block_id, make_document_version
from .parsers import (
    BaseDocumentParser,
    ExcelDocumentParser,
    ImageDocumentParser,
    LegacyWordDocParser,
    PDFDocumentParser,
    PlainTextParser,
    PowerPointDocumentParser,
    RtfDocumentParser,
    WordDocumentParser,
    build_default_parsers,
)
from .quality import (
    ParseQualityReport,
    ProcessedDocument,
    QualityGateError,
    SnapshotQualityReport,
    assess_document,
    assess_snapshot,
    process_document,
)
from .tables import normalize_table

__all__ = [
    "DocBlock",
    "DocIR",
    "DocPage",
    "BaseDocumentParser",
    "PlainTextParser",
    "WordDocumentParser",
    "LegacyWordDocParser",
    "PDFDocumentParser",
    "ImageDocumentParser",
    "ExcelDocumentParser",
    "PowerPointDocumentParser",
    "RtfDocumentParser",
    "build_default_parsers",
    "ParseContext",
    "make_block_id",
    "make_document_version",
    "normalize_table",
    "ParseQualityReport",
    "ProcessedDocument",
    "QualityGateError",
    "SnapshotQualityReport",
    "assess_document",
    "assess_snapshot",
    "process_document",
]
