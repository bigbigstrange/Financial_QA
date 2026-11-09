import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED

from src.pipeline.file_validation import FileLimits, FileValidationError, validate_input_file


class FileValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_renamed_executable_is_rejected(self):
        path = self.tmp_path / "report.pdf"
        path.write_bytes(b"MZ" + b"x" * 100)

        with self.assertRaises(FileValidationError) as raised:
            validate_input_file(path, FileLimits())
        self.assertEqual(raised.exception.code, "signature_mismatch")

    def test_ooxml_declared_size_limit_is_enforced(self):
        path = self.tmp_path / "huge.docx"
        with zipfile.ZipFile(path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "x" * 200_000)

        with self.assertRaises(FileValidationError) as raised:
            validate_input_file(
                path,
                FileLimits(max_uncompressed_bytes=100_000),
            )
        self.assertEqual(raised.exception.code, "decompressed_size_limit")

    def test_ooxml_macro_member_is_rejected(self):
        path = self.tmp_path / "macro.docx"
        with zipfile.ZipFile(path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<w:document/>")
            archive.writestr("word/vbaProject.bin", b"macro")

        with self.assertRaises(FileValidationError) as raised:
            validate_input_file(path, FileLimits())
        self.assertEqual(raised.exception.code, "macro_embedded")

    def test_valid_pdf_and_ooxml_are_accepted(self):
        pdf = self.tmp_path / "rules.pdf"
        pdf.write_bytes(b"%PDF-1.7\n%%EOF")
        validated = validate_input_file(pdf, FileLimits())
        self.assertEqual(validated.detected_format, ".pdf")
        self.assertEqual(len(validated.source_sha256), 64)

        docx = self.tmp_path / "rules.docx"
        with zipfile.ZipFile(docx, "w", ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<w:document/>")
        validated = validate_input_file(docx, FileLimits())
        self.assertEqual(validated.detected_format, ".docx")
        self.assertEqual(validated.display_name, "rules.docx")

    def test_archive_entry_limit_is_enforced(self):
        path = self.tmp_path / "many.docx"
        with zipfile.ZipFile(path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", "<w:document/>")
            for i in range(5):
                archive.writestr(f"extra/{i}.xml", "<x/>")

        with self.assertRaises(FileValidationError) as raised:
            validate_input_file(path, FileLimits(max_zip_entries=4))
        self.assertEqual(raised.exception.code, "archive_entry_limit")

    def test_display_name_never_contains_parent_path(self):
        path = self.tmp_path / "rules.pdf"
        path.write_bytes(b"%PDF-1.7\n%%EOF")

        validated = validate_input_file(path, FileLimits())

        self.assertNotIn("..", validated.display_name)
        self.assertNotIn("/", validated.display_name)


if __name__ == "__main__":
    unittest.main()
