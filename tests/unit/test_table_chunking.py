import unittest

from src.pipeline.tables import normalize_table


class TableChunkingTests(unittest.TestCase):
    def test_large_table_repeats_header_unit_and_parent(self):
        rows = [["期限", "金额"]] + [[str(i), str(i * 10)] for i in range(45)]
        blocks = normalize_table(
            rows,
            document_version="dv-a",
            locator="body/000003/table",
            caption="期限结构",
            unit="万元",
            max_rows=20,
        )
        parent, *groups = blocks

        self.assertEqual([len(group.table_rows) for group in groups], [20, 20, 5])
        self.assertTrue(all(group.parent_id == parent.block_id for group in groups))
        self.assertTrue(all("[单位] 万元" in group.text for group in groups))
        self.assertTrue(all(group.table_headers == ["期限", "金额"] for group in groups))
        self.assertEqual(parent.block_type, "table")
        self.assertTrue(all(group.block_type == "table_rows" for group in groups))

    def test_multi_row_header_is_flattened_without_losing_original_rows(self):
        rows = [
            ["", "2025年", "2025年"],
            ["项目", "数量", "金额"],
            ["存款", "2", "100"],
        ]
        blocks = normalize_table(
            rows,
            document_version="dv-a",
            locator="sheet/S1/rows/1-3",
        )
        child = blocks[1]

        self.assertEqual(child.table_headers, ["项目", "2025年/数量", "2025年/金额"])
        self.assertEqual(child.meta["header_rows"], rows[:2])

    def test_row_groups_are_deterministic(self):
        rows = [["项目", "金额"], ["现金", "10"], ["存款", "20"]]
        first = [
            b.block_id
            for b in normalize_table(
                rows,
                document_version="dv-a",
                locator="sheet/S1/rows/1-100",
            )
        ]
        second = [
            b.block_id
            for b in normalize_table(
                rows,
                document_version="dv-a",
                locator="sheet/S1/rows/1-100",
            )
        ]
        self.assertEqual(first, second)

    def test_empty_rows_produce_no_blocks(self):
        self.assertEqual(
            normalize_table(
                [],
                document_version="dv-a",
                locator="sheet/S1/rows/1-1",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
