import win32com.client
import os
from pathlib import Path


def excel_to_pdf_win32(excel_path, pdf_path):
    """
    使用Excel引擎转换单个文件
    """
    excel_path = os.path.abspath(excel_path)
    pdf_path = os.path.abspath(pdf_path)

    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False

    try:
        workbook = excel.Workbooks.Open(excel_path)

        # 优化页面设置
        for sheet in workbook.Sheets:
            sheet.PageSetup.Zoom = False
            sheet.PageSetup.FitToPagesWide = 1
            sheet.PageSetup.FitToPagesTall = False

        workbook.ExportAsFixedFormat(0, pdf_path)
        print(f"✅ 转换成功: {Path(excel_path).name} → {Path(pdf_path).name}")
        return True

    except Exception as e:
        print(f"❌ 转换失败 {Path(excel_path).name}: {e}")
        return False

    finally:
        if 'workbook' in locals():
            workbook.Close(False)
        excel.Quit()


def batch_convert_excel_to_pdf(input_dir, output_dir):
    """
    批量转换指定目录下所有Excel文件

    :param input_dir: Excel文件所在目录
    :param output_dir: PDF输出目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 扫描所有xls和xlsx文件
    excel_files = list(input_path.glob("*.xlsx")) + list(input_path.glob("*.xls"))

    if not excel_files:
        print(f"⚠️ 未在 {input_dir} 找到Excel文件")
        return

    print(f"📊 找到 {len(excel_files)} 个Excel文件\n")

    # 批量转换
    success_count = 0
    for excel_file in excel_files:
        # 生成输出PDF路径
        pdf_file = output_path / f"{excel_file.stem}.pdf"

        if excel_to_pdf_win32(str(excel_file), str(pdf_file)):
            success_count += 1

    print(f"\n{'=' * 50}")
    print(f"📈 批量转换完成: {success_count}/{len(excel_files)} 个成功")
    print(f"📁 输出目录: {output_path}")


# 使用示例
if __name__ == "__main__":
    code_root = Path(__file__).resolve().parent.parent
    # 转换 data/excel 目录下所有文件到 output/excel 目录
    batch_convert_excel_to_pdf(
        input_dir=str(code_root / "data" / "documents" / "dataset" / "数据集" / "excel"),
        output_dir=str(code_root / "output" / "converted" / "excel")
    )