import win32com.client
import os
from pathlib import Path
import pythoncom  # COM初始化必需
import time

try:
    import psutil
except Exception:
    psutil = None

try:
    import win32process
except Exception:
    win32process = None


def word_to_pdf_win32(doc_path, pdf_path):
    """
    使用Word引擎转换 - 增强稳定版
    """
    doc_path = os.path.abspath(doc_path)
    pdf_path = os.path.abspath(pdf_path)

    word = None
    doc = None
    word_pid = None

    try:
        # 显式初始化COM
        pythoncom.CoInitialize()

        # 创建Word应用
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False  # 后台运行
        try:
            word.DisplayAlerts = 0  # 禁止弹窗
        except Exception:
            pass
        try:
            # 禁止宏（更安全，也能减少弹窗）
            word.AutomationSecurity = 3  # msoAutomationSecurityForceDisable
        except Exception:
            pass
        try:
            word.Options.SaveNormalPrompt = False
        except Exception:
            pass

        # 记录该 Word 进程 PID，后续只在卡死时杀掉它
        try:
            if win32process is not None:
                _, word_pid = win32process.GetWindowThreadProcessId(word.Hwnd)
        except Exception:
            word_pid = None

        # 打开文档
        doc = word.Documents.Open(
            doc_path,
            ReadOnly=True,
            AddToRecentFiles=False,
            ConfirmConversions=False,
            Visible=False,
        )

        # 关键设置：确保格式保留
        # 更新所有域（目录、页码等）
        try:
            doc.Fields.Update()
        except Exception:
            # 部分文档（损坏/保护/兼容性）会在 Fields.Update 报错，跳过即可
            pass

        # 导出为PDF：优先 ExportAsFixedFormat，其次 SaveAs/SaveAs2
        exported = False
        try:
            # 17 = wdExportFormatPDF
            doc.ExportAsFixedFormat(OutputFileName=pdf_path, ExportFormat=17)
            exported = True
        except Exception:
            pass

        if not exported:
            try:
                # 17 = wdFormatPDF
                doc.SaveAs(pdf_path, FileFormat=17)
                exported = True
            except Exception:
                pass

        if not exported:
            try:
                # SaveAs2 在部分 Word 版本更稳定
                doc.SaveAs2(pdf_path, FileFormat=17)
                exported = True
            except Exception as e:
                raise e

        print(f"✅ 转换成功: {Path(doc_path).name}")
        return True

    except Exception as e:
        print(f"❌ 转换失败 {Path(doc_path).name}: {e}")
        return False

    finally:
        # 彻底清理
        if doc:
            try:
                doc.Close(SaveChanges=False)
            except:
                pass
        if word:
            try:
                word.Quit()
            except:
                # 如果有模态对话框卡住 Quit，尝试仅杀掉脚本启动的 WINWORD
                if psutil is not None and word_pid is not None:
                    try:
                        p = psutil.Process(int(word_pid))
                        p.kill()
                        p.wait(timeout=5)
                        time.sleep(1)
                    except Exception:
                        pass

        # 释放COM
        pythoncom.CoUninitialize()

def batch_word_to_pdf(input_dir, output_dir, skip_exist=True):
    """
    批量转换Word文件到PDF
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 扫描所有Word文件
    word_files = list(input_path.glob("*.docx")) + list(input_path.glob("*.doc")) + list(input_path.glob("*.rtf"))

    if not word_files:
        print(f"⚠️ 未找到Word文件")
        return

    print(f"📊 找到 {len(word_files)} 个待转换文件\n")

    stats = {"成功": 0, "失败": 0, "跳过": 0}

    # 逐个转换
    for word_file in word_files:
        pdf_file = output_path / f"{word_file.stem}.pdf"

        # 跳过已存在文件
        if skip_exist and pdf_file.exists():
            stats["跳过"] += 1
            continue

        # 转换
        if word_to_pdf_win32(str(word_file), str(pdf_file)):
            stats["成功"] += 1
        else:
            stats["失败"] += 1

    # 输出统计
    print(f"\n{'=' * 50}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"输出目录: {output_path}")


# 使用示例
if __name__ == "__main__":
    code_root = Path(__file__).resolve().parent.parent
    batch_word_to_pdf(
        input_dir=str(code_root / "data" / "documents" / "dataset" / "数据集" / "word"),
        output_dir=str(code_root / "output" / "converted" / "word")
    )