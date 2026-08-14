import win32com.client
import os
from pathlib import Path
from tqdm import tqdm
import time
import psutil
import pythoncom  # COM初始化必需


def kill_powerpoint_processes():
    """强制关闭所有挂起的PowerPoint进程"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'POWERPNT.EXE' in proc.info['name'].upper():
                proc.kill()
                proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass
    time.sleep(1)  # 等待进程完全释放


def ppt_to_pdf_win32(ppt_path, pdf_path):
    """
    使用PowerPoint引擎转换 - 增强稳定版
    """
    ppt_path = os.path.abspath(ppt_path)
    pdf_path = os.path.abspath(pdf_path)

    # 清理残留进程
    kill_powerpoint_processes()

    presentation = None
    powerpoint = None

    try:
        # 显式初始化COM
        pythoncom.CoInitialize()

        # 创建PowerPoint应用
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        # 跳过Visible设置（如果它经常出错）
        # powerpoint.Visible = False

        # 打开演示文稿
        presentation = powerpoint.Presentations.Open(ppt_path, WithWindow=False)

        # 保存为PDF
        presentation.SaveAs(pdf_path, 32)  # 32 = ppSaveAsPDF

        print(f"✅ 转换成功: {Path(ppt_path).name}")
        return True

    except Exception as e:
        print(f"❌ 转换失败 {Path(ppt_path).name}: {e}")
        return False

    finally:
        # 彻底清理
        if presentation:
            try:
                presentation.Close()
            except:
                pass
        if powerpoint:
            try:
                powerpoint.Quit()
            except:
                pass

        # 释放COM
        pythoncom.CoUninitialize()

        # 再次清理进程
        kill_powerpoint_processes()


def batch_ppt_to_pdf(input_dir, output_dir, skip_exist=True):
    """
    批量转换 - 带错误隔离和重试
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 扫描PPT文件
    ppt_files = list(input_path.glob("*.ppt")) + list(input_path.glob("*.pptx")) + list(input_path.glob("*.pps"))

    if not ppt_files:
        print(f"⚠️ 未找到PPT文件")
        return

    print(f"📊 找到 {len(ppt_files)} 个文件")

    stats = {"成功": 0, "失败": 0, "跳过": 0}

    # 逐个转换（避免并发问题）
    for ppt_file in tqdm(ppt_files, desc="转换进度"):
        pdf_file = output_path / f"{ppt_file.stem}.pdf"

        # 跳过已存在文件
        if skip_exist and pdf_file.exists():
            stats["跳过"] += 1
            continue

        # 转换并等待（避免接口过载）
        if ppt_to_pdf_win32(str(ppt_file), str(pdf_file)):
            stats["成功"] += 1
            time.sleep(2)  # 成功等待2秒
        else:
            stats["失败"] += 1
            time.sleep(5)  # 失败等待5秒

    # 输出统计
    print(f"\n{'=' * 50}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"输出目录: {output_path}")


# 使用示例
if __name__ == "__main__":
    code_root = Path(__file__).resolve().parent.parent
    batch_ppt_to_pdf(
        input_dir=str(code_root / "data" / "documents" / "dataset" / "数据集" / "ppt"),
        output_dir=str(code_root / "output" / "converted" / "ppt"),
        skip_exist=True
    )