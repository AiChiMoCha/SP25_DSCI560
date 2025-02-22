#!/usr/bin/env python3
import os
import tempfile
import ocrmypdf
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_pdf(pdf_path):
    """
    使用 ocrmypdf 将 PDF 转换成带文字层的 PDF，
    然后用 PyPDF2 提取文本。
    如果转换出错，则回退到直接使用 PyPDF2，再兜底使用 pytesseract。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ocr_pdf = os.path.join(tmpdir, "temp_ocr.pdf")
        try:
            ocrmypdf.ocr(pdf_path, ocr_pdf, force_ocr=True)
            reader = PdfReader(ocr_pdf)
            text_content = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
            return text_content
        except Exception as e:
            print(f"ocrmypdf转换 {pdf_path} 时出错: {e}")
            try:
                reader = PdfReader(pdf_path)
                text_content = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                if text_content.strip():
                    return text_content
            except Exception as e:
                print(f"使用 PyPDF2 直接提取时出错: {e}")
            
            try:
                images = convert_from_path(pdf_path)
                text_content = ""
                for image in images:
                    text_content += pytesseract.image_to_string(image) + "\n"
                return text_content
            except Exception as e:
                print(f"使用 pytesseract 处理 {pdf_path} 时出错: {e}")
                return ""

if __name__ == "__main__":
    pdf_path = "../pdfs/W90258.pdf"
    if os.path.exists(pdf_path):
        print(f"===== 开始打印 {pdf_path} 的全文 =====")
        full_text = extract_text_from_pdf(pdf_path)
        print(full_text)
        # 将全文保存到 output.txt 中
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"全文已保存到 output.txt")
        print(f"===== {pdf_path} 的全文结束 =====\n")
    else:
        print(f"文件 {pdf_path} 不存在，请检查路径。")
