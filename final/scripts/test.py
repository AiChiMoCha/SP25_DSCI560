import os
import argparse
import sys
from pathlib import Path

# 设置tesseract语言数据目录
TESSDATA_DIR = "/work/hdd/beaa/cyu7/SP25_DSCI560/final/scripts/tessdata"

# 尝试导入OCR所需的库
try:
    import pytesseract
    from pdf2image import convert_from_path
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("警告: OCR依赖库未安装。请安装所需的库:")
    print("pip install pytesseract pdf2image")
    print("你还需要安装tesseract-ocr和poppler:")
    print("- 在Ubuntu/Debian上: sudo apt-get install tesseract-ocr poppler-utils")
    print("- 在macOS上: brew install tesseract poppler")
    print("- 在Windows上: 从各自网站下载安装")
    exit(1)

# 导入PyPDF2用于常规文本提取
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("警告: PyPDF2库未安装。请安装:")
    print("pip install PyPDF2")
    exit(1)

def extract_text_with_ocr(pdf_path, sample_size=200):
    """
    使用OCR从PDF中提取文本
    
    参数:
        pdf_path: PDF文件路径
        sample_size: 预览文本的字符数
    
    返回:
        提取的文本
    """
    try:
        print(f"  尝试使用OCR处理: {os.path.basename(pdf_path)}")
        
        # 转换PDF为图像
        images = convert_from_path(pdf_path, first_page=1, last_page=10)  # 只处理前3页作为测试
        
        text = ""
        
        # 从每个图像中提取文本
        for i, image in enumerate(images):
            print(f"    OCR处理第 {i+1}/{len(images)} 页...")
            
            # 执行OCR (使用指定的tessdata目录)
            page_text = pytesseract.image_to_string(image, config=f'--tessdata-dir {TESSDATA_DIR}')
            text += f"\n--- 第 {i+1} 页 ---\n{page_text}"
        
        # 显示文本预览
        clean_text = text.replace('\n', ' ').strip()
        preview = clean_text[:sample_size] + "..." if len(clean_text) > sample_size else clean_text
        print(f"    OCR提取结果预览: {preview}")
        
        return text
        
    except Exception as e:
        print(f"  OCR提取错误: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path, sample_size=200):
    """
    从PDF提取文本，如果常规提取失败则使用OCR
    
    参数:
        pdf_path: PDF文件路径
        sample_size: 预览文本的字符数
    
    返回:
        提取的文本
    """
    # 首先尝试常规提取
    try:
        print(f"  尝试常规文本提取: {os.path.basename(pdf_path)}")
        text = ""
        
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            page_count = min(10, len(reader.pages))  # 只处理前3页作为测试
                
            for i in range(page_count):
                page = reader.pages[i]
                page_text = page.extract_text() or ""
                text += page_text
        
        # 检查是否获得有意义的文本
        if len(text.strip()) < 100:
            print("  常规提取结果太少，尝试OCR...")
            return extract_text_with_ocr(pdf_path, sample_size)
        
        # 显示文本预览
        clean_text = text.replace('\n', ' ').strip()
        preview = clean_text[:sample_size] + "..." if len(clean_text) > sample_size else clean_text
        print(f"  常规提取结果预览: {preview}")
            
        return text
        
    except Exception as e:
        print(f"  常规提取错误: {str(e)}")
        print("  尝试使用OCR作为备用...")
        return extract_text_with_ocr(pdf_path, sample_size)

def test_pdf_extraction(docs_folder, output_folder=None):
    """
    测试从文件夹中提取所有PDF的文本
    
    参数:
        docs_folder: 包含PDF文件的文件夹
        output_folder: 输出文本文件的文件夹
    """
    print(f"测试从 {docs_folder} 中提取PDF文本...")
    
    # 检查tessdata目录是否存在
    if not os.path.exists(TESSDATA_DIR):
        print(f"警告: tessdata目录不存在: {TESSDATA_DIR}")
        return
        
    # 检查语言数据文件是否存在
    if not os.path.exists(os.path.join(TESSDATA_DIR, "eng.traineddata")):
        print(f"警告: 英语语言数据文件不存在: {os.path.join(TESSDATA_DIR, 'eng.traineddata')}")
        return
    
    # 创建输出文件夹（如果指定）
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # 处理每个PDF
    pdf_count = 0
    success_count = 0
    
    for file_name in os.listdir(docs_folder):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(docs_folder, file_name)
            pdf_count += 1
            
            try:
                print(f"\n正在处理 ({pdf_count}): {file_name}")
                
                # 提取文本
                text = extract_text_from_pdf(pdf_path)
                
                # 检查是否提取到文本
                if not text or len(text.strip()) < 50:
                    print(f"  警告: 提取的文本很少或为空")
                    continue
                
                # 保存结果
                if output_folder:
                    output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"  文本已保存到: {output_file}")
                
                success_count += 1
                
            except Exception as e:
                print(f"  处理 {file_name} 时出错: {str(e)}")
    
    print(f"\n完成! 处理了 {pdf_count} 个PDF文件，成功提取了 {success_count} 个文件的文本")
    
    if pdf_count == 0:
        print(f"在 {docs_folder} 中没有找到PDF文件")

def main():
    parser = argparse.ArgumentParser(description='测试PDF文本提取和OCR功能')
    parser.add_argument('--docs', type=str, default='../docs',
                      help='包含PDF文件的文件夹路径')
    parser.add_argument('--output', type=str, default=None,
                      help='输出文本文件的文件夹路径')
    parser.add_argument('--max-pages', type=int, default=100,
                      help='每个PDF处理的最大页数 (默认: 10)')
    args = parser.parse_args()
    
    # 检查文件夹是否存在
    if not os.path.exists(args.docs):
        print(f"错误: 文件夹 '{args.docs}' 不存在")
        return
    
    test_pdf_extraction(args.docs, args.output)

if __name__ == "__main__":
    main()