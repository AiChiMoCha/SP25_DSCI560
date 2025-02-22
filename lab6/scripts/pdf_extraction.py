#!/usr/bin/env python3
import os
import re
import pytesseract
import ocrmypdf
import tempfile
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from sqlalchemy import create_engine, text

def extract_text_from_pdf(pdf_path):
    """
    尝试使用 ocrmypdf 将 PDF 转换为带文字层的 PDF，
    然后使用 PyPDF2 提取文本。如果转换出错则回退到原有 OCR 方法。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ocr_pdf = os.path.join(tmpdir, "temp.pdf")
        try:
            # 强制OCR，即使已有文字层
            ocrmypdf.ocr(pdf_path, ocr_pdf, force_ocr=True)
        except Exception as e:
            print(f"使用 ocrmypdf 处理 {pdf_path} 时出错：{e}")
            return extract_text_fallback(pdf_path)
        
        try:
            reader = PdfReader(ocr_pdf)
            text_content = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
            return text_content
        except Exception as e:
            print(f"使用 PyPDF2 读取OCR PDF时出错：{e}")
            return extract_text_fallback(pdf_path)

def extract_text_fallback(pdf_path):
    """
    回退方案：使用 PyPDF2 直接提取文本，如果结果为空，再使用 pdf2image+pytesseract。
    """
    text_content = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"
    except Exception as e:
        print(f"回退方案：使用 PyPDF2 处理 {pdf_path} 时出错：{e}")
    if not text_content.strip():
        try:
            images = convert_from_path(pdf_path)
            for image in images:
                text_content += pytesseract.image_to_string(image) + "\n"
        except Exception as e:
            print(f"回退方案：使用 OCR 处理 {pdf_path} 时出错：{e}")
    return text_content

def extract_api_from_text(full_text):
    """
    在全文中搜索关键词 “API”、“API Number”、“API No.” 或 “API #” 后，
    在接下来的30个字符内查找符合10位数字格式（2位+3位+5位，允许空格或连字符）的字符串。
    返回匹配到的字符串（例如 "33-105-90258"），否则返回 None。
    """
    keywords_pattern = re.compile(r"(?si)(API(?:\s*Number|\s*No\.?|\s*#))")
    for match in keywords_pattern.finditer(full_text):
        start = match.end()
        window = full_text[start:start+30]
        # 注意：字符集中的连字符需要转义
        digits_match = re.search(r"\(?(\d{2}\s*[\-\–]\s*\d{3}\s*[\-\–]\s*\d{5})\)?", window)
        if digits_match:
            candidate = digits_match.group(1).strip()
            candidate_clean = re.sub(r"[\s\-\–]+", "", candidate)
            if len(candidate_clean) == 10:
                return candidate
    return None

def parse_stimulation_data(full_text):
    """
    在全文中查找刺激数据行，例如：
    '07/20/2013 Dakota 5625 6150 8 9000 Gallons'
    若匹配到则返回该字符串，否则返回 None。
    """
    stim_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4}.*Gallons)", full_text)
    if stim_match:
        return stim_match.group(1).strip()
    return None

def extract_well_name(full_text):
    """
    尝试提取井名（Well Name 或 Well Name and Number）
    """
    match = re.search(r"Well Name(?:\s*and Number)?\s*[:：]?\s*(.+)", full_text, re.IGNORECASE)
    if match:
        # 取第一行作为井名
        return match.group(1).splitlines()[0].strip()
    return None

def extract_address(full_text):
    """
    尝试提取地址（Address 后面的内容）
    """
    match = re.search(r"Address\s*[:：]?\s*(.+)", full_text, re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip()
    return None

def extract_coordinates(full_text):
    """
    尝试提取经纬度，例如：
    '48° 06' 35.99" -103° 43' 54.57"'
    返回 (latitude, longitude) 若未找到则返回 (None, None)
    """
    coord_pattern = re.compile(
        r"(\d{1,2}°\s*\d{1,2}'\s*\d{1,2}(?:\.\d+)?\")\s*([\-–]?\d{1,4}°\s*\d{1,2}'\s*\d{1,2}(?:\.\d+)?\")"
    )
    match = coord_pattern.search(full_text)
    if match:
        latitude = match.group(1).strip()
        longitude = match.group(2).strip()
        return latitude, longitude
    return None, None

def extract_field(full_text):
    """
    尝试提取 Field 字段（井所在 Field，例如 "Baker"）
    """
    match = re.search(r"Field\s*[:：]?\s*(\S+)", full_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_county(full_text):
    """
    尝试提取 County 字段，例如 "Williams"
    """
    match = re.search(r"County\s*[:：]?\s*(\S+)", full_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_fields(full_text):
    """
    提取各项字段，包括 API、刺激数据、井名、地址、经纬度、Field、County 等。
    """
    data = {}
    data["api"] = extract_api_from_text(full_text)
    data["stimulation_data"] = parse_stimulation_data(full_text)
    data["well_name"] = extract_well_name(full_text)
    data["address"] = extract_address(full_text)
    lat, lon = extract_coordinates(full_text)
    data["latitude"] = lat
    data["longitude"] = lon
    data["field"] = extract_field(full_text)
    data["county"] = extract_county(full_text)
    return data

def ensure_required_columns(engine):
    """
    检查数据库表 oil_wells 是否存在所需字段，如果缺失则自动添加。
    所需字段包括：api (VARCHAR), stimulation_data (TEXT),
    well_name (VARCHAR), address (TEXT), latitude (VARCHAR), longitude (VARCHAR),
    field (VARCHAR), county (VARCHAR)。
    """
    with engine.connect() as conn:
        result = conn.execute(text("SHOW COLUMNS FROM oil_wells"))
        existing_cols = [row["Field"] for row in result.mappings().all()]
    needed = {
        "api": "VARCHAR(255)",
        "stimulation_data": "TEXT",
        "well_name": "VARCHAR(255)",
        "address": "TEXT",
        "latitude": "VARCHAR(50)",
        "longitude": "VARCHAR(50)",
        "field": "VARCHAR(100)",
        "county": "VARCHAR(100)"
    }
    for col, col_type in needed.items():
        if col not in existing_cols:
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE oil_wells ADD COLUMN {col} {col_type}"))
                print(f"字段 '{col}' 已添加。")
            except Exception as e:
                print(f"添加字段 '{col}' 时出错：{e}")
        else:
            print(f"字段 '{col}' 已存在。")

def main():
    # 修改为你的数据库连接信息
    db_url = "mysql+pymysql://DSCI560:560560@172.16.161.128/oil_wells_db"
    engine = create_engine(db_url, echo=False)
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS oil_wells (
        id INT AUTO_INCREMENT PRIMARY KEY
    )
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))
        print("表 oil_wells 已创建或已存在。")
    
    ensure_required_columns(engine)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "..", "pdfs")
    
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"\n处理 {pdf_path} …")
            
            full_text = extract_text_from_pdf(pdf_path)
            # 不再保存全文到txt，而是直接处理
            fields = extract_fields(full_text)
            print("提取到的字段:", fields)
            
            insert_sql = """
            INSERT INTO oil_wells (api, stimulation_data, well_name, address, latitude, longitude, field, county)
            VALUES (:api, :stimulation_data, :well_name, :address, :latitude, :longitude, :field, :county)
            """
            with engine.begin() as conn:
                conn.execute(text(insert_sql), {
                    "api": fields.get("api"),
                    "stimulation_data": fields.get("stimulation_data"),
                    "well_name": fields.get("well_name"),
                    "address": fields.get("address"),
                    "latitude": fields.get("latitude"),
                    "longitude": fields.get("longitude"),
                    "field": fields.get("field"),
                    "county": fields.get("county")
                })
            print(f"{filename} 已写入数据库。")
    
    print("\n所有 PDF 文件处理完毕。")

if __name__ == "__main__":
    main()
