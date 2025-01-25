import os
import re
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
import pandas as pd

def fetch_html_content(url):
    """Fetch HTML content from a website."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_and_download_pdfs(html_content, base_url, output_dir):
    """Parse HTML content, find PDF links, and download them."""
    soup = BeautifulSoup(html_content, 'html.parser')
    pdf_links = []

    # Find all PDF links
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.pdf'):
            pdf_links.append(href if href.startswith('http') else base_url + href)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Download PDFs
    for pdf_url in pdf_links:
        pdf_name = os.path.basename(pdf_url)
        pdf_path = os.path.join(output_dir, pdf_name)
        response = requests.get(pdf_url)
        with open(pdf_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {pdf_path}")

    return pdf_links

def convert_pdfs_to_text(pdf_dir, text_dir):
    """Convert all PDFs in a directory to text files."""
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text_path = os.path.join(text_dir, pdf_file.replace('.pdf', '.txt'))
            try:
                text = extract_text(pdf_path)
                with open(text_path, 'w') as file:
                    file.write(text)
                print(f"Converted: {text_path}")
            except Exception as e:
                print(f"Error converting {pdf_path}: {e}")

def main():
    # Target URL
    url = "https://cs229.stanford.edu/syllabus-summer2019.html"
    base_url = "https://cs229.stanford.edu/"

    # Fetch HTML content
    html_content = fetch_html_content(url)

    # Save HTML content to a file
    with open("website_content.txt", 'w') as file:
        file.write(html_content)
    print("HTML content saved.")

    # Parse and download PDFs
    pdf_dir = "pdfs"
    pdf_links = parse_and_download_pdfs(html_content, base_url, pdf_dir)

    # Convert PDFs to text
    text_dir = "pdf_texts"
    convert_pdfs_to_text(pdf_dir, text_dir)

    # Example: Creating a CSV summary of PDFs
    pdf_summary = pd.DataFrame({
        "PDF Name": [os.path.basename(link) for link in pdf_links],
        "URL": pdf_links
    })
    pdf_summary.to_csv("pdf_summary.csv", index=False)
    print("PDF summary saved to pdf_summary.csv.")

if __name__ == "__main__":
    main()
