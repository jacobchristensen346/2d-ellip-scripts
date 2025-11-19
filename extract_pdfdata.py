# This script extracts data from the pdf 
# as saved on the Gaertner 2D Ellipsometer
# for a particular scan/run

import PyPDF2
import numpy as np

pdfname = "5x5_Silicon1_CircularScan_NoEdge_Stats.pdf"
all_content = ''
with open(pdfname, "rb") as pdf_file:
    read_pdf = PyPDF2.PdfReader(pdf_file)
    num_pages = len(read_pdf.pages)
    for page_num in np.arange(num_pages):
        page = read_pdf.pages[int(page_num)]
        page_content = page.extract_text()
        all_content += page_content

patt_stddev = f"StdDev\s*(.*?)\s" 
patt_mean = f"Mean\s*(.*?)\s" 
patt_rad = f"R=\s*(.*?),"
patt_theta = f"Theta=\s*(.*?),"
patt_thick = f"Thick1=\s*(.*?),"
stddev = float(re.search(patt_stddev, all_content).group(1))
mean = float(re.search(patt_mean, all_content).group(1))
matches_rad = re.findall(patt_rad, all_content)
matches_theta = re.findall(patt_theta, all_content)
matches_thick = re.findall(patt_thick, all_content)
