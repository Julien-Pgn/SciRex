import io
from pathlib import Path
import pymupdf
from PIL import Image
from chandra.model.vllm import generate_vllm
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown
import csv

def pdf_to_images(pdf_path, max_pages = None, dpi = 192):
    '''Take a pdf, return a list of pages converted in png images'''
    # Open the document and count the number of pages
    doc = pymupdf.open(pdf_path)

    if max_pages is None:
        max_pages = doc.page_count
    else:
        max_pages = min(max_pages, doc.page_count)

    img_list = []
    for page in range(max_pages):
        pix = doc[page].get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        img_list.append(img)
    doc.close()
    
    return img_list


def images_to_markdown(images, max_output_tokens=4096, max_workers=20):
    '''Receives a list of PIL images, returns the concatenated markdown and per-page stats.'''
    
    batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in images]
    results = generate_vllm(
        batch,
        max_output_tokens=max_output_tokens,
        max_workers=max_workers,
    )

    per_page_md = []
    per_page_stats = []

    for i, result in enumerate(results):
        md = parse_markdown(result.raw)
        per_page_md.append(md)
        per_page_stats.append({
            'page': i,
            'tokens': result.token_count,
            'n_chars': len(md),
            'error': result.error,
        })
    
    full = ""
    for i, md in enumerate(per_page_md):
        full += f"\n\n<!-- ===== Page {i+1} ===== -->\n\n"
        full += md
        full += "\n"

    return full, per_page_stats

def stats_path_for(pdf_path, output_dir):
    '''Compute the csv output path for storing the stats of each pdf'''
    return Path(output_dir) / "stats" / (Path(pdf_path).stem + ".csv")

def save_stats(stats, pdf_path, output_dir):
    '''Save the per-page stats to a .csv file in output_dir, naming the file after the source PDF.'''
    output_csv = stats_path_for(pdf_path, output_dir)
    output_csv.parent.mkdir(parents = True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["page", "tokens", "n_chars", "error"])
        writer.writeheader()
        writer.writerows(stats)
    
    return output_csv


def output_path_for(pdf_path, output_dir):
    '''Compute the markdown output path for a given pdf'''
    return Path(output_dir) / (Path(pdf_path).stem + ".md")

def save_markdown(md, pdf_path, output_dir):
    '''Save the markdown to output_dir, naming the file after the source PDF.'''
    output_md = output_path_for(pdf_path, output_dir)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    
    header = f"<!-- Source: {Path(pdf_path).name} -->\n"
    output_md.write_text(header + md)
    
    return output_md