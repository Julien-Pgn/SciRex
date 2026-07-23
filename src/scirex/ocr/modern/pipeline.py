import io
import os
import re
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from chandra.model.schema import BatchInputItem
from chandra.model.vllm import generate_vllm
from chandra.output import Markdownify, get_image_name, parse_chunks
from PIL import Image


def pdf_to_images(pdf_path, max_pages=None, dpi=192):
    """Take a pdf, return a list of pages converted in png images"""

    # Open the document and count the number of pages
    doc = pymupdf.open(pdf_path)

    max_pages = doc.page_count if max_pages is None else min(max_pages, doc.page_count)

    img_list = []
    for page in range(max_pages):
        pix = doc[page].get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        img_list.append(img)
    doc.close()

    return img_list


# Vendored from chandra/output.py v0.2.0 — modifs : IMAGE_LABELS tunable
IMAGE_LABELS = ("Image", "Figure", "Diagram")


# Modification 1: parse_html now takes chunks as input instead of the raw html string
def parse_html(
    html: str,
    chunks: list[dict],
    include_headers_footers: bool = False,
    include_images: bool = True,
):
    out_html = ""
    for div_idx, chunk in enumerate(chunks, start=1):
        label = chunk["label"]
        content = chunk["content"]

        if label == "Blank-Page":
            continue

        # Skip headers and footers if not included
        if not include_headers_footers and label in ["Page-Header", "Page-Footer"]:
            continue
        if not include_images and label in IMAGE_LABELS:
            continue

        # Traiter les images
        if label in IMAGE_LABELS:
            img_src = get_image_name(html, div_idx)
            img_soup = BeautifulSoup(content, "html.parser")
            img = img_soup.find("img")
            if img:
                img["src"] = img_src
            else:
                img = BeautifulSoup(f"<img src='{img_src}'/>", "html.parser")
                img_soup.append(img)
            content = str(img_soup)

        # Traiter le texte
        elif label == "Text" and not re.search("<.+>", content.strip()):
            content = f"<p>{content.strip()}</p>"

        out_html += content
    return out_html


# Vendored from chandra/output.py v0.2.0 — modifs : IMAGE_LABELS tunable
# Modification 2: extract_images now takes chunks as input instead of the raw html string
def extract_images(html: str, chunks: list[dict], image: Image.Image):
    images = {}
    for div_idx, chunk in enumerate(chunks, start=1):
        if chunk["label"] in IMAGE_LABELS:
            img = BeautifulSoup(chunk["content"], "html.parser").find("img")
            if not img:
                continue
            bbox = chunk["bbox"]
            try:
                block_image = image.crop(bbox)
            except ValueError:
                continue
            img_name = get_image_name(html, div_idx)
            images[img_name] = block_image
    return images


# Vendored from chandra/output.py v0.2.0 — modifs : IMAGE_LABELS tunable
# Modification 3: parse_markdown now takes chunks as input instead of the raw html string
def parse_markdown(html: str):
    md_cls = Markdownify(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
    )
    try:
        markdown = md_cls.convert(html)
    except Exception as e:
        print(f"Error converting HTML to Markdown: {e}")
        markdown = ""
    return markdown.strip()


def ocr_images(page_images, max_output_tokens=8192, max_workers=20):
    """Receives a list of PIL images, returns the concatenated markdown and per-page stats, html and images from the article."""

    # Create a batch of images to send to vLLM
    batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in page_images]

    # vLLM OCR
    results = generate_vllm(
        batch,
        max_output_tokens=max_output_tokens,
        max_workers=max_workers,
    )

    per_page_md = []
    per_page_stats = []
    per_page_figures = []
    per_page_html = []

    for i, (result, page_img) in enumerate(zip(results, page_images, strict=True)):
        # Get the markdown, chunks and images for the page
        chunks = parse_chunks(result.raw, page_img)
        figures = extract_images(result.raw, chunks, page_img)
        html = parse_html(result.raw, chunks)
        md = parse_markdown(html)

        per_page_md.append(md)
        per_page_html.append(result.raw)
        per_page_figures.append(figures)
        per_page_stats.append(
            {
                "page": i,
                "n_tokens": result.token_count,
                "n_chars": len(md),
                "n_images": len(figures),
                "error": result.error,
            }
        )

    # Aggregate the per-page results into a single html for the entire document
    # "" is to instanciate an empty string, then we concatenate the per-page html with a page separator comment
    full_html = "\n\n".join(
        f"<!-- ===== Page {i + 1} ===== -->\n\n{html}" for i, html in enumerate(per_page_html)
    )

    # Aggregate the per-page results into a single md for the entire document
    full_md = "\n\n".join(
        f"<!-- ===== Page {i + 1} ===== -->\n\n{md}" for i, md in enumerate(per_page_md)
    )
    return full_md, per_page_stats, per_page_figures, full_html


def make_paper_dir(arxiv_id: str) -> Path:
    """Creates the output path for each paper in data/processed as a data lake"""

    paper_dir = Path("data/processed") / arxiv_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    return paper_dir


def atomic_write_text(path: Path, text: str) -> None:
    """A function to write text to a file in an atomic way = to avoid partial writes."""
    tmp_path = path.parent / (path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_image(path: Path, image: Image.Image) -> None:
    """A function to write an image to a file in an atomic way = to avoid partial writes."""
    tmp_path = path.parent / (path.name + ".tmp")
    image.save(tmp_path, format="WEBP")
    os.replace(tmp_path, path)


def save_all(paper_dir: Path, arxiv_id: str, html, md, images):
    """Writes all artefacts from the OCR process to disk."""

    # html:
    atomic_write_text(paper_dir / f"{arxiv_id}.html", html)

    # figures:
    for figures in images:
        for name, img in figures.items():
            atomic_write_image(paper_dir / name, img)

    # md:
    atomic_write_text(paper_dir / f"{arxiv_id}.md", md)


def get_pdf_paths_for_keyword(conn, keyword: str) -> list[Path]:
    """Return PDF paths for papers fetched under this topic (paper_local.keyword_for_ocr).

    This is what makes run_ocr.py safe to point at data/raw/pdfs/, a folder shared
    across every topic: without this filter, OCR would process every topic's
    backlog sitting in that directory, not just the one --keyword asks for.
    """
    rows = conn.execute(
        "SELECT pdf_path FROM paper_local WHERE keyword_for_ocr = ? AND pdf_path IS NOT NULL",
        [keyword],
    ).fetchall()
    return [Path(row[0]) for row in rows]


def update_db(
    conn,
    arxiv_id,
    stats,
    nb_pages_pdf,
    nb_pages_ocr,
    keyword_for_ocr,
    ocr_model,
    paper_dir,
    pdf_path,
):
    """Fills ocr_stats (one row per page) and paper_local (one row per paper)."""

    # ocr_stats : delete the lines for this paper first, then reinsert
    # -> replayable without duplicates if we rerun the OCR on the same paper
    conn.execute("DELETE FROM ocr_stats WHERE arxiv_id = ?", [arxiv_id])
    for stat in stats:
        conn.execute(
            """
            INSERT INTO ocr_stats (arxiv_id, page, n_tokens, n_chars, n_images, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                arxiv_id,
                stat["page"],
                stat["n_tokens"],
                stat["n_chars"],
                stat["n_images"],
                stat["error"],
            ),
        )

    # paper_local : upsert (INSERT ... ON CONFLICT) for idempotency, to avoid duplicates if we rerun the OCR on the same paper
    conn.execute(
        """
        INSERT INTO paper_local
            (arxiv_id, pdf_path, ocr_path, ocr_model, ocr_date,
             keyword_for_ocr, nb_pages_pdf, nb_pages_ocr, ocr_done)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, TRUE)
        ON CONFLICT (arxiv_id) DO UPDATE SET
            pdf_path        = excluded.pdf_path,
            ocr_path        = excluded.ocr_path,
            ocr_model       = excluded.ocr_model,
            ocr_date        = excluded.ocr_date,
            keyword_for_ocr = excluded.keyword_for_ocr,
            nb_pages_pdf    = excluded.nb_pages_pdf,
            nb_pages_ocr    = excluded.nb_pages_ocr,
            ocr_done        = TRUE
        """,
        (
            arxiv_id,
            str(pdf_path),
            str(paper_dir),
            ocr_model,
            keyword_for_ocr,
            nb_pages_pdf,
            nb_pages_ocr,
        ),
    )
