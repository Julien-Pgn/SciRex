from pathlib import Path

import duckdb
import pytest
from PIL import Image

from scirex.ocr.modern.pipeline import (
    extract_images,
    get_pdf_paths_for_keyword,
    parse_html,
    parse_markdown,
)


def _chunk(label, content, bbox=None):
    return {"label": label, "content": content, "bbox": bbox}


def test_parse_html_skips_blank_page():
    chunks = [_chunk("Blank-Page", "<p>ignored</p>")]
    assert parse_html("<html></html>", chunks) == ""


def test_parse_html_skips_headers_and_footers_by_default():
    chunks = [
        _chunk("Page-Header", "<p>header</p>"),
        _chunk("Page-Footer", "<p>footer</p>"),
        _chunk("Text", "<p>body</p>"),
    ]
    assert parse_html("<html></html>", chunks) == "<p>body</p>"


def test_parse_html_includes_headers_and_footers_when_requested():
    chunks = [_chunk("Page-Header", "<p>header</p>")]
    assert parse_html("<html></html>", chunks, include_headers_footers=True) == "<p>header</p>"


def test_parse_html_skips_images_when_not_requested():
    chunks = [_chunk("Image", "<div><img/></div>")]
    assert parse_html("<html></html>", chunks, include_images=False) == ""


def test_parse_html_wraps_plain_text_in_paragraph():
    chunks = [_chunk("Text", "raw text without tags")]
    assert parse_html("<html></html>", chunks) == "<p>raw text without tags</p>"


def test_parse_html_adds_src_to_image_tag():
    chunks = [_chunk("Figure", "<div><img/></div>")]
    out = parse_html("<html></html>", chunks)
    assert "src=" in out


def test_extract_images_crops_bbox():
    image = Image.new("RGB", (100, 100), color="white")
    chunks = [_chunk("Image", "<div><img/></div>", bbox=(0, 0, 10, 10))]
    images = extract_images("<html></html>", chunks, image)
    assert len(images) == 1
    cropped = next(iter(images.values()))
    assert cropped.size == (10, 10)


def test_extract_images_skips_invalid_bbox():
    image = Image.new("RGB", (100, 100), color="white")
    chunks = [_chunk("Image", "<div><img/></div>", bbox=(10, 10, 0, 0))]
    images = extract_images("<html></html>", chunks, image)
    assert images == {}


def test_extract_images_skips_chunk_without_img_tag():
    image = Image.new("RGB", (100, 100), color="white")
    chunks = [_chunk("Image", "<div>no image here</div>", bbox=(0, 0, 10, 10))]
    images = extract_images("<html></html>", chunks, image)
    assert images == {}


def test_parse_markdown_converts_basic_html():
    md = parse_markdown("<h1>Title</h1><p>Body text</p>")
    assert "# Title" in md
    assert "Body text" in md


def test_parse_markdown_returns_empty_string_on_error(monkeypatch):
    import scirex.ocr.modern.pipeline as pipeline

    class BoomMarkdownify:
        def __init__(self, **kwargs):
            pass

        def convert(self, html):
            raise ValueError("boom")

    monkeypatch.setattr(pipeline, "Markdownify", BoomMarkdownify)
    assert parse_markdown("<p>irrelevant</p>") == ""


@pytest.fixture
def paper_local_conn():
    connection = duckdb.connect(":memory:")
    connection.execute("""
        CREATE TABLE paper_local (
            arxiv_id VARCHAR PRIMARY KEY,
            pdf_path VARCHAR,
            keyword_for_ocr VARCHAR
        )
        """)
    connection.execute("""
        INSERT INTO paper_local (arxiv_id, pdf_path, keyword_for_ocr) VALUES
            ('1111.1111', 'data/raw/pdfs/1111.1111.pdf', 'quantization'),
            ('2222.2222', 'data/raw/pdfs/2222.2222.pdf', 'quantization'),
            ('3333.3333', 'data/raw/pdfs/3333.3333.pdf', 'genai_subset'),
            ('4444.4444', NULL, 'quantization')
        """)
    yield connection
    connection.close()


def test_get_pdf_paths_for_keyword_filters_to_matching_keyword(paper_local_conn):
    result = get_pdf_paths_for_keyword(paper_local_conn, "quantization")
    assert set(result) == {Path("data/raw/pdfs/1111.1111.pdf"), Path("data/raw/pdfs/2222.2222.pdf")}


def test_get_pdf_paths_for_keyword_excludes_other_keywords(paper_local_conn):
    result = get_pdf_paths_for_keyword(paper_local_conn, "genai_subset")
    assert result == [Path("data/raw/pdfs/3333.3333.pdf")]


def test_get_pdf_paths_for_keyword_excludes_null_pdf_path(paper_local_conn):
    # 4444.4444 matches the keyword but has no pdf_path yet (fetch failed/pending) — must be skipped
    result = get_pdf_paths_for_keyword(paper_local_conn, "quantization")
    assert Path("data/raw/pdfs/4444.4444.pdf") not in result
    assert len(result) == 2


def test_get_pdf_paths_for_keyword_returns_empty_for_unknown_keyword(paper_local_conn):
    assert get_pdf_paths_for_keyword(paper_local_conn, "nonexistent-topic") == []
