"""Render a ViDoRe corpus page's markdown text into a simple PDF.

We don't try to reproduce the original page layout -- we just need a PDF whose
extractable text matches the page content, so the search agent can ingest and
retrieve it. reportlab flows the text into paragraphs on letter-sized pages.
"""

from __future__ import annotations

import io
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from evaluator.types import GoldenPage

_styles = getSampleStyleSheet()



def _escape(text: str) -> str:
    """reportlab Paragraph uses a mini-HTML; escape the angle brackets/ampersands."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )



def render_pdf(page: GoldenPage) -> bytes:
    """Return PDF bytes for one golden page."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title=page.pdf_name)
    flow = [Paragraph(_escape(f"{page.doc_id} (page {page.page_number})"), _styles["Title"])]
    flow.append(Spacer(1, 12))
    # Split markdown into paragraph-ish blocks on blank lines; collapse the rest.
    blocks = re.split(r"\n\s*\n", page.markdown.strip())
    for block in blocks:
        text = " ".join(block.split())
        if text:
            flow.append(Paragraph(_escape(text), _styles["BodyText"]))
            flow.append(Spacer(1, 6))
    if len(flow) <= 2:
        flow.append(Paragraph("(no extractable text on this page)", _styles["BodyText"]))
    doc.build(flow)
    return buf.getvalue()