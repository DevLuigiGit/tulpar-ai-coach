"""Minimal PDF writer for tests: no PDF library needed to create fixtures."""

from pathlib import Path


def write_pdf(path: Path, pages: list[str]) -> Path:
    """One Helvetica text line per "\\n"-separated line of each page; "" makes a page without a text layer."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "", "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in text.split("\n")]
        stream = ("BT /F1 12 Tf 14 TL 72 720 Td " + " T* ".join(f"({line}) Tj" for line in lines) + " ET") if text else ""
        objects.append(f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objects)} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path = Path(path)
    path.write_bytes(bytes(out))
    return path
