import sys
from pathlib import Path

def extract_pdf_to_text(pdf_path: Path) -> Path:
    from pdfminer.high_level import extract_text
    text = extract_text(str(pdf_path))
    out_path = pdf_path.with_suffix('.txt')
    out_path.write_text(text or "", encoding='utf-8')
    return out_path

def main() -> int:
    root = Path('.').resolve()
    inputs = [
        root / 'Hackathon AI Agent Approach Outline.pdf',
        root / 'Project Synapse.pdf',
    ]
    try:
        for pdf in inputs:
            if not pdf.exists():
                print(f"Missing file: {pdf}")
                return 1
            out_txt = extract_pdf_to_text(pdf)
            print(f"Extracted: {out_txt}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

