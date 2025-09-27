import argparse
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer

def load_model(model_name: str):
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tok, model

def translate_block(text: str, tok, model) -> str:
    # Translate one block/paragraph; keeps formatting simple
    inputs = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    generated = model.generate(**inputs, max_new_tokens=512)
    return tok.batch_decode(generated, skip_special_tokens=True)[0]

def chunk_paragraph(p: str, tok, max_tokens=450):
    """
    Simple chunker for long paragraphs: splits by sentences (.) and fills chunks
    without exceeding ~max_tokens. Falls back to rough word chunks if needed.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', p.strip())
    chunks, cur = [], ""
    for s in sentences:
        trial = (cur + " " + s).strip() if cur else s
        if len(tok(trial)["input_ids"]) <= max_tokens:
            cur = trial
        else:
            if cur:
                chunks.append(cur)
            # if single sentence is too big, hard-split by words
            if len(tok(s)["input_ids"]) > max_tokens:
                words = s.split()
                buf = []
                for w in words:
                    t = (" ".join(buf + [w])).strip()
                    if len(tok(t)["input_ids"]) <= max_tokens:
                        buf.append(w)
                    else:
                        chunks.append(" ".join(buf))
                        buf = [w]
                if buf:
                    chunks.append(" ".join(buf))
                cur = ""
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return chunks

def translate_text_preserve_layout(text: str, model_name: str) -> str:
    tok, model = load_model(model_name)
    out_lines = []
    # Preserve blank lines/paragraphs
    for para in text.splitlines():
        if not para.strip():
            out_lines.append("")  # keep empty line
            continue
        for chunk in chunk_paragraph(para, tok):
            out_lines.append(translate_block(chunk, tok, model))
    return "\n".join(out_lines)

def main():
    ap = argparse.ArgumentParser(description="Translate a text file with MarianMT")
    ap.add_argument("--in", dest="inp", default="transcription.txt", help="Input text file")
    ap.add_argument("--out", dest="out", default="translation.txt", help="Output file")
    ap.add_argument("--model", default="Helsinki-NLP/opus-mt-en-es", help="Marian model name")
    args = ap.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    src_text = in_path.read_text(encoding="utf-8").strip()
    if not src_text:
        raise SystemExit("Input file is empty.")

    print(f"Translating {in_path} → {args.out} using {args.model} ...")
    translated = translate_text_preserve_layout(src_text, args.model)
    Path(args.out).write_text(translated, encoding="utf-8")
    print("Done. Saved:", args.out)

if __name__ == "__main__":
    main()
