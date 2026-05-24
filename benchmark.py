import argparse
import os
import re
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import yaml
import tiktoken
from transformers import AutoTokenizer


def load_tokenizer(config: Dict) -> Callable[[str], List[str]]:
    tokenizer_type = config["type"]

    if tokenizer_type == "huggingface":
        tok = AutoTokenizer.from_pretrained(config["name"], use_fast=True)

        def encode(text: str) -> List[str]:
            ids = tok.encode(text, add_special_tokens=False)
            return tok.convert_ids_to_tokens(ids)

        return encode

    if tokenizer_type == "tiktoken":
        enc = tiktoken.get_encoding(config["name"])

        def encode(text: str) -> List[str]:
            ids = enc.encode(text)
            tokens = []
            for token_id in ids:
                try:
                    tokens.append(enc.decode([token_id]))
                except Exception:
                    tokens.append(f"<{token_id}>")
            return tokens

        return encode

    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def is_japanese_like(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tokenizer fragmentation for pharmaceutical terms."
    )
    parser.add_argument("--tokenizer_a", required=True, help="Tokenizer key in tokenizers.yaml")
    parser.add_argument("--tokenizer_b", required=True, help="Tokenizer key in tokenizers.yaml")
    parser.add_argument("--input", required=True, help="CSV file with an 'input' column")
    parser.add_argument("--config", default="tokenizers.yaml", help="Tokenizer config YAML")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--frag_threshold", type=int, default=3, help="Over-fragmentation threshold")

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.tokenizer_a not in cfg["tokenizers"]:
        raise KeyError(f"{args.tokenizer_a} not found in {args.config}")
    if args.tokenizer_b not in cfg["tokenizers"]:
        raise KeyError(f"{args.tokenizer_b} not found in {args.config}")

    encode_a = load_tokenizer(cfg["tokenizers"][args.tokenizer_a])
    encode_b = load_tokenizer(cfg["tokenizers"][args.tokenizer_b])

    df = pd.read_csv(args.input)
    if "input" not in df.columns:
        raise ValueError("Input CSV must contain an 'input' column.")

    rows = []
    for text in df["input"].astype(str):
        tokens_a = encode_a(text)
        tokens_b = encode_b(text)
        chars = max(len(text), 1)

        rows.append({
            "input": text,
            "len_A": len(tokens_a),
            "len_B": len(tokens_b),
            "delta_B_minus_A": len(tokens_b) - len(tokens_a),
            "tokens_A": " | ".join(tokens_a),
            "tokens_B": " | ".join(tokens_b),
            "token_per_char_A": len(tokens_a) / chars,
            "token_per_char_B": len(tokens_b) / chars,
            "overfrag_A": len(tokens_a) > args.frag_threshold,
            "overfrag_B": len(tokens_b) > args.frag_threshold,
        })

    out = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "report.csv"
    jsonl_path = output_dir / "report.jsonl"
    cand_path = output_dir / "add_tokens_candidates.txt"

    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    out.to_json(jsonl_path, orient="records", force_ascii=False, lines=True)

    suggestions = (
        out[(out["overfrag_A"]) | (out["overfrag_B"])]["input"]
        .drop_duplicates()
        .tolist()
    )
    suggestions = [s for s in suggestions if is_japanese_like(s)]

    cand_path.write_text("\n".join(suggestions) + "\n", encoding="utf-8")

    print(out[[
        "input", "len_A", "len_B", "delta_B_minus_A",
        "token_per_char_A", "token_per_char_B",
        "overfrag_A", "overfrag_B"
    ]])

    print("\nSummary")
    print(f"A={args.tokenizer_a}: {int(out['overfrag_A'].sum())}/{len(out)} over threshold")
    print(f"B={args.tokenizer_b}: {int(out['overfrag_B'].sum())}/{len(out)} over threshold")
    print("\nSaved:")
    print(csv_path)
    print(jsonl_path)
    print(cand_path)


if __name__ == "__main__":
    main()
