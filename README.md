# Pharma Tokenizer Benchmark

Benchmark token fragmentation for Japanese pharmaceutical and medical terminology.

This repository evaluates how different tokenizers split:

- Japanese drug names
- Katakana medical terms
- Salt forms
- Long chemical names
- IUPAC-style nomenclature

## Why this matters

Medical LLMs frequently hallucinate or misspell pharmaceutical terms.

One hidden cause is tokenizer fragmentation.

Example:

```text
ロプリノン塩酸塩
```

may become:

```text
ロ | プ | リ | ノ | ン | 塩 | 酸 | 塩
```

instead of:

```text
オルプリノン | 塩酸塩
```

Excessive fragmentation can lead to:

- hallucination
- unstable spelling
- context inefficiency
- increased inference cost
- degraded embedding quality

## Features

- Compare Hugging Face tokenizers and tiktoken vocabularies
- Measure token fragmentation
- Export CSV / JSONL reports
- Compute token-per-character efficiency
- Suggest candidate vocabulary additions
- Includes Google Colab notebook

## Installation

```bash
pip install -r requirements.txt
```

## CLI Usage

```bash
python benchmark.py \
  --tokenizer_a bert_ja \
  --tokenizer_b cl100k \
  --input sample_data/jp_drugs.csv
```

Output:

```text
outputs/report.csv
outputs/report.jsonl
outputs/add_tokens_candidates.txt
```

## Google Colab

Open:

```text
notebooks/colab_demo.ipynb
```

or upload it to Google Colab.

The notebook installs dependencies, runs the benchmark, displays tables, and downloads output files.

## Example Output

| input | len_A | len_B |
|---|---:|---:|
| ロリノン塩酸塩 | 6 | 13 |
| ミルリノン | 3 | 7 |

Actual values depend on the selected tokenizers.



## Why start from tokenizer-level evaluation?

This project intentionally starts from low-level tokenization analysis before evaluating full LLM responses.

LLM processing roughly follows:

```text
text
↓
tokenizer
↓
embedding
↓
model
↓
response
```

Evaluating responses directly is often difficult because multiple failure sources become mixed together:

- tokenizer fragmentation
- missing domain knowledge in the corpus
- prompting issues
- retrieval errors
- reasoning failures

By starting from terminology-level tokenization, we can isolate one of the earliest and most fundamental failure modes.

For pharmaceutical and medical NLP, this matters because even minor tokenization instability can affect:

- spelling robustness
- embedding quality
- retrieval consistency
- token efficiency
- hallucination behavior

This repository therefore begins with:

```text
term-level tokenizer audit
```

before moving toward:

```text
phrase-level analysis
↓
document chunk analysis
↓
RAG evaluation
↓
downstream QA / hallucination benchmarks
```

The initial benchmark intentionally focuses on:

- drug names
- salt forms
- katakana terminology
- rare medical vocabulary
- long chemical names

rather than full free-form response evaluation.

The goal is not to claim downstream clinical performance, but to provide interpretable diagnostics for domain-specific tokenization behavior.


## Philosophy

This project focuses on:

- tokenizer fragmentation
- token efficiency
- Japanese medical terminology
- practical LLM deployment

It does not claim that every drug name should become a single token.

## Devil's Advocate

Adding every drug name as a single token is not always optimal.

Over-specialized vocabularies may reduce generalization to unseen compounds.
A better target is often stable semantic segmentation, for example:

```text
ロプリノン | 塩酸塩
```

rather than:

```text
ロ | プ | リ | ノ | ン | 塩 | 酸 | 塩
```

## Future Work

- Hugging Face Dataset release
- PMDA-focused corpus evaluation
- Multilingual medical tokenizer benchmark
- Tokenizer retraining experiments
- RAG chunk fragmentation analysis

## License

MIT
