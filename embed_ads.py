#!/usr/bin/env python3
"""
Embed an ad-copy CSV with Google's Gemini embedding API -> embeddings.parquet

Setup (Kaggle / Colab / local):
    pip install google-genai pandas pyarrow
    export GEMINI_API_KEY="your-key"        # Kaggle: use Secrets, see note at bottom

Build the index:
    python embed_ads.py --csv ad_copy_dataset.csv
    python embed_ads.py --csv ad_copy_dataset.csv --dry-run     # no API calls, check the text first

Query it (sanity check that retrieval actually works):
    python embed_ads.py --query "playful self-care copy for a lingerie brand"
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

MODEL = "gemini-embedding-001"
DIMS = 768          # 768 / 1536 / 3072 are the recommended sizes
BATCH = 50          # rows per API request
DOC_TASK = "RETRIEVAL_DOCUMENT"   # how the dataset rows are embedded
QUERY_TASK = "RETRIEVAL_QUERY"    # how a user's prompt is embedded


# ---------------------------------------------------------------- text builder
def build_text(row: pd.Series) -> str:
    """One compact string per ad. This is the single most important choice in the
    whole pipeline: the model only ever sees this text, so anything you leave
    out is invisible to retrieval later."""
    parts = [
        f"Brand: {row.get('brand') or 'unknown'}",
        f"Category: {row.get('category') or 'unknown'}",
        f"Headline: {row.get('headline') or ''}",
    ]
    if str(row.get("tagline") or "").strip():
        parts.append(f"Tagline: {row['tagline']}")
    if str(row.get("image_description") or "").strip():
        parts.append(f"Visual: {row['image_description']}")
    if str(row.get("creator") or "").strip():
        parts.append(f"Creator: {row['creator']}")
    return " | ".join(parts)


def l2_normalize(vec):
    """Required for gemini-embedding-001 at any dimension other than 3072,
    otherwise cosine similarity is distorted by vector magnitude."""
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec] if n else vec


# ---------------------------------------------------------------- api plumbing
def get_client():
    from google import genai

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set. Get a free key at aistudio.google.com/apikey")
    return genai.Client(api_key=key)


def embed_batch(client, texts, task, dims, model, attempts=5):
    """One request, many texts. Retries with exponential backoff on 429/5xx,
    which is what free-tier rate limits look like."""
    from google.genai import types

    for attempt in range(attempts):
        try:
            res = client.models.embed_content(
                model=model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=dims
                ),
            )
            return [l2_normalize(e.values) for e in res.embeddings]
        except Exception as err:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  ! {type(err).__name__}: retrying in {wait}s", flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------- build / query
def build(args):
    df = pd.read_csv(args.csv).fillna("")
    if args.limit:
        df = df.head(args.limit)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    df["embed_text"] = df.apply(build_text, axis=1)
    print(f"{len(df)} rows -> {math.ceil(len(df) / args.batch)} requests "
          f"| model={args.model} dims={args.dims}")
    print("\nSample of what the model will actually see:")
    for t in df["embed_text"].head(2):
        print("  " + t[:180] + ("..." if len(t) > 180 else ""))
    print()

    if args.dry_run:
        print("Dry run — no API calls made, no file written.")
        return

    # Checkpoint file: re-running skips rows already embedded, so a rate-limit
    # error halfway through 1000 rows never costs you the work already done.
    cache_path = Path(args.out).with_suffix(".cache.jsonl")
    done = {}
    if cache_path.exists():
        for line in cache_path.open():
            rec = json.loads(line)
            done[str(rec["id"])] = rec["embedding"]
        print(f"Resuming: {len(done)} rows already in {cache_path.name}")

    client = get_client()
    todo = df[~df["id"].astype(str).isin(done)]

    with cache_path.open("a") as cache:
        for start in range(0, len(todo), args.batch):
            chunk = todo.iloc[start:start + args.batch]
            vectors = embed_batch(
                client, chunk["embed_text"].tolist(), DOC_TASK, args.dims, args.model
            )
            for row_id, vec in zip(chunk["id"], vectors):
                cache.write(json.dumps({"id": row_id, "embedding": vec}) + "\n")
            cache.flush()
            done.update({str(i): v for i, v in zip(chunk["id"], vectors)})
            print(f"  embedded {min(start + args.batch, len(todo))}/{len(todo)}", flush=True)
            time.sleep(args.sleep)

    df["embedding"] = df["id"].astype(str).map(done)
    df.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out} — {len(df)} rows, {len(df['embedding'].iloc[0])} dims each")
    print(f"Checkpoint kept at {cache_path.name} (safe to delete once you're happy)")


def query(args):
    df = pd.read_parquet(args.out)
    client = get_client()
    qvec = embed_batch(client, [args.query], QUERY_TASK, args.dims, args.model)[0]

    def cosine(v):
        return sum(a * b for a, b in zip(qvec, v))  # both sides are normalized

    df["score"] = df["embedding"].apply(cosine)
    top = df.nlargest(args.k, "score")
    print(f'\nTop {args.k} for: "{args.query}"\n')
    for _, r in top.iterrows():
        brand = r.get("brand") or "—"
        print(f"  {r['score']:.3f}  [{brand}] {r.get('headline', '')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="ad_copy_dataset.csv")
    p.add_argument("--out", default="embeddings.parquet")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--dims", type=int, default=DIMS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--sleep", type=float, default=1.0, help="pause between requests")
    p.add_argument("--limit", type=int, help="only embed the first N rows (test run)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--query", help="search an existing embeddings.parquet")
    p.add_argument("-k", type=int, default=5)
    a = p.parse_args()
    query(a) if a.query else build(a)

# Kaggle note: instead of an env var, add the key under Add-ons > Secrets, then
#   from kaggle_secrets import UserSecretsClient
#   os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
# before calling build().
