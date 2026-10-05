#!/usr/bin/env python3
"""
Push embeddings.parquet → Supabase pgvector table, with search.

Setup:
    export SUPABASE_URL="https://your-project.supabase.co"
    export SUPABASE_SERVICE_KEY="your-service-role-key"
    export GEMINI_API_KEY="your-key"          # only needed for --query

Usage:
    python3 push_to_supabase.py --dry-run     # inspect without writing
    python3 push_to_supabase.py               # upsert all rows
    python3 push_to_supabase.py --query "playful self-care copy for a lingerie brand"
"""

import argparse
import json
import math
import os
import sys

import pandas as pd

TABLE = "ad_embeddings"
PARQUET = "embeddings.parquet"
DIMS = 768
GEMINI_MODEL = "gemini-embedding-001"


def get_supabase():
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars")
    return create_client(url, key)


def load_parquet(path):
    df = pd.read_parquet(path)
    if "embedding" not in df.columns:
        sys.exit(f"No 'embedding' column in {path}")
    return df


def dry_run(args):
    df = load_parquet(args.parquet)
    print(f"Parquet: {args.parquet}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    emb = df["embedding"].iloc[0]
    print(f"  Embedding dim: {len(emb)} (type: {type(emb).__name__})")
    print(f"  Target table: {TABLE}")
    print(f"\nSample row:")
    row = df.iloc[0]
    for col in ["id", "brand", "headline", "category", "embed_text"]:
        if col in df.columns:
            print(f"  {col}: {row[col]}")
    print(f"  embedding: [{emb[0]:.6f}, {emb[1]:.6f}, ... {emb[-1]:.6f}]")
    print(f"\nDry run — nothing pushed. Remove --dry-run to upsert {len(df)} rows.")


def push(args):
    df = load_parquet(args.parquet)
    sb = get_supabase()
    rows = []
    for _, r in df.iterrows():
        emb = r["embedding"]
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        rows.append({
            "id": int(r["id"]),
            "headline": r.get("headline", ""),
            "tagline": r.get("tagline", ""),
            "brand": r.get("brand", ""),
            "creator": r.get("creator", ""),
            "category": r.get("category", ""),
            "image_description": r.get("image_description", ""),
            "embed_text": r.get("embed_text", ""),
            "embedding": emb,
        })

    batch_size = args.batch
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        sb.table(TABLE).upsert(batch).execute()
        print(f"  upserted {min(i + batch_size, len(rows))}/{len(rows)}")

    print(f"\nDone — {len(rows)} rows in '{TABLE}'")


def query(args):
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    client = genai.Client(api_key=key)

    res = client.models.embed_content(
        model=GEMINI_MODEL,
        contents=[args.query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY", output_dimensionality=DIMS
        ),
    )
    qvec = res.embeddings[0].values
    n = math.sqrt(sum(v * v for v in qvec))
    qvec = [v / n for v in qvec] if n else qvec

    sb = get_supabase()
    result = sb.rpc(
        "match_ads",
        {"query_embedding": qvec, "match_count": args.k},
    ).execute()

    print(f'\nTop {args.k} for: "{args.query}"\n')
    for r in result.data:
        print(f"  {r['similarity']:.3f}  [{r.get('brand', '—')}] {r.get('headline', '')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default=PARQUET)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--query", help="semantic search")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--batch", type=int, default=50)
    a = p.parse_args()

    if a.dry_run:
        dry_run(a)
    elif a.query:
        query(a)
    else:
        push(a)
