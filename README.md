# RAG MVP — Ad Copy Retrieval Pipeline

A minimal RAG (Retrieval-Augmented Generation) pipeline that embeds ad copy data and enables semantic search using vector similarity.

## What It Does

1. Takes a CSV of ad copy (headlines, taglines, brands, categories, visuals)
2. Embeds each row into a 768-dimensional vector using Google Gemini
3. Stores the vectors in Supabase (Postgres + pgvector)
4. Lets you search by natural language — "playful self-care copy for a lingerie brand" returns the most relevant ads ranked by similarity

## Tools & Platforms

| Tool | Role |
|------|------|
| **Python 3** | Core language |
| **Google Gemini** (`gemini-embedding-001`) | Text embedding model (768-dim vectors) |
| **Supabase** | Postgres database with pgvector extension |
| **pgvector** | Vector similarity search in Postgres |
| **Pandas / PyArrow** | Data handling and parquet serialization |

## Project Structure

```
ad-copy-rag/
├── ad_copy_dataset.csv       # Source data (23 ad copy rows)
├── embed_ads.py              # Embed CSV → embeddings.parquet (Gemini API)
├── push_to_supabase.py       # Push parquet → Supabase + semantic search
├── embeddings.parquet        # Pre-built embeddings (768-dim)
├── requirements.txt          # Python dependencies
└── .gitignore
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export SUPABASE_URL="your-supabase-project-url"
export SUPABASE_SERVICE_KEY="your-service-role-key"
```

## Usage

### Embed the dataset

```bash
python3 embed_ads.py --csv ad_copy_dataset.csv --dry-run   # preview
python3 embed_ads.py --csv ad_copy_dataset.csv              # build embeddings.parquet
```

### Push to Supabase

```bash
python3 push_to_supabase.py --dry-run   # inspect without writing
python3 push_to_supabase.py             # upsert all rows
```

### Semantic search

```bash
# Search via Supabase (pgvector)
python3 push_to_supabase.py --query "playful self-care copy for a lingerie brand"

# Search locally against parquet
python3 embed_ads.py --query "playful self-care copy for a lingerie brand"
```

### Supabase SQL Setup

Run this in the Supabase SQL Editor before pushing:

```sql
create extension if not exists vector;

create table if not exists ad_embeddings (
  id bigint primary key,
  headline text,
  tagline text,
  brand text,
  creator text,
  category text,
  image_description text,
  embed_text text,
  embedding vector(768)
);

create or replace function match_ads(
  query_embedding vector(768),
  match_count int default 5
)
returns table (
  id bigint,
  headline text,
  brand text,
  category text,
  embed_text text,
  similarity float
)
language plpgsql
as $$
begin
  return query
    select
      a.id, a.headline, a.brand, a.category, a.embed_text,
      1 - (a.embedding <=> query_embedding) as similarity
    from ad_embeddings a
    order by a.embedding <=> query_embedding
    limit match_count;
end;
$$;
```
