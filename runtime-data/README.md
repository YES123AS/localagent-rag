# Local runtime data

This directory is reserved for machine-local secrets and generated state. Only
this README is committed; every other file below `runtime-data/` is ignored by
Git and excluded from the Docker build context.

At first use, copy `../.env.example` to `.env`. Docker Compose creates and uses
the following local directories automatically:

- `app/` — SQLite conversations, runs, and caches
- `qdrant/` — vector database storage
- `phoenix/` — local tracing data
- `model-cache/` — downloaded Hugging Face models

Delete the generated directories only when you intentionally want to reset all
local application data and rebuild the vector index.
