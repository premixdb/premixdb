# Storage

Keep the database directory to reuse captured text, fields, selections, and packed
datasets. It contains SQLite metadata and immutable objects with verified digests.

```python
import premixdb as p

with p.PremixDB(storage=".premixdb", read_only=True) as db:
    snapshot = db.corpus("training")
    print(snapshot.preview())
```

A corpus name opens its latest successful snapshot. Existing snapshot handles stay
unchanged. Repeating a recipe reuses completed work.

Back up metadata and objects together. For a live backup, use SQLite's backup API
and preserve the object directory. Unused artifacts aren't automatically collected.

Catalog listings use a derived SQLite parent index to filter and page before
loading resource payloads. Writable opening backfills this index in older catalogs;
read-only opening continues to support catalogs without it. Blob namespace
directories appear only when objects are published. Legacy `.ref` resources and
version-1 snapshot inventories remain readable.
