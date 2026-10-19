# LexiLite Data Policy

What the API stores about your documents, why, for how long, and how to remove it.

## What is stored

Nothing, by default. A document analysed via `/api/documents/analyze/` or `/upload/` is read in memory, the result is returned, and nothing is written to the database.

Only if the request sets `save` to true is one `AnalysisResult` row created:

| Field | Contents |
|---|---|
| `owner` | The account that ran the analysis |
| `filename` | The name you sent (truncated to 255 chars) |
| `clauses` | **The full text of every clause extracted from the document** (first 100 clauses) |
| `analysis` | Per-clause risk results, **including the clause text again** for the risky ones |
| `processing_time_ms`, `model_used`, timestamps | Operational metadata |

**Not stored:** the uploaded PDF/DOCX file itself. It is parsed in memory and discarded; nothing is written to disk. Clauses beyond the first 100 are not analyzed or stored.

**Why:** so you can look up past results (`/api/documents/recent/`, retrieve by id).

**Logs:** the server logs a document id, character count and timing for each analysis. Document text is never logged. A failed upload logs its filename.

**Access:** each user can only see, retrieve or delete their own rows. There is no admin-facing API for reading other users' analyses (a Django superuser with database access can, as with any application).

## How long it is kept

Analyses are automatically deleted **30 days** after creation.

- Configure with the `RETENTION_DAYS` environment variable (`0` disables automatic deletion).
- Deletion is performed by a management command that must be scheduled (daily is plenty):

```bash
python manage.py purge_expired_analyses             # uses RETENTION_DAYS
python manage.py purge_expired_analyses --dry-run   # report only
python manage.py purge_expired_analyses --days 7    # override for one run
```

Nothing deletes data automatically unless this command is scheduled (cron on Linux, Task Scheduler on Windows).

## Deleting your data

```bash
# One analysis (id from /recent/ or the list endpoint)
curl -X DELETE http://localhost:8000/api/documents/<id>/ -H "Authorization: Token <key>"

# Everything you have stored
curl -X DELETE http://localhost:8000/api/documents/delete-all/ -H "Authorization: Token <key>"
```

Deletion is immediate and permanent. Other users' data is never touched (their ids return 404).

## Encryption

- **In transit:** the app sets secure cookies, HSTS and an HTTPS redirect when `DEBUG` is off, but TLS itself must be terminated by a reverse proxy (Nginx etc.) in front. Local development runs plain HTTP.
- **At rest: NOT encrypted at the application level.** Clause text is stored as plaintext in the database (currently SQLite). Protection today depends on the host: use full-disk encryption (BitLocker / FileVault / LUKS) on any machine holding real documents, and restrict access to `db.sqlite3`. Field-level encryption was considered and deferred: it would break nothing today but adds key-management complexity that makes more sense together with the PostgreSQL migration (LEX-7), where encrypted storage can be provided by the database or hosting layer.
- **Backups** (including the `.bak` files and any copies of the database) contain the same plaintext and are not covered by the retention command. Delete them on the same schedule.

## Known limits

- Retention covers the database only, not backups or copies you make.
- Users cannot yet export their data in bulk before deleting.
- Deleting an account deletes its analyses (cascade), but there is no self-service account-deletion endpoint yet.
