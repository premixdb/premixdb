# Development

Python 3.12+ and uv 0.12.20+.

Run the checkout with `make shell` (`uvx --from . premixdb shell`).
`make install` installs it as an editable tool so `premixdb shell` follows local
changes. Neither command needs the development environment; checks use the locked
project environment below.

```bash
make sync
make lint
make typecheck
uv run --locked pytest tests/test_cli.py
```

| Command | Check |
| --- | --- |
| `make shell` | Run the local checkout through uvx |
| `make install` | Install the local CLI as an editable tool |
| `make sync` | Install locked development dependencies |
| `make lint` | Ruff and formatting |
| `make typecheck` | ty across library, scripts, examples, and tests |
| `make test` | Fast tests with bounded fixtures |
| `make test-all` | Fast tests plus integration checks |
| `make integration` | Package, example, and process checks |
| `make coverage` | All non-benchmark tests; branch coverage, 85% minimum |
| `make protos` | Generated bindings match schemas |
| `make build` | Fresh wheel, sdist, and metadata in `reports/dist` |
| `make publish` | Build, validate, and upload to PyPI |
| `make audit` | Locked runtime dependencies |
| `make benchmark` | Local CPU benchmarks |

`make check` runs the full quality suite. Local iteration can use individual
commands. There are no commit or push hooks.

Create a [PyPI API token](https://pypi.org/help/#apitoken), then set
`UV_PUBLISH_TOKEN` locally before `make publish`. Your PyPI account password cannot
be used to upload packages. The target checks for the token before building, ignores
`UV_PUBLISH_USERNAME` and `UV_PUBLISH_PASSWORD`, and validates fresh distributions
before uploading them with token authentication.

`pytest` defaults to fast tests. Use `-m integration` for integration checks or
`-m 'not performance'` for both suites. Benchmarks run only through `make benchmark`.
Tests use up to six CPU workers, keeping each test file together. Use `pytest -n 0`
for serial runs and debugging, or `pytest -n 2` to use fewer workers. Benchmarks
always run serially. Native OpenMP, BLAS, and Rayon pools default to one thread
per process in tests, including spawned loader workers; explicit environment
settings override those defaults. Heavy libraries load only in tests that need
them, rather than during collection on every worker.
When integration checks are selected, files with expensive process/build checks
start early so they do not hold up the end of the run. Fast-only runs schedule
larger files first; order within each file is preserved.
Corpus fixtures use one or two documents unless a test needs a specific boundary
or additional distinct roles. Boundary fixtures use the smallest input that crosses
the boundary under test. Tests of validation, display, or storage use byte tokens;
dedicated tests cover GPT-2 defaults and model-token alignment. Decoder tests use
the small bundled WordPiece fixture when they do not need GPT-2 behavior.
All package build modes run fresh-process capture/query/byte-token checks. One
installed-wheel check additionally runs the full GPT-2/PyTorch workflow.

Writable query handles wait for local job completion directly. Polling remains a
fallback when no local job is active. Wait deadlines do not cancel the job;
completed publication can be read on a later call.

Use concrete types, bounded generics, and protocols. Ruff rejects `Any` and
missing annotations; ty checks assignments, returns, yields, and generic arguments.
Validate untyped data at its boundary.

CLI tests use a two-document corpus. README examples run with offline Hub/model
fixtures; packaging tests exercise an installed wheel. Generated protobuf bindings
are ignored build outputs; edit schemas and regenerate as described [here](../proto/README.md).

`intrinsic.proto` also generates the tracked `fields/catalog.py` source, including
statically typed language attributes and compatibility enum names. Regenerate it
with `scripts/generate_protos.py`; `make protos` checks both the catalog and local
protobuf bindings. Numeric field IDs remain explicit in the schema.

The independent engine reference harness lives in `tests/_reference.py`. It is
excluded from wheels; application workflows use the main `PremixDB` API.

See [architecture](architecture.md) for package responsibilities and import
boundaries. Implementation modules use descriptive filenames inside those
packages; retain the top-level public exports when reorganizing internals.
