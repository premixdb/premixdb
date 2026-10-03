"""Validated provenance records and explicit public identity conversion."""

from __future__ import annotations

from ._ids import _encode_id
from ._typing import checked_record, json_integer, json_list, json_object, load_json
from .engine.contracts import Provenance


def decode_lineage(data: bytes, *, public: bool = False) -> dict[str, Provenance]:
    """Read stored lineage once, normalizing ranges and optionally exposing public IDs."""
    result: dict[str, Provenance] = {}
    for identity, raw in json_object(load_json(data)).items():
        origin = json_object(raw)
        normalized: dict[str, object] = dict(origin)
        if "retained_ranges" in origin:
            ranges = []
            for pair in json_list(origin["retained_ranges"]):
                values = json_list(pair)
                if len(values) != 2:
                    raise ValueError("invalid retained byte range")
                ranges.append((json_integer(values[0]), json_integer(values[1])))
            normalized["retained_ranges"] = ranges
        record = checked_record(normalized, Provenance)
        if public:
            identity = _public_id(identity)
            _public_record(record)
        result[identity] = record
    return result


def _public_id(identity: str) -> str:
    return _encode_id(bytes.fromhex(identity))


def _public_record(record: Provenance) -> None:
    record["corpus_id"] = _public_id(record["corpus_id"])
    record["snapshots"] = [_public_id(identity) for identity in record["snapshots"]]
    selection = record["selection"]
    kept = selection.get("kept")
    if isinstance(kept, str):
        selection["kept"] = _public_id(kept)
    elif kept is not None:
        kept["document"] = _public_id(kept["document"])
    if "matched" in selection:
        matched = selection["matched"]
        matched["document"] = _public_id(matched["document"])
    for witnesses in (record.get("contamination", []), selection.get("references", [])):
        for witness in witnesses:
            witness["reference"] = _public_id(witness["reference"])
