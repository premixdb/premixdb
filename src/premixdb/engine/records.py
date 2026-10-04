"""Captured-document storage contract shared by snapshots and saved selections."""

from __future__ import annotations

from typing import Iterable

from .._typing import JSON, json_integer, json_integers, json_list, json_object, json_string
from .contracts import DocumentRecord, Frame, TextProfile

FRAME_BYTES = 1024 * 1024


def schema(value: object, required: Iterable[str], optional: Iterable[str] = ()) -> None:
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise RuntimeError("invalid stored schema")


def totals(value: object, names: Iterable[str]) -> None:
    schema(value, names)
    assert isinstance(value, dict)
    if any(type(v) is not int or not 0 <= v < 2**64 for v in value.values()):
        raise RuntimeError("invalid stored counts")


def text_profile(data: bytes) -> TextProfile:
    text = data.decode()
    return TextProfile(content_bytes=len(data), characters=len(text), newlines=text.count("\n"))


def decode_profile(raw: JSON) -> TextProfile:
    obj = json_object(raw)
    totals(obj, ("content_bytes", "characters", "newlines"))
    return TextProfile(
        content_bytes=json_integer(obj["content_bytes"]),
        characters=json_integer(obj["characters"]),
        newlines=json_integer(obj["newlines"]),
    )


def decode_frame(raw: JSON) -> Frame:
    obj = json_object(raw)
    schema(obj, ("digest", "bytes"), ("profile",))
    result = Frame(digest=json_integers(obj["digest"]), bytes=json_integer(obj["bytes"]))
    if "profile" in obj:
        result["profile"] = decode_profile(obj["profile"])
    return result


def decode_document(raw: dict[str, JSON]) -> DocumentRecord:
    schema(raw, ("key", "content", "bytes", "frames"))
    return DocumentRecord(
        key=json_string(raw["key"]),
        content=json_integers(raw["content"]),
        bytes=json_integer(raw["bytes"]),
        frames=[decode_frame(f) for f in json_list(raw["frames"])],
    )


def validate_document(record: DocumentRecord, *, require_profiles: bool = False) -> None:
    """Validate structure and frame coverage without loading captured text."""
    schema(record, ("key", "content", "bytes", "frames"))
    if (
        len(bytes(record["content"])) != 32
        or type(record["bytes"]) is not int
        or not 0 <= record["bytes"] < 2**64
    ):
        raise RuntimeError("invalid stored document")
    for frame in record["frames"]:
        schema(
            frame,
            ("digest", "bytes", "profile") if require_profiles else ("digest", "bytes"),
            () if require_profiles else ("profile",),
        )
        if (
            len(bytes(frame["digest"])) != 32
            or type(frame["bytes"]) is not int
            or not 0 < frame["bytes"] <= 8 * FRAME_BYTES
        ):
            raise RuntimeError("invalid frame size or digest")
        if "profile" in frame:
            totals(frame["profile"], ("content_bytes", "characters", "newlines"))
            p = frame["profile"]
            if (
                p["content_bytes"] != frame["bytes"]
                or not p["newlines"] <= p["characters"] <= p["content_bytes"]
            ):
                raise RuntimeError("invalid text profile")
    if sum(frame["bytes"] for frame in record["frames"]) != record["bytes"]:
        raise RuntimeError("stored document has incomplete frame coverage")
