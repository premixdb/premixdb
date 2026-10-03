"""Operations shared by generated resource and request messages."""

from __future__ import annotations

from google.protobuf.message import Message

from ._protobuf import copy_message


def reject_unknown(message: Message) -> None:
    known = copy_message(message)
    known.DiscardUnknownFields()
    if known != message:
        raise NotImplementedError("message contains fields unsupported by this engine")


def copy_fields[T: Message](source: Message, target: T) -> T:
    """Copy common schema fields without a second, drifting spec representation."""
    assert target.DESCRIPTOR is not None
    for descriptor, value in source.ListFields():
        if descriptor.name not in target.DESCRIPTOR.fields_by_name:
            continue
        if descriptor.is_repeated:
            dest = getattr(target, descriptor.name)
            if descriptor.message_type and descriptor.message_type.GetOptions().map_entry:
                dest.update(value)
            else:
                dest.extend(value)
        elif descriptor.message_type:
            getattr(target, descriptor.name).CopyFrom(value)
        else:
            setattr(target, descriptor.name, value)
    return target
