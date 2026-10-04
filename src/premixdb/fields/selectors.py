"""Convert field projections and classifier namespaces into schema selectors."""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol, runtime_checkable

from premixdb.fields import (
    ContentType,
    ContentTypeFields,
    Language,
    LanguageFields,
    Topic,
    TopicFields,
    content_type,
    language,
    topic,
)
from premixdb.fields.expressions import FieldProjection, VectorField
from premixdb.fields.ids import field_id
from premixdb.schemas.protobuf import copy_message
from premixdb.v1 import data_mixture_pb2 as mix_pb
from premixdb.v1 import query_pb2 as q


@runtime_checkable
class ClassifierProjection(Protocol):
    @property
    def label(self) -> FieldProjection: ...


type FieldSelector = (
    str
    | q.FieldComparison
    | FieldProjection
    | VectorField
    | ClassifierProjection
    | type[Topic]
    | type[ContentType]
    | type[Language]
)


def selector(field: FieldSelector) -> q.FieldComparison:
    if isinstance(field, q.FieldComparison):
        return copy_message(field)
    if field is Topic:
        field = topic.label
    elif field is ContentType:
        field = content_type.label
    elif field is Language:
        field = language.label
    # A classifier namespace naturally selects its label; individual projections
    # remain available for explicit scalar probabilities and multi-field strata.
    if isinstance(field, ClassifierProjection):
        field = field.label
    if not isinstance(field, (str, FieldProjection, VectorField)):
        raise TypeError("expected a field projection or classifier namespace")
    name = field if isinstance(field, str) else field.name
    if not isinstance(name, str):
        raise TypeError("expected a field projection with a string name")
    result = q.FieldComparison(field=field_id(name), projection=q.FieldComparison.SCALAR)
    if isinstance(field, FieldProjection):
        result.projection = field.projection
        result.class_name = field.class_name
        if field.component_index is not None:
            result.component = field.component_index
    return result


DomainInput = (
    type[Topic]
    | type[ContentType]
    | type[Language]
    | FieldProjection
    | LanguageFields
    | TopicFields
    | ContentTypeFields
    | Iterable[FieldProjection]
    | mix_pb.Domains
    | Mapping[str, str]
)


type ProfileSelector = str | FieldProjection | VectorField | q.FieldComparison
