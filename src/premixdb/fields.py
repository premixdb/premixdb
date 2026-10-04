"""Public typed field catalog, aligned with upstream model label vocabularies.

fastText: https://fasttext.cc/docs/en/language-identification.html
WebOrganizer: TopicClassifier and FormatClassifier config.id2label.
Keyword language codes use trailing underscores: language.as_, language.is_, language.or_.
"""

from enum import Enum
from typing import Literal, overload

from ._field_catalog import Language as Language
from ._field_catalog import LanguageFields as LanguageFields
from ._field_expr import ScalarField, VectorField
from .v1.query_pb2 import FieldComparison


class Topic(str, Enum):
    ADULT = "Adult"
    ART_AND_DESIGN = "Art & Design"
    SOFTWARE_DEV = "Software Dev."
    CRIME_AND_LAW = "Crime & Law"
    EDUCATION_AND_JOBS = "Education & Jobs"
    HARDWARE = "Hardware"
    ENTERTAINMENT = "Entertainment"
    SOCIAL_LIFE = "Social Life"
    FASHION_AND_BEAUTY = "Fashion & Beauty"
    FINANCE_AND_BUSINESS = "Finance & Business"
    FOOD_AND_DINING = "Food & Dining"
    GAMES = "Games"
    HEALTH = "Health"
    HISTORY = "History"
    HOME_AND_HOBBIES = "Home & Hobbies"
    INDUSTRIAL = "Industrial"
    LITERATURE = "Literature"
    POLITICS = "Politics"
    RELIGION = "Religion"
    SCIENCE_AND_TECH = "Science & Tech."
    SOFTWARE = "Software"
    SPORTS_AND_FITNESS = "Sports & Fitness"
    TRANSPORTATION = "Transportation"
    TRAVEL = "Travel"


class ContentType(str, Enum):
    ACADEMIC_WRITING = "Academic Writing"
    CONTENT_LISTING = "Content Listing"
    CREATIVE_WRITING = "Creative Writing"
    CUSTOMER_SUPPORT = "Customer Support"
    COMMENT_SECTION = "Comment Section"
    FAQ = "FAQ"
    TRUNCATED = "Truncated"
    KNOWLEDGE_ARTICLE = "Knowledge Article"
    LEGAL_NOTICES = "Legal Notices"
    LISTICLE = "Listicle"
    NEWS_ARTICLE = "News Article"
    NONFICTION_WRITING = "Nonfiction Writing"
    ABOUT_ORG = "About (Org.)"
    NEWS_ORG = "News (Org.)"
    ABOUT_PERS = "About (Pers.)"
    PERSONAL_BLOG = "Personal Blog"
    PRODUCT_PAGE = "Product Page"
    Q_AND_A_FORUM = "Q&A Forum"
    SPAM_ADS = "Spam / Ads"
    STRUCTURED_DATA = "Structured Data"
    DOCUMENTATION = "Documentation"
    AUDIO_TRANSCRIPT = "Audio Transcript"
    TUTORIAL = "Tutorial"
    USER_REVIEW = "User Review"


class Quality(str, Enum):
    WRITING_STYLE = "writing_style"
    REQUIRED_EXPERTISE = "required_expertise"
    FACTS_AND_TRIVIA = "facts_and_trivia"
    EDUCATIONAL_VALUE = "educational_value"


class DataTroveMetric(str, Enum):
    LENGTH = "length"
    WHITE_SPACE_RATIO = "white_space_ratio"
    NON_ALPHA_DIGIT_RATIO = "non_alpha_digit_ratio"
    DIGIT_RATIO = "digit_ratio"
    UPPERCASE_RATIO = "uppercase_ratio"
    ELIPSIS_RATIO = "elipsis_ratio"
    PUNCTUATION_RATIO = "punctuation_ratio"
    N_WORDS = "n_words"
    AVG_WORD_LENGTH = "avg_word_length"
    AVG_WORDS_PER_LINE = "avg_words_per_line"
    SHORT_WORD_RATIO_3 = "short_word_ratio_3"
    LONG_WORD_RATIO_7 = "long_word_ratio_7"
    TYPE_TOKEN_RATIO = "type_token_ratio"
    UPPERCASE_WORD_RATIO = "uppercase_word_ratio"
    CAPITALIZED_WORD_RATIO = "capitalized_word_ratio"
    STOP_WORD_RATIO = "stop_word_ratio"


class EmbeddingModel(str, Enum):
    HARRIER = "harrier"


class DedupeIndex(str, Enum):
    EXACT_DOCUMENT = "dupekit.exact_candidates"
    MINHASH_LSH = "dupekit.lsh"


def _class_probability(label: Topic | ContentType) -> ScalarField[float]:
    name = "weborganizer.topic" if isinstance(label, Topic) else "weborganizer.content_type"
    return ScalarField(name, float, FieldComparison.CLASS_PROBABILITY, class_name=label.value)


class TopicFields:
    adult: ScalarField[float] = _class_probability(Topic.ADULT)
    art_and_design: ScalarField[float] = _class_probability(Topic.ART_AND_DESIGN)
    software_dev: ScalarField[float] = _class_probability(Topic.SOFTWARE_DEV)
    crime_and_law: ScalarField[float] = _class_probability(Topic.CRIME_AND_LAW)
    education_and_jobs: ScalarField[float] = _class_probability(Topic.EDUCATION_AND_JOBS)
    hardware: ScalarField[float] = _class_probability(Topic.HARDWARE)
    entertainment: ScalarField[float] = _class_probability(Topic.ENTERTAINMENT)
    social_life: ScalarField[float] = _class_probability(Topic.SOCIAL_LIFE)
    fashion_and_beauty: ScalarField[float] = _class_probability(Topic.FASHION_AND_BEAUTY)
    finance_and_business: ScalarField[float] = _class_probability(Topic.FINANCE_AND_BUSINESS)
    food_and_dining: ScalarField[float] = _class_probability(Topic.FOOD_AND_DINING)
    games: ScalarField[float] = _class_probability(Topic.GAMES)
    health: ScalarField[float] = _class_probability(Topic.HEALTH)
    history: ScalarField[float] = _class_probability(Topic.HISTORY)
    home_and_hobbies: ScalarField[float] = _class_probability(Topic.HOME_AND_HOBBIES)
    industrial: ScalarField[float] = _class_probability(Topic.INDUSTRIAL)
    literature: ScalarField[float] = _class_probability(Topic.LITERATURE)
    politics: ScalarField[float] = _class_probability(Topic.POLITICS)
    religion: ScalarField[float] = _class_probability(Topic.RELIGION)
    science_and_tech: ScalarField[float] = _class_probability(Topic.SCIENCE_AND_TECH)
    software: ScalarField[float] = _class_probability(Topic.SOFTWARE)
    sports_and_fitness: ScalarField[float] = _class_probability(Topic.SPORTS_AND_FITNESS)
    transportation: ScalarField[float] = _class_probability(Topic.TRANSPORTATION)
    travel: ScalarField[float] = _class_probability(Topic.TRAVEL)
    label: ScalarField[Topic] = ScalarField("weborganizer.topic", Topic, FieldComparison.TOP_CLASS)
    logits: VectorField = VectorField("weborganizer.topic")

    def __getitem__(self, label: Topic) -> ScalarField[float]:
        if not isinstance(label, Topic):
            raise TypeError("expected Topic")
        return _class_probability(label)


class ContentTypeFields:
    academic_writing: ScalarField[float] = _class_probability(ContentType.ACADEMIC_WRITING)
    content_listing: ScalarField[float] = _class_probability(ContentType.CONTENT_LISTING)
    creative_writing: ScalarField[float] = _class_probability(ContentType.CREATIVE_WRITING)
    customer_support: ScalarField[float] = _class_probability(ContentType.CUSTOMER_SUPPORT)
    comment_section: ScalarField[float] = _class_probability(ContentType.COMMENT_SECTION)
    faq: ScalarField[float] = _class_probability(ContentType.FAQ)
    truncated: ScalarField[float] = _class_probability(ContentType.TRUNCATED)
    knowledge_article: ScalarField[float] = _class_probability(ContentType.KNOWLEDGE_ARTICLE)
    legal_notices: ScalarField[float] = _class_probability(ContentType.LEGAL_NOTICES)
    listicle: ScalarField[float] = _class_probability(ContentType.LISTICLE)
    news_article: ScalarField[float] = _class_probability(ContentType.NEWS_ARTICLE)
    nonfiction_writing: ScalarField[float] = _class_probability(ContentType.NONFICTION_WRITING)
    about_org: ScalarField[float] = _class_probability(ContentType.ABOUT_ORG)
    news_org: ScalarField[float] = _class_probability(ContentType.NEWS_ORG)
    about_pers: ScalarField[float] = _class_probability(ContentType.ABOUT_PERS)
    personal_blog: ScalarField[float] = _class_probability(ContentType.PERSONAL_BLOG)
    product_page: ScalarField[float] = _class_probability(ContentType.PRODUCT_PAGE)
    q_and_a_forum: ScalarField[float] = _class_probability(ContentType.Q_AND_A_FORUM)
    spam_ads: ScalarField[float] = _class_probability(ContentType.SPAM_ADS)
    structured_data: ScalarField[float] = _class_probability(ContentType.STRUCTURED_DATA)
    documentation: ScalarField[float] = _class_probability(ContentType.DOCUMENTATION)
    audio_transcript: ScalarField[float] = _class_probability(ContentType.AUDIO_TRANSCRIPT)
    tutorial: ScalarField[float] = _class_probability(ContentType.TUTORIAL)
    user_review: ScalarField[float] = _class_probability(ContentType.USER_REVIEW)
    label: ScalarField[ContentType] = ScalarField(
        "weborganizer.content_type", ContentType, FieldComparison.TOP_CLASS
    )
    logits: VectorField = VectorField("weborganizer.content_type")

    def __getitem__(self, label: ContentType) -> ScalarField[float]:
        if not isinstance(label, ContentType):
            raise TypeError("expected ContentType")
        return _class_probability(label)


class QualityFields:
    writing_style: ScalarField[float] = ScalarField("quality.writing_style", float)
    required_expertise: ScalarField[float] = ScalarField("quality.required_expertise", float)
    facts_and_trivia: ScalarField[float] = ScalarField("quality.facts_and_trivia", float)
    educational_value: ScalarField[float] = ScalarField("quality.educational_value", float)

    def __getitem__(self, label: Quality) -> ScalarField[float]:
        if not isinstance(label, Quality):
            raise TypeError("expected Quality")
        return ScalarField("quality." + label.value, float)


class DataTroveFields:
    length: ScalarField[int] = ScalarField("datatrove.length", int)
    white_space_ratio: ScalarField[float] = ScalarField("datatrove.white_space_ratio", float)
    non_alpha_digit_ratio: ScalarField[float] = ScalarField(
        "datatrove.non_alpha_digit_ratio", float
    )
    digit_ratio: ScalarField[float] = ScalarField("datatrove.digit_ratio", float)
    uppercase_ratio: ScalarField[float] = ScalarField("datatrove.uppercase_ratio", float)
    elipsis_ratio: ScalarField[float] = ScalarField("datatrove.elipsis_ratio", float)
    punctuation_ratio: ScalarField[float] = ScalarField("datatrove.punctuation_ratio", float)
    n_words: ScalarField[int] = ScalarField("datatrove.n_words", int)
    avg_word_length: ScalarField[float] = ScalarField("datatrove.avg_word_length", float)
    avg_words_per_line: ScalarField[float] = ScalarField("datatrove.avg_words_per_line", float)
    short_word_ratio_3: ScalarField[float] = ScalarField("datatrove.short_word_ratio_3", float)
    long_word_ratio_7: ScalarField[float] = ScalarField("datatrove.long_word_ratio_7", float)
    type_token_ratio: ScalarField[float] = ScalarField("datatrove.type_token_ratio", float)
    uppercase_word_ratio: ScalarField[float] = ScalarField("datatrove.uppercase_word_ratio", float)
    capitalized_word_ratio: ScalarField[float] = ScalarField(
        "datatrove.capitalized_word_ratio", float
    )
    stop_word_ratio: ScalarField[float] = ScalarField("datatrove.stop_word_ratio", float)

    @overload
    def __getitem__(
        self, label: Literal[DataTroveMetric.LENGTH, DataTroveMetric.N_WORDS]
    ) -> ScalarField[int]: ...

    @overload
    def __getitem__(
        self,
        label: Literal[
            DataTroveMetric.WHITE_SPACE_RATIO,
            DataTroveMetric.NON_ALPHA_DIGIT_RATIO,
            DataTroveMetric.DIGIT_RATIO,
            DataTroveMetric.UPPERCASE_RATIO,
            DataTroveMetric.ELIPSIS_RATIO,
            DataTroveMetric.PUNCTUATION_RATIO,
            DataTroveMetric.AVG_WORD_LENGTH,
            DataTroveMetric.AVG_WORDS_PER_LINE,
            DataTroveMetric.SHORT_WORD_RATIO_3,
            DataTroveMetric.LONG_WORD_RATIO_7,
            DataTroveMetric.TYPE_TOKEN_RATIO,
            DataTroveMetric.UPPERCASE_WORD_RATIO,
            DataTroveMetric.CAPITALIZED_WORD_RATIO,
            DataTroveMetric.STOP_WORD_RATIO,
        ],
    ) -> ScalarField[float]: ...

    @overload
    def __getitem__(self, label: DataTroveMetric) -> ScalarField[int] | ScalarField[float]: ...

    def __getitem__(self, label: DataTroveMetric) -> ScalarField[int] | ScalarField[float]:
        if not isinstance(label, DataTroveMetric):
            raise TypeError("expected DataTroveMetric")
        if label in (DataTroveMetric.LENGTH, DataTroveMetric.N_WORDS):
            return ScalarField("datatrove." + label.value, int)
        return ScalarField("datatrove." + label.value, float)


class EmbeddingFields:
    harrier: VectorField = VectorField("embedding.harrier")

    def __getitem__(self, model: EmbeddingModel) -> VectorField:
        if not isinstance(model, EmbeddingModel):
            raise TypeError("expected EmbeddingModel")
        return VectorField("embedding." + model.value)


language = LanguageFields()
topic = TopicFields()
content_type = ContentTypeFields()
quality = QualityFields()
datatrove = DataTroveFields()
embedding = EmbeddingFields()
