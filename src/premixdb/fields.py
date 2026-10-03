"""Public typed field catalog, aligned with upstream model label vocabularies.

fastText: https://fasttext.cc/docs/en/language-identification.html
WebOrganizer: TopicClassifier and FormatClassifier config.id2label.
Keyword language codes use trailing underscores: language.as_, language.is_, language.or_.
"""

from enum import Enum
from typing import Literal, overload

from ._field_expr import ScalarField, VectorField
from .v1.query_pb2 import FieldComparison


class Language(str, Enum):
    AF = "af"
    ALS = "als"
    AM = "am"
    AN = "an"
    AR = "ar"
    ARZ = "arz"
    AS = "as"
    AST = "ast"
    AV = "av"
    AZ = "az"
    AZB = "azb"
    BA = "ba"
    BAR = "bar"
    BCL = "bcl"
    BE = "be"
    BG = "bg"
    BH = "bh"
    BN = "bn"
    BO = "bo"
    BPY = "bpy"
    BR = "br"
    BS = "bs"
    BXR = "bxr"
    CA = "ca"
    CBK = "cbk"
    CE = "ce"
    CEB = "ceb"
    CKB = "ckb"
    CO = "co"
    CS = "cs"
    CV = "cv"
    CY = "cy"
    DA = "da"
    DE = "de"
    DIQ = "diq"
    DSB = "dsb"
    DTY = "dty"
    DV = "dv"
    EL = "el"
    EML = "eml"
    EN = "en"
    EO = "eo"
    ES = "es"
    ET = "et"
    EU = "eu"
    FA = "fa"
    FI = "fi"
    FR = "fr"
    FRR = "frr"
    FY = "fy"
    GA = "ga"
    GD = "gd"
    GL = "gl"
    GN = "gn"
    GOM = "gom"
    GU = "gu"
    GV = "gv"
    HE = "he"
    HI = "hi"
    HIF = "hif"
    HR = "hr"
    HSB = "hsb"
    HT = "ht"
    HU = "hu"
    HY = "hy"
    IA = "ia"
    ID = "id"
    IE = "ie"
    ILO = "ilo"
    IO = "io"
    IS = "is"
    IT = "it"
    JA = "ja"
    JBO = "jbo"
    JV = "jv"
    KA = "ka"
    KK = "kk"
    KM = "km"
    KN = "kn"
    KO = "ko"
    KRC = "krc"
    KU = "ku"
    KV = "kv"
    KW = "kw"
    KY = "ky"
    LA = "la"
    LB = "lb"
    LEZ = "lez"
    LI = "li"
    LMO = "lmo"
    LO = "lo"
    LRC = "lrc"
    LT = "lt"
    LV = "lv"
    MAI = "mai"
    MG = "mg"
    MHR = "mhr"
    MIN = "min"
    MK = "mk"
    ML = "ml"
    MN = "mn"
    MR = "mr"
    MRJ = "mrj"
    MS = "ms"
    MT = "mt"
    MWL = "mwl"
    MY = "my"
    MYV = "myv"
    MZN = "mzn"
    NAH = "nah"
    NAP = "nap"
    NDS = "nds"
    NE = "ne"
    NEW = "new"
    NL = "nl"
    NN = "nn"
    NO = "no"
    OC = "oc"
    OR = "or"
    OS = "os"
    PA = "pa"
    PAM = "pam"
    PFL = "pfl"
    PL = "pl"
    PMS = "pms"
    PNB = "pnb"
    PS = "ps"
    PT = "pt"
    QU = "qu"
    RM = "rm"
    RO = "ro"
    RU = "ru"
    RUE = "rue"
    SA = "sa"
    SAH = "sah"
    SC = "sc"
    SCN = "scn"
    SCO = "sco"
    SD = "sd"
    SH = "sh"
    SI = "si"
    SK = "sk"
    SL = "sl"
    SO = "so"
    SQ = "sq"
    SR = "sr"
    SU = "su"
    SV = "sv"
    SW = "sw"
    TA = "ta"
    TE = "te"
    TG = "tg"
    TH = "th"
    TK = "tk"
    TL = "tl"
    TR = "tr"
    TT = "tt"
    TYV = "tyv"
    UG = "ug"
    UK = "uk"
    UR = "ur"
    UZ = "uz"
    VEC = "vec"
    VEP = "vep"
    VI = "vi"
    VLS = "vls"
    VO = "vo"
    WA = "wa"
    WAR = "war"
    WUU = "wuu"
    XAL = "xal"
    XMF = "xmf"
    YI = "yi"
    YO = "yo"
    YUE = "yue"
    ZH = "zh"


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


class LanguageFields:
    label: ScalarField[str] = ScalarField("language.label", str)
    af: ScalarField[float] = ScalarField("language.af", float)
    als: ScalarField[float] = ScalarField("language.als", float)
    am: ScalarField[float] = ScalarField("language.am", float)
    an: ScalarField[float] = ScalarField("language.an", float)
    ar: ScalarField[float] = ScalarField("language.ar", float)
    arz: ScalarField[float] = ScalarField("language.arz", float)
    as_: ScalarField[float] = ScalarField("language.as", float)
    ast: ScalarField[float] = ScalarField("language.ast", float)
    av: ScalarField[float] = ScalarField("language.av", float)
    az: ScalarField[float] = ScalarField("language.az", float)
    azb: ScalarField[float] = ScalarField("language.azb", float)
    ba: ScalarField[float] = ScalarField("language.ba", float)
    bar: ScalarField[float] = ScalarField("language.bar", float)
    bcl: ScalarField[float] = ScalarField("language.bcl", float)
    be: ScalarField[float] = ScalarField("language.be", float)
    bg: ScalarField[float] = ScalarField("language.bg", float)
    bh: ScalarField[float] = ScalarField("language.bh", float)
    bn: ScalarField[float] = ScalarField("language.bn", float)
    bo: ScalarField[float] = ScalarField("language.bo", float)
    bpy: ScalarField[float] = ScalarField("language.bpy", float)
    br: ScalarField[float] = ScalarField("language.br", float)
    bs: ScalarField[float] = ScalarField("language.bs", float)
    bxr: ScalarField[float] = ScalarField("language.bxr", float)
    ca: ScalarField[float] = ScalarField("language.ca", float)
    cbk: ScalarField[float] = ScalarField("language.cbk", float)
    ce: ScalarField[float] = ScalarField("language.ce", float)
    ceb: ScalarField[float] = ScalarField("language.ceb", float)
    ckb: ScalarField[float] = ScalarField("language.ckb", float)
    co: ScalarField[float] = ScalarField("language.co", float)
    cs: ScalarField[float] = ScalarField("language.cs", float)
    cv: ScalarField[float] = ScalarField("language.cv", float)
    cy: ScalarField[float] = ScalarField("language.cy", float)
    da: ScalarField[float] = ScalarField("language.da", float)
    de: ScalarField[float] = ScalarField("language.de", float)
    diq: ScalarField[float] = ScalarField("language.diq", float)
    dsb: ScalarField[float] = ScalarField("language.dsb", float)
    dty: ScalarField[float] = ScalarField("language.dty", float)
    dv: ScalarField[float] = ScalarField("language.dv", float)
    el: ScalarField[float] = ScalarField("language.el", float)
    eml: ScalarField[float] = ScalarField("language.eml", float)
    en: ScalarField[float] = ScalarField("language.en", float)
    eo: ScalarField[float] = ScalarField("language.eo", float)
    es: ScalarField[float] = ScalarField("language.es", float)
    et: ScalarField[float] = ScalarField("language.et", float)
    eu: ScalarField[float] = ScalarField("language.eu", float)
    fa: ScalarField[float] = ScalarField("language.fa", float)
    fi: ScalarField[float] = ScalarField("language.fi", float)
    fr: ScalarField[float] = ScalarField("language.fr", float)
    frr: ScalarField[float] = ScalarField("language.frr", float)
    fy: ScalarField[float] = ScalarField("language.fy", float)
    ga: ScalarField[float] = ScalarField("language.ga", float)
    gd: ScalarField[float] = ScalarField("language.gd", float)
    gl: ScalarField[float] = ScalarField("language.gl", float)
    gn: ScalarField[float] = ScalarField("language.gn", float)
    gom: ScalarField[float] = ScalarField("language.gom", float)
    gu: ScalarField[float] = ScalarField("language.gu", float)
    gv: ScalarField[float] = ScalarField("language.gv", float)
    he: ScalarField[float] = ScalarField("language.he", float)
    hi: ScalarField[float] = ScalarField("language.hi", float)
    hif: ScalarField[float] = ScalarField("language.hif", float)
    hr: ScalarField[float] = ScalarField("language.hr", float)
    hsb: ScalarField[float] = ScalarField("language.hsb", float)
    ht: ScalarField[float] = ScalarField("language.ht", float)
    hu: ScalarField[float] = ScalarField("language.hu", float)
    hy: ScalarField[float] = ScalarField("language.hy", float)
    ia: ScalarField[float] = ScalarField("language.ia", float)
    id: ScalarField[float] = ScalarField("language.id", float)
    ie: ScalarField[float] = ScalarField("language.ie", float)
    ilo: ScalarField[float] = ScalarField("language.ilo", float)
    io: ScalarField[float] = ScalarField("language.io", float)
    is_: ScalarField[float] = ScalarField("language.is", float)
    it: ScalarField[float] = ScalarField("language.it", float)
    ja: ScalarField[float] = ScalarField("language.ja", float)
    jbo: ScalarField[float] = ScalarField("language.jbo", float)
    jv: ScalarField[float] = ScalarField("language.jv", float)
    ka: ScalarField[float] = ScalarField("language.ka", float)
    kk: ScalarField[float] = ScalarField("language.kk", float)
    km: ScalarField[float] = ScalarField("language.km", float)
    kn: ScalarField[float] = ScalarField("language.kn", float)
    ko: ScalarField[float] = ScalarField("language.ko", float)
    krc: ScalarField[float] = ScalarField("language.krc", float)
    ku: ScalarField[float] = ScalarField("language.ku", float)
    kv: ScalarField[float] = ScalarField("language.kv", float)
    kw: ScalarField[float] = ScalarField("language.kw", float)
    ky: ScalarField[float] = ScalarField("language.ky", float)
    la: ScalarField[float] = ScalarField("language.la", float)
    lb: ScalarField[float] = ScalarField("language.lb", float)
    lez: ScalarField[float] = ScalarField("language.lez", float)
    li: ScalarField[float] = ScalarField("language.li", float)
    lmo: ScalarField[float] = ScalarField("language.lmo", float)
    lo: ScalarField[float] = ScalarField("language.lo", float)
    lrc: ScalarField[float] = ScalarField("language.lrc", float)
    lt: ScalarField[float] = ScalarField("language.lt", float)
    lv: ScalarField[float] = ScalarField("language.lv", float)
    mai: ScalarField[float] = ScalarField("language.mai", float)
    mg: ScalarField[float] = ScalarField("language.mg", float)
    mhr: ScalarField[float] = ScalarField("language.mhr", float)
    min: ScalarField[float] = ScalarField("language.min", float)
    mk: ScalarField[float] = ScalarField("language.mk", float)
    ml: ScalarField[float] = ScalarField("language.ml", float)
    mn: ScalarField[float] = ScalarField("language.mn", float)
    mr: ScalarField[float] = ScalarField("language.mr", float)
    mrj: ScalarField[float] = ScalarField("language.mrj", float)
    ms: ScalarField[float] = ScalarField("language.ms", float)
    mt: ScalarField[float] = ScalarField("language.mt", float)
    mwl: ScalarField[float] = ScalarField("language.mwl", float)
    my: ScalarField[float] = ScalarField("language.my", float)
    myv: ScalarField[float] = ScalarField("language.myv", float)
    mzn: ScalarField[float] = ScalarField("language.mzn", float)
    nah: ScalarField[float] = ScalarField("language.nah", float)
    nap: ScalarField[float] = ScalarField("language.nap", float)
    nds: ScalarField[float] = ScalarField("language.nds", float)
    ne: ScalarField[float] = ScalarField("language.ne", float)
    new: ScalarField[float] = ScalarField("language.new", float)
    nl: ScalarField[float] = ScalarField("language.nl", float)
    nn: ScalarField[float] = ScalarField("language.nn", float)
    no: ScalarField[float] = ScalarField("language.no", float)
    oc: ScalarField[float] = ScalarField("language.oc", float)
    or_: ScalarField[float] = ScalarField("language.or", float)
    os: ScalarField[float] = ScalarField("language.os", float)
    pa: ScalarField[float] = ScalarField("language.pa", float)
    pam: ScalarField[float] = ScalarField("language.pam", float)
    pfl: ScalarField[float] = ScalarField("language.pfl", float)
    pl: ScalarField[float] = ScalarField("language.pl", float)
    pms: ScalarField[float] = ScalarField("language.pms", float)
    pnb: ScalarField[float] = ScalarField("language.pnb", float)
    ps: ScalarField[float] = ScalarField("language.ps", float)
    pt: ScalarField[float] = ScalarField("language.pt", float)
    qu: ScalarField[float] = ScalarField("language.qu", float)
    rm: ScalarField[float] = ScalarField("language.rm", float)
    ro: ScalarField[float] = ScalarField("language.ro", float)
    ru: ScalarField[float] = ScalarField("language.ru", float)
    rue: ScalarField[float] = ScalarField("language.rue", float)
    sa: ScalarField[float] = ScalarField("language.sa", float)
    sah: ScalarField[float] = ScalarField("language.sah", float)
    sc: ScalarField[float] = ScalarField("language.sc", float)
    scn: ScalarField[float] = ScalarField("language.scn", float)
    sco: ScalarField[float] = ScalarField("language.sco", float)
    sd: ScalarField[float] = ScalarField("language.sd", float)
    sh: ScalarField[float] = ScalarField("language.sh", float)
    si: ScalarField[float] = ScalarField("language.si", float)
    sk: ScalarField[float] = ScalarField("language.sk", float)
    sl: ScalarField[float] = ScalarField("language.sl", float)
    so: ScalarField[float] = ScalarField("language.so", float)
    sq: ScalarField[float] = ScalarField("language.sq", float)
    sr: ScalarField[float] = ScalarField("language.sr", float)
    su: ScalarField[float] = ScalarField("language.su", float)
    sv: ScalarField[float] = ScalarField("language.sv", float)
    sw: ScalarField[float] = ScalarField("language.sw", float)
    ta: ScalarField[float] = ScalarField("language.ta", float)
    te: ScalarField[float] = ScalarField("language.te", float)
    tg: ScalarField[float] = ScalarField("language.tg", float)
    th: ScalarField[float] = ScalarField("language.th", float)
    tk: ScalarField[float] = ScalarField("language.tk", float)
    tl: ScalarField[float] = ScalarField("language.tl", float)
    tr: ScalarField[float] = ScalarField("language.tr", float)
    tt: ScalarField[float] = ScalarField("language.tt", float)
    tyv: ScalarField[float] = ScalarField("language.tyv", float)
    ug: ScalarField[float] = ScalarField("language.ug", float)
    uk: ScalarField[float] = ScalarField("language.uk", float)
    ur: ScalarField[float] = ScalarField("language.ur", float)
    uz: ScalarField[float] = ScalarField("language.uz", float)
    vec: ScalarField[float] = ScalarField("language.vec", float)
    vep: ScalarField[float] = ScalarField("language.vep", float)
    vi: ScalarField[float] = ScalarField("language.vi", float)
    vls: ScalarField[float] = ScalarField("language.vls", float)
    vo: ScalarField[float] = ScalarField("language.vo", float)
    wa: ScalarField[float] = ScalarField("language.wa", float)
    war: ScalarField[float] = ScalarField("language.war", float)
    wuu: ScalarField[float] = ScalarField("language.wuu", float)
    xal: ScalarField[float] = ScalarField("language.xal", float)
    xmf: ScalarField[float] = ScalarField("language.xmf", float)
    yi: ScalarField[float] = ScalarField("language.yi", float)
    yo: ScalarField[float] = ScalarField("language.yo", float)
    yue: ScalarField[float] = ScalarField("language.yue", float)
    zh: ScalarField[float] = ScalarField("language.zh", float)

    def __getitem__(self, label: Language) -> ScalarField[float]:
        if not isinstance(label, Language):
            raise TypeError("expected Language")
        return ScalarField("language." + label.value, float)


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
