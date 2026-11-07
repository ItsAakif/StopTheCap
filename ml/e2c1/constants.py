"""Frozen E2-C1 constants.

Changing a value here (especially a safety regex or a taxonomy entry) changes the
meaning of a built manifest. Bump the corresponding ``*_VERSION`` when doing so.
"""

from __future__ import annotations

# Version stamps recorded in every artefact.
PIPELINE_VERSION = "e2c1-prompts-v1"
SAFETY_RULESET_VERSION = "e2c1-safety-v1"
SPLIT_VERSION = "e2c1-prompt-split-v1"

#: Ordered taxonomy. The final value is the fallback.
TOPIC_TAXONOMY: tuple[str, ...] = (
    "medicine_health",
    "stem",
    "history",
    "geography_places",
    "society_politics",
    "business_economics",
    "arts_culture",
    "sports",
    "religion_philosophy",
    "other",
)

#: ``(topic, category_keywords, title_keywords)`` in match order.
#: Category keywords are case-insensitive substrings of the joined categories;
#: title keywords are case-insensitive substrings of the title.
TOPIC_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "medicine_health",
        (
            "Medicine",
            "Diseases",
            "Health",
            "Anatomy",
            "Pharmacology",
            "Surgery",
            "Viruses",
            "Vaccines",
            "Cancer",
            "Mental health",
        ),
        ("disease", "syndrome", "hospital", "vaccine", "medicine", "surgery"),
    ),
    (
        "stem",
        (
            "Physics",
            "Chemistry",
            "Biology",
            "Mathematics",
            "Computer science",
            "Engineering",
            "Technology",
            "Astronomy",
            "Geology",
            "Statistics",
            "Software",
        ),
        (
            "physics",
            "chemistry",
            "biology",
            "mathematics",
            "computer",
            "software",
            "algorithm",
            "spacecraft",
            "species",
        ),
    ),
    (
        "history",
        (
            "History",
            "Historiography",
            "Archaeology",
            "Ancient",
            "Middle Ages",
            "19th century",
            "20th century",
            "Empires",
            "Wars",
        ),
        ("history of", "battle", "dynasty", "empire", "century"),
    ),
    (
        "geography_places",
        (
            "Geography",
            "Countries",
            "Cities",
            "Rivers",
            "Mountains",
            "Islands",
            "Regions",
            "Populated places",
        ),
        ("river", "mountain", "city", "island", "country"),
    ),
    (
        "society_politics",
        (
            "Politics",
            "Government",
            "Law",
            "Sociology",
            "Education",
            "Media",
            "Journalism",
            "Human rights",
            "Elections",
            "Military",
        ),
        ("election", "government", "law", "police"),
    ),
    (
        "business_economics",
        (
            "Business",
            "Economics",
            "Companies",
            "Finance",
            "Trade",
            "Marketing",
            "Labour",
            "Currency",
        ),
        ("company", "bank", "economy", "market", "currency"),
    ),
    (
        "arts_culture",
        (
            "Arts",
            "Music",
            "Film",
            "Literature",
            "Painting",
            "Architecture",
            "Theatre",
            "Television",
            "Video games",
            "Fashion",
            "Museums",
        ),
        ("film", "album", "novel", "painting", "architecture", "game"),
    ),
    (
        "sports",
        (
            "Sports",
            "Football",
            "Basketball",
            "Cricket",
            "Olympics",
            "Athletes",
            "Championships",
        ),
        ("football", "cricket", "olympic", "championship", "player"),
    ),
    (
        "religion_philosophy",
        (
            "Religion",
            "Philosophy",
            "Theology",
            "Mythology",
            "Ethics",
            "Buddhism",
            "Christianity",
            "Islam",
        ),
        ("religion", "philosophy", "myth", "god", "church"),
    ),
    ("other", (), ()),
)

MIN_WORDS = 40
MAX_WORDS = 400
LATIN_RATIO_MIN = 0.95

NEAR_DUP_THRESHOLD = 0.85
#: Applies to the ``api_prompts`` path only; the wikitext builder uses ``DEFAULT_MAX_PER_PAGE = 3`` from ``wikitext_prompts.py``.
MAX_PROMPTS_PER_PAGE = 2

SELECTION_SEED = "e2c1-selection-2026-11-06"
SPLIT_SALT = "e2c1-prompt-split-2026-11-06"
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test_holdout": 0.1}
SPLIT_TOLERANCE = 10
GENERATION_SEED = "e2c1-generation-2026-11-06"

# API / cache.
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "StopTheCap-E2C1/1.0 (private academic research)"
MAX_TITLES_PER_REQUEST = 20
CACHE_DIR = "datasets/raw/e2c1_api"
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)

# Validation regexes (§A.1).
PROMPT_ID_RE = r"^pmt:[0-9a-f]{24}$"
PROMPT_HASH_RE = r"^[0-9a-f]{64}$"
REVISION_ID_RE = r"^\d{5,12}$"
FAMILY_ID_RE = r"^fam:[0-9a-f]{16}$"
DATE_RE = r"^\d{4}-\d{2}-\d{2}$"

# Safety regexes (§A.5).
BIO_CATEGORY_RE = r"\b(living people|\d{4} births|\d{4} deaths|people from|biography)\b"
PERSON_CATEGORY_RE = (
    r"\b(directors|actors|actresses|singers|musicians|politicians|writers|authors|"
    r"players|athletes|scientists|artists)\b"
)
BLOCKLIST_RE = (
    r"\b(sexual|pornograph|rape|incest|pedophil|child sexual|masturbat|prostitut|"
    r"suicide|self-harm|self harm|murder|massacre|torture|gore|dismember|genocide|"
    r"terrorist attack|child abuse)\b"
)
#: Stage 5 applies the blocklist regex to the title only. The SPEC does not define a
#: separate title regex; reusing the §A.5 blocklist is the only fully specified rule.
TITLE_BLOCKLIST_RE = BLOCKLIST_RE
QUOTE_RATIO_MAX = 0.02
QUOTE_LONG_RE = r'"[^"]{60,}"'
CONTENT_BLOCKLIST_CHARS = 1500

SOURCE_LICENCE = "CC-BY-SA-4.0"
