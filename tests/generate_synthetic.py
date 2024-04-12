from relation_suite.models.relationships import Relationships
from relation_suite.utils.synthetic_generator import SyntheticGenerator
from constants import *

generator = SyntheticGenerator(
    ENTITIES_DIR + "/marine_ecology.json",
    SYNTHETIC_DIR + "/marine_ecology/",
    Relationships.DEFAULT_RELATIONSHIPS,
)
generator.generate_all(k=5, n=20)
