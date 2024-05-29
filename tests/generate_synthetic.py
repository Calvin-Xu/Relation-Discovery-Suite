from relation_suite.models.relationships import Relationships
from relation_suite.utils.synthetic_generator import SyntheticGenerator
from constants import *

generator = SyntheticGenerator(
    ENTITIES_DIR + "/desert_ecology.json",
    SYNTHETIC_DIR + "/desert_ecology/",
    Relationships.DEFAULT_RELATIONSHIPS,
)
generator.generate_all(k=(2, 4), n=40)
