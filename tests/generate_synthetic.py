from relation_suite.models.relationships import Relationships
from relation_suite.utils.synthetic_generator import SyntheticGenerator
from constants import *

generator = SyntheticGenerator(
    ENTITIES_DIR + "/biomedical.json",
    SYNTHETIC_DIR + "/biomedical/",
    Relationships.DEFAULT_RELATIONSHIPS,
)
generator.generate_all(k=(2, 3), n=40)
