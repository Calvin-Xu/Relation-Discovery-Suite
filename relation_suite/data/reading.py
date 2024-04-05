from typing import List, Dict, Set
from relation_suite.models.relationships import Relationships
import json


class Reading:
    def __init__(
        self,
        title: str,
        abstract: str,
        entities: List[str],
        relationship_types: Set[str],
        relationships: Relationships,
    ) -> None:
        """
        Initializes a Reading object.

        :param title: The title of the reading.
        :param abstract: The abstract of the reading.
        :param entities: A list of entities mentioned in the reading.
        :param relationship_types: A set of all allowed relationship types for this reading.
        :param relationships: A list of relationships extracted from the reading.
        """
        self.title = title
        self.abstract = abstract
        self.entities = entities
        self.relationship_types = relationship_types
        self.relationships = relationships

    def to_json(self) -> str:
        """
        Converts the Reading object into a JSON serializable format.

        :return: A dictionary representing the Reading object.
        """
        return json.dumps(
            {
                "title": self.title,
                "abstract": self.abstract,
                "entities": self.entities,
                "relationship_types": list(self.relationship_types),
                "relationships": self.relationships.to_json(),
            },
            indent=4,
        )
