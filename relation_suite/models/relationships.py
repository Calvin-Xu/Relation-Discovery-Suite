import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Set, Tuple


class Relationships:
    DEFAULT_RELATIONSHIPS = {
        "cause",
        "inhibit",
        "positively correlate",
        "negatively correlate",
    }

    def __init__(self, relationship_types: Set[str] = DEFAULT_RELATIONSHIPS) -> None:
        """
        Initializes the Relationships object with a custom set of allowed relationships or a default set.

        :param relationship_types: A set of strings representing allowed relationship types.
        """
        self.relationship_types = relationship_types
        self.relationships: Dict[Tuple[str, str], str] = {}

    def add_relation(self, entity_1: str, entity_2: str, relationship: str) -> bool:
        """
        Adds a relationship between two entities if the relationship type is allowed.

        :param entity_1: The first entity in the relationship.
        :param entity_2: The second entity in the relationship.
        :param relationship: The type of relationship.
        :return: True if the relationship was added, False otherwise.
        """
        if relationship not in self.relationship_types:
            print(f"Relationship type '{relationship}' is not allowed.")
            return False

        self.relationships[(entity_1, entity_2)] = relationship
        return True

    def remove_relation(self, entity_1: str, entity_2: str, relationship: str) -> bool:
        """
        Removes a specified relationship between two entities.

        :param entity_1: The first entity in the relationship.
        :param entity_2: The second entity in the relationship.
        :param relationship: The type of relationship to be removed.
        :return: True if the relationship was removed, False otherwise.
        """
        try:
            if self.relationships[(entity_1, entity_2)] == relationship:
                del self.relationships[(entity_1, entity_2)]
                return True
        except KeyError:
            pass
        return False

    def check_relation(self, entity_1: str, entity_2: str, relationship: str) -> bool:
        """
        Checks if a specific relationship exists between two entities.

        :param entity_1: The first entity in the relationship.
        :param entity_2: The second entity in the relationship.
        :param relationship: The type of relationship.
        :return: True if the relationship exists, False otherwise.
        """
        return self.relationships.get((entity_1, entity_2)) == relationship

    def get_relationships(self) -> Dict[Tuple[str, str], str]:
        """
        Retrieves all relationships stored in the Relationships object, maintaining the internal dictionary structure.

        :return: A dictionary with keys as a tuple of (entity_1, entity_2) and values as the relationship type.
        """
        return self.relationships

    def to_json(self) -> str:
        """
        Exports the relationships to a JSON string.

        :return: A JSON string representing the relationships.
        """
        serializable_dict = {
            f"{entity_1} | {entity_2}": relationship
            for (entity_1, entity_2), relationship in self.relationships.items()
        }
        return json.dumps(serializable_dict, indent=4)

    def to_digraph(self) -> nx.DiGraph:
        """
        Outputs the relationships as a directed graph (DiGraph) using the NetworkX library.

        :return: A NetworkX DiGraph representing the relationships.
        """
        G = nx.DiGraph()
        for (entity_1, entity_2), relationship in self.relationships.items():
            G.add_edge(entity_1, entity_2, relationship=relationship)
        return G

    def plot_digraph(self, filename: str = "relations.png") -> None:
        """
        Plots the directed graph (DiGraph) of the relationships and saves it to a file.

        :param filename: The filename to save the plot. Defaults to 'relations.png'.
        """
        G = self.to_digraph()
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(
            G, arrows=True, with_labels=True, pos=nx.spring_layout(G, iterations=200)
        )
        plt.savefig(filename, dpi=300)

    def __str__(self) -> str:
        """
        Provides a string representation of the Relationships object, listing all stored relationships.

        :return: A string representing the relationships in the object.
        """
        if not self.relationships:
            return "No relationships stored."
        relationships_str_list = [
            f"{entity_1} -> {entity_2} [{relationship}]"
            for (entity_1, entity_2), relationship in self.relationships.items()
        ]
        return "\n".join(relationships_str_list)

    def populate(self, relationship_tuples: Set[Tuple[str, str, str]]) -> None:
        """
        Clears the existing relationships and repopulates the Relationships object
        with a new set of relationship tuples. Also updates the relationship_types.

        :param relationship_tuples: A set of tuples (entity_1, entity_2, relationship_type) to repopulate the object.
        """
        self.relationships.clear()
        self.relationship_types = set()

        for entity_1, entity_2, relationship in relationship_tuples:
            self.relationships[(entity_1, entity_2)] = relationship
            self.relationship_types.add(relationship)
