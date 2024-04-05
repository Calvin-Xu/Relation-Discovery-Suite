import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Set

class Relationships:
    DEFAULT_RELATIONSHIPS = {"cause", "inhibit", "positively correlate", "negatively correlate"}

    def __init__(self, relationship_types: Set[str] = DEFAULT_RELATIONSHIPS) -> None:
        """
        Initializes the Relationships object with a custom set of allowed relationships or a default set.

        :param relationship_types: A set of strings representing allowed relationship types.
                                       If None, defaults to a predefined set of relationships.
        """
        self.relationship_types = relationship_types
        self.relationships: Dict[Tuple[str, str, str], None] = {}

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

        self.relationships[(entity_1, entity_2, relationship)] = None
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
            del self.relationships[(entity_1, entity_2, relationship)]
            return True
        except KeyError:
            return False

    def check_relation(self, entity_1: str, entity_2: str, relationship: str) -> bool:
        """
        Checks if a specific relationship exists between two entities.

        :param entity_1: The first entity in the relationship.
        :param entity_2: The second entity in the relationship.
        :param relationship: The type of relationship.
        :return: True if the relationship exists, False otherwise.
        """
        return (entity_1, entity_2, relationship) in self.relationships

    def to_json(self) -> str:
        """
        Exports the relationships to a JSON string.

        :return: A JSON string representing the relationships.
        """
        relationships_list = [{"entity_1": key[0], "entity_2": key[1], "relationship": key[2]} for key in self.relationships.keys()]
        return json.dumps(relationships_list, indent=4)

    def to_digraph(self) -> nx.DiGraph:
        """
        Outputs the relationships as a directed graph (DiGraph) using the NetworkX library.

        :return: A NetworkX DiGraph representing the relationships.
        """
        G = nx.DiGraph()
        for (entity_1, entity_2, relationship) in self.relationships:
            G.add_edge(entity_1, entity_2, relationship=relationship)
        return G

    def plot_digraph(self, filename: str = 'relations.png') -> None:
        """
        Plots the directed graph (DiGraph) of the relationships and saves it to a file.

        :param filename: The filename to save the plot. Defaults to 'relations.png'.
        """
        G = self.to_digraph()
        plt.figure(figsize=(10, 10))
        nx.draw_networkx(G, arrows=True, with_labels=True, pos=nx.spring_layout(G, iterations=200))
        plt.savefig(filename, dpi=300)
