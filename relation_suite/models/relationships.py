import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple
from relation_suite.models.relationship_type import RelationshipType
from networkx.drawing.nx_agraph import graphviz_layout
from math import sqrt


class Relationships:
    DEFAULT_RELATIONSHIPS = {
        RelationshipType("cause"),
        RelationshipType("inhibit"),
        RelationshipType("positively correlate", is_symmetric=True),
        RelationshipType("negatively correlate", is_symmetric=True),
    }

    def __init__(
        self,
        relationship_types: Set[RelationshipType] = DEFAULT_RELATIONSHIPS,
        lowercase: bool = True,
    ) -> None:
        self.relationship_types = {rel.name: rel for rel in relationship_types}
        self.relationships: Dict[Tuple[str, str], str] = {}
        self.lowercase = lowercase

    def __iter__(self):
        return self.relationships

    def add(self, entity_1: str, entity_2: str, relationship: str) -> bool:
        if relationship not in self.relationship_types:
            print(f"Relationship type '{relationship}' is not allowed.")
            return False

        if self.lowercase:
            entity_1 = entity_1.lower()
            entity_2 = entity_2.lower()
            relationship = relationship.lower()

        rel_type = self.relationship_types[relationship]
        self.relationships[(entity_1, entity_2)] = relationship
        if rel_type.is_symmetric:
            self.relationships[(entity_2, entity_1)] = relationship
        return True

    def remove(self, entity_1: str, entity_2: str, relationship: str) -> bool:
        try:
            if self.relationships.pop((entity_1, entity_2), None) == relationship:
                if self.relationship_types[relationship].is_symmetric:
                    self.relationships.pop((entity_2, entity_1), None)
                return True
        except KeyError:
            pass
        return False

    def check_entity(self, entity: str) -> bool:
        """
        Checks if a specific entity exists in the relationships.

        :param entity: The entity to check.
        :return: True if the entity exists, False otherwise.
        """
        return any(entity in pair for pair in self.relationships.keys())

    def get(self, entity_1: str, entity_2: str) -> str:
        """
        Retrieves the relationship between two entities.

        :param entity_1: The first entity in the relationship.
        :param entity_2: The second entity in the relationship.
        :return: The relationship type if it exists, None otherwise.
        """
        return self.relationships.get((entity_1, entity_2))

    def get_all(self) -> Dict[Tuple[str, str], str]:
        """
        Retrieves all relationships stored in the Relationships object, maintaining the internal dictionary structure.

        :return: A dictionary with keys as a tuple of (entity_1, entity_2) and values as the relationship type.
        """
        return self.relationships

    def get_num_relationships(self) -> int:
        """
        Retrieves the number of relationships stored in the Relationships object.

        :return: The number of relationships stored.
        """
        return len(self.relationships)

    def to_json(self) -> str:
        """
        Exports the relationships to a JSON string.

        :return: A JSON string representing the relationships.
        """
        serializable_dict = {
            "relationships": [
                {
                    "entity_1": entity_1,
                    "entity_2": entity_2,
                    "relationship": relationship,
                }
                for (entity_1, entity_2), relationship in self.relationships.items()
            ]
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

    def plot_digraph(
        self,
        filename: str = "relations.png",
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 200,
        colored_relationships: Optional[List[Tuple["Relationships", str]]] = None,
    ) -> None:
        """
        Plots the directed graph (DiGraph) of the relationships and saves it to a file,
        coloring edges based on provided specifications. If no specific colors are provided,
        all edges are plotted in a default color.

        :param colored_relationships: Optional list of tuples, each containing a Relationships instance and a color.
        :param filename: The filename to save the plot. Defaults to 'relations.png'.
        :param figsize: The size of the figure (width, height) in inches.
        :param dpi: The resolution of the figure in dots per inch.
        """
        G = self.to_digraph()
        if figsize is None:
            figsize = (sqrt(len(G.nodes)) * 5, sqrt(len(G.nodes)) * 5)
        plt.figure(figsize=figsize)
        if self.get_num_relationships() > 50:
            pos = graphviz_layout(
                G, prog="neato", args="-Goverlap=false -Gsplines=true"
            )
        else:
            pos = graphviz_layout(G, prog="dot")

        # Default edge drawing
        nx.draw_networkx_edges(
            G, pos, edge_color="black", arrowstyle="-|>", arrowsize=20
        )
        if colored_relationships:
            # Draw edges from specified Relationships instances in given colors
            for relationships_instance, color in colored_relationships:
                if relationships_instance:
                    subgraph = relationships_instance.to_digraph()
                for u, v, d in subgraph.edges(data=True):
                    if (u, v) not in G.edges():
                        raise ValueError(
                            f"Edge from {u} to {v} does not exist in the main graph."
                        )
                    if G.edges[(u, v)]["relationship"] != d["relationship"]:
                        raise ValueError(
                            f"Edge from {u} to {v} has different relationship type."
                        )
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(u, v)],
                        width=2,
                        edge_color=color,
                        arrowstyle="-|>",
                        arrowsize=20,
                    )

        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        edge_labels = {(u, v): d["relationship"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_color="red", font_size=12
        )

        plt.axis("off")
        plt.savefig(filename, dpi=dpi)
        plt.close()

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

        return self.to_json()

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

    def from_json(self, json_str: str) -> None:
        """
        Imports relationships from a JSON string.

        :param json_str: A JSON string representing the relationships.
        """
        self.relationships.clear()
        self.relationship_types = set()
        try:
            data = json.loads(json_str)
            for relationship in data["relationships"]:
                self.relationships[
                    (relationship["entity_1"], relationship["entity_2"])
                ] = relationship["relationship"]
                self.relationship_types.add(relationship["relationship"])
        except json.JSONDecodeError:
            print("Relationships::from_json: Invalid JSON string.")
        except KeyError:
            print("Relationships::from_json: Invalid serialized format.")
