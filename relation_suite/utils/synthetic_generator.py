import os
import json
import random
from typing import List, Dict, Set
from openai import OpenAI
from relation_suite.models.relationship_type import RelationshipType
from relation_suite.models.relationships import Relationships

MODEL = "gpt-4-turbo"


class SyntheticGenerator:
    def __init__(
        self,
        json_file_path: str,
        output_dir: str,
        relationship_types: Set[RelationshipType],
    ):
        self.entities = []
        self.load_data(json_file_path)
        self.output_directory = output_dir
        self.relationship_types = relationship_types
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.generated_relationships = Relationships(relationship_types)

    def load_data(self, file_path: str) -> None:
        with open(file_path, "r") as file:
            data = json.load(file)
            self.entities = data.get("entities", [])

    def _generate_example(self, selected_entities: List[str]) -> Dict:
        # Step 1: Generate Relationships
        entities_str = ", ".join(selected_entities)
        relationship_types_str = ", ".join([r.name for r in self.relationship_types])
        relationship_prompt = f"""
        Instruction: Identify plausible relationships between the following entities: {entities_str}.
        Each relationship should be one of these types: {relationship_types_str}.
        Please provide the list of relationships in JSON format with keys 'entity_1', 'entity_2', and 'relationship'.
        """.strip()

        relationship_response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": relationship_prompt}],
            model=MODEL,
            response_format={"type": "json_object"},
        )

        relationships = json.loads(relationship_response.choices[0].message.content)[
            "relationships"
        ]
        print(f"Generated plausible relationships: {relationships}")

        # Step 2: Generate Academic Paper
        paper_prompt = f"""
        Instruction: Create the title and abstract of a hypothetical academic paper that reports results or insights related to the following relationships: {relationships}.
        The abstract should be written in a formal, academic style, and might use synonyms or equivalent expressions to report on the relationships between the entities. The entities might not always be listed explicitly.
        Provide the title and abstract in JSON format.
        """.strip()

        paper_response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": paper_prompt}],
            model=MODEL,
            response_format={"type": "json_object"},
        )

        paper_details = json.loads(paper_response.choices[0].message.content)
        print(paper_details)

        # Combine results
        result = {
            "title": paper_details["title"],
            "abstract": paper_details["abstract"],
            "entities": selected_entities,
            "relationship_types": [r.name for r in self.relationship_types],
            "relationships": relationships,
        }
        return result

    def _write_json(self, data: Dict, file_name: str) -> None:
        with open(self.output_directory + file_name, "w") as file:
            json.dump(data, file, indent=4)

    def _validate_output(self, output: Dict) -> bool:
        if not output:
            return False

        if (
            "title" not in output
            or "abstract" not in output
            or "relationships" not in output
        ):
            return False

        for relationship in output["relationships"]:
            # check that relationship has required keys
            if (
                "entity_1" not in relationship
                or "entity_2" not in relationship
                or "relationship" not in relationship
            ):
                print(f"Invalid relationship: {relationship}")
                return False
            entity_1 = relationship["entity_1"]
            entity_2 = relationship["entity_2"]
            proposed_relation = relationship["relationship"]
            lowercase_entities = [entity.lower() for entity in self.entities]
            # check that entities and relationship type are valid
            if (
                entity_1.lower() not in lowercase_entities
                or entity_2.lower() not in lowercase_entities
            ):
                print(f"Invalid entities: {relationship}")
                return False
            if proposed_relation.lower() not in [
                r.name.lower() for r in self.relationship_types
            ]:
                print(f"Invalid relationship type: {relationship}")
                return False
            existing_relationship = self.generated_relationships.get(entity_1, entity_2)
            # check that relationship is consistent with existing relationships
            if (
                existing_relationship != None
                and existing_relationship.lower() != proposed_relation.lower()
            ):
                print(
                    f"Inconsistent relationship: {relationship}, {existing_relationship}"
                )
                return False

            self.generated_relationships.add(
                entity_1,
                entity_2,
                proposed_relation,
            )

        return True

    def generate_all(self, k: int, n: int) -> None:
        if not self.entities:
            print("No entities loaded.")
            return

        for i in range(n):
            print(f"Generating example {i}...")
            selected_entities = random.sample(self.entities, k)
            print(f"Selected entities: {selected_entities}")
            result = self._generate_example(selected_entities)
            print(f"Validating output...")
            if not self._validate_output(result):
                print("Invalid output. Skipping...")
                continue
            print("Output is valid.")
            print(result)
            self._write_json(result, f"example_{i}.json")
