import os
import json
import random
from typing import List, Dict, Optional, Set, Tuple, Union
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
        selected_entities = [e.lower() for e in selected_entities]
        entities_str = ", ".join(selected_entities)
        relationship_types_str = ", ".join([r.name for r in self.relationship_types])
        relationship_prompt = f"""
        Instruction: Identify plausible relationships between the following entities: {entities_str}.
        Each relationship should be one of these types: {relationship_types_str}.
        Please provide the relationships in JSON format in list called 'relationships' with keys 'entity_1', 'entity_2', and 'relationship'.
        Only generate relationships that are plausible (generally true in non-contrived scenarios) given the entities. Return an empty list if you think there are none.
        You are not to generate a set of contradictory or redundant relationships. For example, if you generate "A causes B", you should not generate "B is negatively correlated with A" (contradictory) or "B is positively correlated with A" (redundant).
        """.strip()

        relationship_response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": relationship_prompt}],
            model=MODEL,
            response_format={"type": "json_object"},
        )

        relationships = json.loads(relationship_response.choices[0].message.content)[
            "relationships"
        ]

        if len(relationships) == 0:
            print("No plausible relationships found.")
            return None

        relationship_reflect_prompt = f"""
        Instruction: You have been instructed to perform the following task:
        ---Previous Task---
        {relationship_prompt}
        ---End of Previous Task---
        These are the relationships you generated: {relationships}.
        Do you think these relationships are plausible and satisfactory? Are there contradictory or redundant relationships? If not, please regenerate the relationships in the same json format: in list called 'relationships' with keys 'entity_1', 'entity_2', and 'relationship'. If they are satisfactory, return an empty json object.
        """
        print("Revising relationships...")
        reflect_response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": relationship_reflect_prompt}],
            model=MODEL,
            response_format={"type": "json_object"},
        )
        try:
            serialized = json.loads(reflect_response.choices[0].message.content)
            if serialized != {} or "relationships" not in serialized:
                return None
            if serialized["relationships"] != []:
                print(f"Revised relationships: {relationships}")
                relationships = serialized["relationships"]
        except json.JSONDecodeError:
            return None

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

        abstract_reflect_prompt = f"""
        Instruction: You have been instructed to reate the title and abstract of a hypothetical academic paper that reports results or insights related to the following relationships: {relationships}.
        The abstract you generated is as follows: {paper_details["abstract"]}.
        Do you think the abstract effectively reports on the relationships between the entities? The abstract should be written in a formal, academic style, and might use synonyms or equivalent expressions to report on the relationships between the entities. However, all the relationships should be clearly and accurately represented, and evident to a reader with adequate academic background.
        Please regenerate the abstract if you think it does not effectively report on the relationships between the entities. If it is satisfactory, say "SATISFACTORY".
        """

        reflect_response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": abstract_reflect_prompt}],
            model=MODEL,
        )
        if reflect_response.choices[0].message.content != "SATISFACTORY":
            print("Revising abstract...")
            paper_details["abstract"] = reflect_response.choices[0].message.content

        # Combine results
        result = {
            "title": paper_details["title"],
            "abstract": paper_details["abstract"],
            "entities": selected_entities,
            "relationship_types": [r.name for r in self.relationship_types],
            "relationships": relationships,
        }
        print(f"Generated example: {result}")
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
            # check that entities and relationship type are valid
            if entity_1 not in output["entities"] or entity_2 not in output["entities"]:
                print(f"Invalid entities: {relationship}")
                return False
            if proposed_relation not in [r.name for r in self.relationship_types]:
                print(f"Invalid relationship type: {relationship}")
                return False
            existing_relationship = self.generated_relationships.get(entity_1, entity_2)
            # check that relationship is consistent with existing relationships
            if (
                existing_relationship != None
                and existing_relationship != proposed_relation
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

    def generate_all(self, k: Union[Tuple[int, int], int], n: int) -> None:
        if not self.entities:
            print("No entities loaded.")
            return

        for i in range(n):
            print(f"Generating example {i}...")
            if isinstance(k, tuple):
                k = random.randint(k[0], k[1])
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
