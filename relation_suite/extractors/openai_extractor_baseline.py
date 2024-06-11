import os
from typing import List, Dict, Any
from relation_suite.extractors.extractor import Extractor
from relation_suite.models.relationships import Relationships
from relation_suite.data.input_data import InputData

import openai
import json


class OpenAIExtractorBaseline(Extractor):
    DEFAULT_PROMPT = """You are an expert in determining causal and noncausal relationships from academic literature. The user needs your help to read a paper abstract and extract the relationships of certain types between certain entities.

The user will provide you with the paper's title, abstract, the entities that have relationships with one another, and the types of relationships possible. Report your answer in json format.
For example, if the types of relationships are {"cause", "inhibit", "positively correlate", "negatively correlate"}, and the paper says "the presence of green spaces within urban environments not only inhibits the adverse effects of air pollution on respiratory health but also positively correlates with improvements in mental well-being", then the output should be:
{"Relationships": [{"entity_1": "green spaces", "entity_2": "air pollution", "relationship": "inhibit"}, {"entity_1": "green spaces", "entity_2": "mental well-being", "relationship": "positively correlate"}]}

Be exhaustive and identify all plausible relationships.""".strip()

    # DEFAULT_RESPONSE_FORMAT = {
    #     "type": "object",
    #     "properties": {
    #         "Relationships": {
    #             "type": "array",
    #             "items": {
    #                 "type": "object",
    #                 "properties": {
    #                     "A": {"type": "string"},
    #                     "B": {"type": "string"},
    #                     "Relationship": {"type": "string"},
    #                 },
    #                 "additionalProperties": False,
    #                 "required": ["A", "B", "Relationship"],
    #             },
    #         }
    #     },
    #     "required": ["Relationships"],
    # }

    RETRIES = 5

    def __init__(
        self,
        model: str,
        prompt: str = DEFAULT_PROMPT,
        response_format: Dict[str, Any] = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.response_format = response_format
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def parse_llm_output(self, output):
        try:
            data = json.loads(output)
            if data and "Relationships" in data and len(data["Relationships"]) > 0:
                return data["Relationships"]
            else:
                return None
        except json.JSONDecodeError:
            print("Invalid JSON string.")
            return None
        except TypeError:
            print("An error occurred while parsing the JSON string.")
            return None

    def extract(self, input_data: InputData) -> Relationships:
        relationships = Relationships()
        n_readings = len(input_data.get_data())
        for i, reading in enumerate(input_data.get_data()):
            print(f"\nReading {i+1}/{n_readings}: {reading.title}")
            data = {
                "Title": reading.title,
                "Abstract": reading.abstract,
                "Entities": reading.entities,
                "Relationship_Types": reading.relationship_types,
            }
            messages = [
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": "Please answer in JSON encoding and nothing else:\n"
                    + str(data),
                },
            ]

            for _ in range(self.RETRIES):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format=self.response_format,
                )
                print(response.choices[0].message.content)
                output = self.parse_llm_output(
                    response.choices[0].message.content.strip()
                )
                if output is None:
                    continue
                success = False
                for relationship in output:
                    if (
                        "entity_1" in relationship
                        and "entity_2" in relationship
                        and "relationship" in relationship
                    ):
                        success = True
                        break
                    else:
                        print("Invalid output format, regenerating…")
                if success:
                    break
            print(f"Serialized: {output}")

            if output:
                for relationship in output:
                    try:
                        if (
                            relationship["relationship"]
                            in relationships.relationship_types
                            and relationship["entity_1"] in reading.entities
                            and relationship["entity_2"] in reading.entities
                        ):
                            relationships.add(
                                relationship["entity_1"],
                                relationship["entity_2"],
                                relationship["relationship"],
                            )
                    except KeyError:
                        print("Invalid serialized format.")

        return relationships
