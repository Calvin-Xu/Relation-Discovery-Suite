from typing import List, Dict, Any
from relation_suite.extractors.extractor import Extractor
from relation_suite.models.relationships import Relationships
from relation_suite.data.input_data import InputData

from llama_cpp import Llama
import json


class LlamaExtractor(Extractor):
    DEFAULT_PROMPT = """You are an expert in determining causal and noncausal relationships from academic literature. I need your help to read a paper abstract and inform me of the relationships of certain types between certain entities.

I will provide you with the paper's title, abstract, the entities that have relationships with one another, and the types of relationships possible. Tell me your answer in json format.
For example, if the types of relationships are {"cause", "inhibit", "positively correlate", "negatively correlate"}, and the paper says "the presence of green spaces within urban environments not only inhibits the adverse effects of air pollution on respiratory health but also positively correlates with improvements in mental well-being", then the output should be:
{"Relationships": [{'A': 'green spaces', 'B': 'air pollution', 'Relation': 'inhibit'}, {'A': 'green spaces', 'B': 'mental well-being', 'VERB': 'positively correlate'}]}

If you cannot answer the question, return an empty JSON object. Please provide no explanation or justification. Just the JSON encoding.
    """.strip()  # TODO: how to handle inline long prompts?

    DEFAULT_RESPONSE_FORMAT = {
        "type": "object",
        "properties": {
            "Relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "A": {"type": "string"},
                        "B": {"type": "string"},
                        "Relationship": {"type": "string"},
                    },
                    "additionalProperties": False,
                    "required": ["A", "B", "Relationship"],
                },
            }
        },
        "required": ["Relationships"],
    }

    RETRIES = 5

    def __init__(
        self,
        model_path: str,
        prompt: str = DEFAULT_PROMPT,
        response_format: Dict[str, Any] = DEFAULT_RESPONSE_FORMAT,
    ) -> None:
        self.llm = Llama(
            model_path=model_path,
            chat_format="llama-2",
            n_gpu_layers=-1,
            n_batch=2048,
            n_ctx=2048,
            verbose=True,
        )
        self.prompt = prompt
        self.response_format = response_format

    def parse_llm_output(self, output):
        try:
            output = output.split("\n\n")[0]
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
        for reading in input_data.get_data():
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

            output, n = None, self.RETRIES
            while output is None and n > 0:
                response = self.llm.create_chat_completion(
                    messages, self.response_format, temperature=0.0
                )
                print(response)
                print(f'LLM output: {response["choices"][0]["message"]}')
                output = self.parse_llm_output(
                    response["choices"][0]["message"]["content"].strip()
                )
                n -= 1
            print(f"Serialized: {output}")

            if output:
                for relationship in output:
                    if relationship["Relation"] in relationships.relationship_types:
                        relationships.add_relation(
                            relationship["A"],
                            relationship["B"],
                            relationship["Relation"],
                        )

        return relationships
