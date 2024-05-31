import os
import json
from typing import List, Dict


class FinetuneFileGenerator:
    def __init__(
        self,
        system_prompt: str,
        user_template: str,
        assistant_template: str,
        keys: List[str],
    ):
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.keys = keys

    def collect_json_files(self, dir_paths: List[str]) -> List[str]:
        json_files = []
        for dir_path in dir_paths:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith(".json"):
                        json_files.append(os.path.join(root, file))
        return json_files

    def process_json_file(self, file_path: str) -> Dict:
        with open(file_path, "r") as file:
            data = json.load(file)

        content_values = {}
        try:
            for key in self.keys:
                content_values[key] = data[key]
        except KeyError as e:
            raise KeyError(f"Key {str(e)} not found in JSON file {file_path}")

        user_content = self.user_template.format(**content_values)
        assistant_content = self.assistant_template.format(**content_values)

        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    def generate_training_file(self, dir_paths: List[str], output_file: str):
        json_files = self.collect_json_files(dir_paths)
        with open(output_file, "w") as outfile:
            for json_file in json_files:
                try:
                    example = self.process_json_file(json_file)
                    json.dump(example, outfile)
                    outfile.write("\n")
                except KeyError as e:
                    print(e)
