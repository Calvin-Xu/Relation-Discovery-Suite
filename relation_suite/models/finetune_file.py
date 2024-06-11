import os
import json
from typing import List, Dict


class FinetuneFile:
    def __init__(
        self,
        output_path: str,
        dir_paths: List[str],
        system_prompt: str,
        user_template: str,
        assistant_template: str,
        keys: List[str],
    ):
        self.output_path = output_path
        self.dir_paths = dir_paths
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.keys = keys

    def collect_json_files(self) -> List[str]:
        json_files = []
        for dir_path in self.dir_paths:
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
                if key == "relationships":
                    # serialize the 'relationships' data into JSON format string if it's the key
                    # otherwise Python dicts have single quotes, which is not valid JSON
                    content_values[key] = json.dumps(data[key])
                else:
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

    def generate(self):
        json_files = self.collect_json_files()
        with open(self.output_path, "w") as outfile:
            for json_file in json_files:
                try:
                    example = self.process_json_file(json_file)
                    json.dump(example, outfile)
                    outfile.write("\n")
                except KeyError as e:
                    print(f"Skipping file {json_file} due to error: {str(e)}")
