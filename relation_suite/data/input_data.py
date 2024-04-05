import json
from typing import List, Union
from pathlib import Path

from relation_suite.data.reading import Reading
from relation_suite.models.relationships import Relationships


class InputData:
    def __init__(self) -> None:
        """
        Initializes the InputData object.
        """
        self.readings: List[Reading] = []

    def load_data(self, paths: Union[str, List[str]]) -> None:
        """
        Loads and parses JSON files from given path(s). Each JSON file is converted into a Reading object.

        :param paths: A path to a directory containing JSON files, a single JSON file path, or a list of JSON file paths.
        """
        print(paths)
        if isinstance(paths, str):
            paths = [paths]  # ensure paths is always a list

        for path_str in paths:
            path = Path(path_str)
            if path.is_dir():
                for file_path in path.glob("*.json"):
                    self._load_json(file_path)
            elif path.is_file():
                self._load_json(path)
            else:
                raise FileNotFoundError(f"Path does not exist: {path}")

    def _load_json(self, file_path: Path) -> None:
        """
        Loads a single JSON file and converts it into a Reading object.

        :param file_path: Path object representing the file to load.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                # TODO: good way to incorporate a schema?
                relationships = Relationships()
                for relationship in data["relationships"]:
                    relationships.add_relation(
                        relationship["entity_1"],
                        relationship["entity_2"],
                        relationship["relationship"],
                    )
                reading = Reading(
                    title=data["title"],
                    abstract=data["abstract"],
                    entities=data["entities"],
                    relationship_types=set(data["relationship_types"]),
                    relationships=relationships,
                )
                self.readings.append(reading)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {file_path}")

    def get_data(self) -> List[Reading]:
        """
        Returns the loaded Reading objects.

        :return: A list of Reading objects.
        """
        return self.readings
