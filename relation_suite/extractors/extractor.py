from abc import ABC, abstractmethod
from pathlib import Path


class Extractor(ABC):
    @abstractmethod
    def extract(self, input_data: "InputData") -> "Relationships":
        """
        Extracts relationships from the given input data.

        :param input_data: An InputData instance containing the data to be processed.
        :return: A Relationships object containing the extracted relationships.
        """
        pass
