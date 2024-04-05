from relation_suite.data.input_data import InputData
from relation_suite.models.relationships import Relationships
from typing import Union, Set, Tuple


from typing import Union, Set, Tuple, Dict


class Benchmark:
    def __init__(
        self,
        ground_truth: Union[InputData, Relationships],
        our_result: Relationships,
    ) -> None:
        """
        Initializes the benchmarking tool with either an InputData object and a Relationships object,
        or two Relationships objects for direct comparison.

        :param ground_truth: An InputData object containing the ground truth, or a Relationships object for direct comparison.
        :param our_result: The Relationships object to be evaluated.
        """
        if isinstance(ground_truth, InputData):
            self.ground_truth_relationships = (
                self._extract_relationships_from_input_data(
                    ground_truth
                ).get_relationships()
            )
        elif isinstance(ground_truth, Relationships):
            self.ground_truth_relationships = ground_truth.get_relationships()
        else:
            raise ValueError(
                "ground_truth must be either an InputData or Relationships object"
            )

        self.system_relationships = our_result.get_relationships()
        self.correct_identified = {
            (entity_1, entity_2): relationship
            for (entity_1, entity_2), relationship in self.system_relationships.items()
            if self.ground_truth_relationships.get((entity_1, entity_2)) == relationship
        }

    def _extract_relationships_from_input_data(
        self, input_data: InputData
    ) -> Relationships:
        """
        Extracts relationships from an InputData object and returns them in a Relationships object.

        :param input_data: The InputData object containing ground truth relationships.
        :return: A Relationships object populated with the extracted relationships.
        """
        extracted_relationships = Relationships()
        for reading in input_data.get_data():
            for (
                entity_1,
                entity_2,
            ), relationship in reading.relationships.get_relationships().items():
                extracted_relationships.add_relation(entity_1, entity_2, relationship)
        return extracted_relationships

    def calculate_precision(self) -> float:
        """
        Calculates the precision of the system's output.

        :return: The precision as a float.
        """
        return (
            len(self.correct_identified) / len(self.system_relationships)
            if self.system_relationships
            else 0
        )

    def calculate_recall(self) -> float:
        """
        Calculates the recall of the system's output.

        :return: The recall as a float.
        """
        return (
            len(self.correct_identified) / len(self.ground_truth_relationships)
            if self.ground_truth_relationships
            else 0
        )

    def calculate_f1(self) -> float:
        """
        Calculates the F1 score of the system's output.

        :return: The F1 score as a float.
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0
        )

    def get_false_positives(self) -> Relationships:
        """
        Identifies the relationships that were incorrectly identified by the system,
        i.e., the system-identified relationships that do not match the ground truth.

        :return: A Relationships object containing the false positives.
        """
        ground_truth = {
            (entity_1, entity_2, relationship)
            for (
                entity_1,
                entity_2,
            ), relationship in self.ground_truth_relationships.items()
        }
        our_result = {
            (entity_1, entity_2, relationship)
            for (entity_1, entity_2), relationship in self.system_relationships.items()
        }

        false_positive_tuples = our_result - ground_truth

        false_positives = Relationships()
        false_positives.populate(false_positive_tuples)
        return false_positives
