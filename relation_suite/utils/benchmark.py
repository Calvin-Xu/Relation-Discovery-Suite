from relation_suite.data.input_data import InputData
from relation_suite.models.relationships import Relationships


class Benchmark:
    @staticmethod
    def calculate_precision_recall(
        input_data: InputData, system_relationships: Relationships
    ):
        # Step 1: Extract ground truth relationships from input data
        ground_truths = set()
        for (
            reading
        ) in (
            input_data.get_data()
        ):  # Utilizing the get_data() method to obtain Reading objects
            for rel_key in reading.relationships.relationships.keys():
                ground_truths.add(rel_key)

        # Step 2: Compare with system-found relationships
        system_found = set(system_relationships.relationships.keys())
        correct_identified = ground_truths.intersection(system_found)

        # Step 3: Calculate precision and recall
        precision = len(correct_identified) / len(system_found) if system_found else 0
        recall = len(correct_identified) / len(ground_truths) if ground_truths else 0

        return precision, recall
