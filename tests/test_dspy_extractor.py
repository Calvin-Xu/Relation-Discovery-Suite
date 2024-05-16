from relation_suite.data.input_data import InputData
from relation_suite.data.reading import Reading
from relation_suite.extractors.dspy_extractor import CoT, DSPyExtractor
from relation_suite.models.relationships import Relationships
from relation_suite.utils.benchmark import (
    Benchmark,
    extract_relationships_from_input_data,
)
from constants import *
from dspy.evaluate import Evaluate
import dspy

DIR_NAME = "biomedical"

if __name__ == "__main__":
    input_data = InputData(SYNTHETIC_DIR + f"/{DIR_NAME}/")
    ground_truth = extract_relationships_from_input_data(input_data)
    # ground_truth.plot_digraph(f"{SYNTHETIC_DIR}/{DIR_NAME}/ground_truth.png")

    # extractor = DSPyExtractor()
    # extractor.extract(input_data)

    cot_fewshot = CoT()
    cot_fewshot.load("cot.json")

    llm = dspy.OpenAI(model="gpt-4-turbo", max_tokens=4096)
    dspy.settings.configure(lm=llm)

    examples = InputData(SYNTHETIC_DIR + f"/marine_ecology_2")
    reading: Reading = examples.get_data()[0]
    print(reading.to_json())
    print(cot_fewshot(reading.title, reading.abstract, str(reading.entities)))

    test = [
        dspy.Example(
            title=reading.title,
            abstract=reading.abstract,
            entities=str(reading.entities),
            relationship_types=str(reading.relationship_types),
            relationships=reading.relationships.to_json(),
        ).with_inputs("title", "abstract", "entities")
        for reading in examples.get_data()
    ]

    evaluator = Evaluate(
        devset=test, num_threads=2, display_progress=True, display_table=5
    )

    def metric(gold, pred, trace=None):
        original = Relationships()
        original.from_json(gold.relationships)
        ours = Relationships()
        ours.from_json(pred.relationships)
        benchmark = Benchmark(original, ours)
        return benchmark.calculate_f1()
        return 0

    print(evaluator(cot_fewshot, metric=metric))
