from relation_suite.data.input_data import InputData
from relation_suite.data.reading import Reading
from relation_suite.extractors.dspy_extractor import DSPyExtractor
from relation_suite.models.relationships import Relationships
from relation_suite.utils.benchmark import (
    Benchmark,
    extract_relationships_from_input_data,
)
from constants import SYNTHETIC_DIR
import matplotlib.pyplot as plt
import pandas as pd
import dspy
from great_tables import GT

TRAIN_DIR_NAME = "biomedical"
TEST_DIR_NAME = "marine_ecology_2"


def metric(gold, pred, trace=None):
    original = Relationships()
    original.from_json(gold.relationships)
    ours = Relationships()
    ours.from_json(pred.relationships)
    benchmark = Benchmark(original, ours)
    return benchmark.calculate_f1()


if __name__ == "__main__":
    test_input_data = InputData(SYNTHETIC_DIR + f"/{TEST_DIR_NAME}/")
    ground_truth = extract_relationships_from_input_data(test_input_data)

    train_input_data = InputData(SYNTHETIC_DIR + f"/{TRAIN_DIR_NAME}/")

    MODELS = ["gpt-4-turbo", "gpt-4o", "gpt-4", "gpt-3.5-turbo-0125"]
    performance = {}

    for model in MODELS:
        extractor = DSPyExtractor(
            name=f"cot_fewshot_predictor_{model}",
            training_data=train_input_data,
            llm=dspy.OpenAI(model=model, max_tokens=4096),
        )

        extracted_relationships = extractor.extract(test_input_data)

        benchmark = Benchmark(ground_truth, extracted_relationships)
        precision, recall, f1 = (
            benchmark.calculate_precision(),
            benchmark.calculate_recall(),
            benchmark.calculate_f1(),
        )
        print(f"Model: {model}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        fp, fn = benchmark.get_fp_fn()
        print("False positives:")
        print(fp.__str__(), "\n")
        print("False negatives:")
        print(fn.__str__(), "\n")
        if fp.get_num_relationships() > 0:
            extracted_relationships.plot_digraph(
                f"{SYNTHETIC_DIR}/{TEST_DIR_NAME}/{model}.png",
                colored_relationships=[(fp, "orange")],
            )
        if fn.get_num_relationships() > 0:
            ground_truth.plot_digraph(
                f"{SYNTHETIC_DIR}/{TEST_DIR_NAME}/{model}_fn.png",
                colored_relationships=[(fn, "blue")],
            )

        performance[model] = (precision, recall, f1)

        performance_df = pd.DataFrame.from_dict(
            performance, orient="index", columns=["Precision", "Recall", "F1"]
        )

    table = GT(performance_df)
    table.save(f"{SYNTHETIC_DIR}/{TEST_DIR_NAME}/performance_table.png")

    # Plot performance as a bar graph
    performance_df.plot(kind="bar", figsize=(10, 8))
    plt.title("Model Performance on Test Data")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.xticks(rotation=45)
    plt.grid(axis="y")

    # Save graph as an image
    plt.savefig(f"{SYNTHETIC_DIR}/{TEST_DIR_NAME}/performance_graph.png", dpi=300)
