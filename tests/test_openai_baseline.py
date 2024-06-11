from relation_suite.data.input_data import InputData
from constants import *
from relation_suite.extractors.openai_extractor_baseline import OpenAIExtractorBaseline
from relation_suite.utils.benchmark import (
    Benchmark,
    extract_relationships_from_input_data,
)

MODELS = {
    "gpt-4-turbo": {"type": "json_object"},
    # "gpt-4o": {"type": "json_object"},
    # "gpt-4": None,
    "gpt-3.5-turbo": {"type": "json_object"},
    "ft:gpt-3.5-turbo-1106:syrgkanislab:relation-ft-3-2:9VEjEX91": {
        "type": "json_object"
    },
}

DIR_NAME = "desert_ecology"

if __name__ == "__main__":
    input_data = InputData(SYNTHETIC_DIR + f"/{DIR_NAME}/")
    ground_truth = extract_relationships_from_input_data(input_data)
    ground_truth.plot_digraph(f"{SYNTHETIC_DIR}/{DIR_NAME}/ground_truth.png")

    performance = {}
    for model_name in MODELS:
        print(f"Testing LlamaExtractor with model: {model_name}")
        extractor = OpenAIExtractorBaseline(model=model_name)
        relationships = extractor.extract(input_data)
        print()
        print(f"Relationships extracted from {model_name}:")
        print(relationships.__str__(), "\n")

        benchmark = Benchmark(ground_truth, relationships)
        precision, recall, f1 = (
            benchmark.calculate_precision(),
            benchmark.calculate_recall(),
            benchmark.calculate_f1(),
        )
        performance[model_name] = (precision, recall, f1)
        fp, fn = benchmark.get_fp_fn()
        print("False positives:")
        print(fp.__str__(), "\n")
        print("False negatives:")
        print(fn.__str__(), "\n")
        relationships.plot_digraph(
            f"{SYNTHETIC_DIR}/{DIR_NAME}/{model_name}_baseline.png",
            colored_relationships=[(fp, "orange")],
        )
        ground_truth.plot_digraph(
            f"{SYNTHETIC_DIR}/{DIR_NAME}/{model_name}_baseline_fn.png",
            colored_relationships=[(fn, "blue")],
        )

    for model_name, (precision, recall, f1) in performance.items():
        print(
            f"Model: {model_name}, Precision: {precision}, Recall: {recall}, F1: {f1}"
        )
