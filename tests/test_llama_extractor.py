from relation_suite.extractors.llama_extractor import LlamaExtractor
from relation_suite.data.input_data import InputData
from constants import *
from relation_suite.utils.benchmark import (
    Benchmark,
    extract_relationships_from_input_data,
)

MODELS = {
    "mistral7b:instruct": "/Users/calvinxu/.ollama/models/blobs/sha256-e8a35b5937a5e6d5c35d1f2a15f161e07eefe5e5bb0a3cdd42998ee79b057730"
}

if __name__ == "__main__":
    input_data = InputData(SYNTHETIC_DIR + "/marine_ecology/")
    ground_truth = extract_relationships_from_input_data(input_data)
    ground_truth.plot_digraph(f"{SYNTHETIC_DIR}/marine_ecology/ground_truth.png")

    performance = {}
    for model_name, model_path in MODELS.items():
        print(f"Testing LlamaExtractor with model: {model_name}")
        extractor = LlamaExtractor(model_path=model_path)
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
            f"{SYNTHETIC_DIR}/marine_ecology/{model_name}.png",
            colored_relationships=[(fp, "orange")],
        )
        ground_truth.plot_digraph(
            f"{SYNTHETIC_DIR}/marine_ecology/{model_name}_fn.png",
            colored_relationships=[(fn, "blue")],
        )

    for model_name, (precision, recall, f1) in performance.items():
        print(
            f"Model: {model_name}, Precision: {precision}, Recall: {recall}, F1: {f1}"
        )
