from relation_suite.extractors.llama_extractor import LlamaExtractor
from relation_suite.data.input_data import InputData
from constants import *
from relation_suite.utils.benchmark import Benchmark

MODELS = {
    "mistral7b:instruct": "/Users/calvinxu/.ollama/models/blobs/sha256-e8a35b5937a5e6d5c35d1f2a15f161e07eefe5e5bb0a3cdd42998ee79b057730"
}

if __name__ == "__main__":
    input_data = InputData(SYNTHETIC_DIR)
    # for reading in input_data.readings:
    #     print(reading.to_json())

    performance = {}
    for model_name, model_path in MODELS.items():
        print(f"Testing LlamaExtractor with model: {model_name}")
        extractor = LlamaExtractor(model_path=model_path)
        relationships = extractor.extract(input_data)
        print()
        print(f"Relationships extracted from {model_name}:")
        print(relationships.__str__(), "\n")

        benchmark = Benchmark(input_data, relationships)
        precision, recall, f1 = (
            benchmark.calculate_precision(),
            benchmark.calculate_recall(),
            benchmark.calculate_f1(),
        )
        performance[model_name] = (precision, recall, f1)
        print("False positives:")
        print(benchmark.get_false_positives().__str__(), "\n")

    for model_name, (precision, recall, f1) in performance.items():
        print(
            f"Model: {model_name}, Precision: {precision}, Recall: {recall}, F1: {f1}"
        )
