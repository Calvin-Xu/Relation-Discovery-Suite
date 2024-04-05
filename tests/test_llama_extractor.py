from relation_suite.extractors.llama_extractor import LlamaExtractor
from relation_suite.data.input_data import InputData
from constants import *
from relation_suite.utils.benchmark import Benchmark

MODELS = {
    "mistral7b:instruct": "/Users/calvinxu/.ollama/models/blobs/sha256-e8a35b5937a5e6d5c35d1f2a15f161e07eefe5e5bb0a3cdd42998ee79b057730",
}

if __name__ == "__main__":
    input_data = InputData()
    input_data.load_data(SYNTHETIC_DIR)
    # for reading in input_data.readings:
    #     print(reading.to_json())

    performance = {}
    for model_name, model_path in MODELS.items():
        print(f"Testing LlamaExtractor with model: {model_name}")
        extractor = LlamaExtractor(model_path=model_path)
        relationships = extractor.extract(input_data)
        print(f"Relationships extracted from {model_name}:")
        print(relationships.to_json())

        benchmark_tool = Benchmark()
        precision, recall = benchmark_tool.calculate_precision_recall(
            input_data, relationships
        )
        performance[model_name] = (precision, recall)

    for model_name, (precision, recall) in performance.items():
        print(f"Model: {model_name}, Precision: {precision}, Recall: {recall}")
