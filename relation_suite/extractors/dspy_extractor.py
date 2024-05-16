import json
from typing import List
import dspy
from relation_suite.data.input_data import InputData
from relation_suite.extractors.extractor import Extractor
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from relation_suite.models.relationships import Relationships
from relation_suite.utils.benchmark import Benchmark


class FindRelations(dspy.Signature):
    __doc__ = f"""You are an expert in determining causal and noncausal relationships from academic literature. I need your help to read a paper abstract and inform me of the relationships of certain types between certain entities.\nI will provide you with the paper's title, abstract, the entities that have relationships with one another, and the types of relationships possible. Tell me your answer in json format. Be exhaustive and identify all plausible relationships. If you cannot find any or cannot answer the question, return no relationships."""

    title = dspy.InputField()
    abstract = dspy.InputField()
    entities = dspy.InputField()
    relationships = dspy.OutputField(
        desc="""json encoding the relationships between entities present in the entities list. Entities names must be reported verbatim."""
    )


class CoT(dspy.Module):
    def __init__(self, num_preds=1):
        super().__init__()
        self.predict = dspy.ChainOfThought(FindRelations, n=num_preds)

    def forward(self, title: str, abstract: str, entities: str):
        _relationships = Relationships()
        answer = self.predict(
            title=title,
            abstract=abstract,
            entities=entities,
        )
        if answer.relationships.split("\n")[0] == "```json":
            answer.relationships = "\n".join(answer.relationships.split("\n")[1:-1])
        # print(answer.relationships)
        _relationships.from_json(answer.relationships)
        # return _relationships
        return dspy.Prediction(
            relationships=_relationships.to_json(),
        )


class DSPyExtractor(Extractor):

    def __init__(self) -> None:
        self.llm = dspy.OpenAI(model="gpt-4-turbo", max_tokens=4096)
        dspy.settings.configure(lm=self.llm)

    def metric(self, gold, pred, trace=None):
        original = Relationships()
        original.from_json(gold.relationships)
        ours = Relationships()
        ours.from_json(pred.relationships)
        benchmark = Benchmark(original, ours)
        return benchmark.calculate_f1()
        return 0

    def extract(self, input_data: InputData):
        examples = [
            dspy.Example(
                title=reading.title,
                abstract=reading.abstract,
                entities=str(reading.entities),
                relationship_types=str(reading.relationship_types),
                relationships=reading.relationships.to_json(),
            ).with_inputs("title", "abstract", "entities")
            for reading in input_data.get_data()
        ]

        train, dev = examples[: len(examples) - 10], examples[len(examples) - 10 :]

        bootstrap_optimizer = BootstrapFewShotWithRandomSearch(
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
            num_candidate_programs=10,
            num_threads=8,
            metric=self.metric,
            teacher_settings=dict(lm=self.llm),
        )
        cot_fewshot = bootstrap_optimizer.compile(CoT(), trainset=train, valset=dev)

        cot_fewshot.save("cot2.json")
