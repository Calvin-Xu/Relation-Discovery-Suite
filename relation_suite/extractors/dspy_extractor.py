import json
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
        _relationships.from_json(answer.relationships)
        return dspy.Prediction(
            relationships=_relationships.to_json(),
        )


class DSPyExtractor(Extractor):
    def __init__(
        self,
        name: str = None,
        llm: "dspy.LM" = dspy.OpenAI(model="gpt-4-turbo", max_tokens=4096),
        training_data: InputData = None,
    ) -> None:
        self.llm = llm
        dspy.settings.configure(lm=self.llm)

        self.cot_fewshot = CoT()
        if name:
            try:
                self.cot_fewshot.load(name)
            except Exception as e:
                self.generate_predictor(name, training_data)
        else:
            self.generate_predictor("cot_fewshot_predictor", training_data)

    def generate_predictor(self, name: str, training_data: InputData) -> None:
        examples = [
            dspy.Example(
                title=reading.title,
                abstract=reading.abstract,
                entities=str(reading.entities),
                relationship_types=str(reading.relationship_types),
                relationships=reading.relationships.to_json(),
            ).with_inputs("title", "abstract", "entities")
            for reading in training_data.get_data()
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
        cot_fewshot = bootstrap_optimizer.compile(
            self.cot_fewshot, trainset=train, valset=dev
        )

        cot_fewshot.save(f"{name}.json")
        self.cot_fewshot = cot_fewshot

    def metric(self, gold, pred, trace=None):
        original = Relationships()
        original.from_json(gold.relationships)
        ours = Relationships()
        ours.from_json(pred.relationships)
        benchmark = Benchmark(original, ours)
        return benchmark.calculate_f1()

    def extract(self, input_data: InputData) -> Relationships:
        aggregated_relationships = Relationships()
        n_readings = len(input_data.get_data())
        for i, reading in enumerate(input_data.get_data()):
            print(f"\nReading {i+1}/{n_readings}: {reading.title}")
            result = self.cot_fewshot.predict(
                title=reading.title,
                abstract=reading.abstract,
                entities=str(reading.entities),
            )
            if result.relationships:
                reading_relationships = Relationships()
                reading_relationships.from_json(result.relationships)
                for (
                    entity_1,
                    entity_2,
                ), relationship in reading_relationships.get_all().items():
                    aggregated_relationships.add(entity_1, entity_2, relationship)
        return aggregated_relationships
