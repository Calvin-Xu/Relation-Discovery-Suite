from relation_suite.utils.finetune_data_checker import OpenAIFinetuneDataCheck
from relation_suite.utils.finetune_file_generator import FinetuneFileGenerator
from constants import SYNTHETIC_DIR

system_prompt = """You are an expert in determining causal and noncausal relationships from academic literature. I need your help to read a paper abstract and inform me of the relationships of certain types between certain entities.

I will provide you with the paper's title, abstract, the entities that have relationships with one another, and the types of relationships possible. Tell me your answer in json format.
For example, if the types of relationships are {"cause", "inhibit", "positively correlate", "negatively correlate"}, and the paper says "the presence of green spaces within urban environments not only inhibits the adverse effects of air pollution on respiratory health but also positively correlates with improvements in mental well-being", then the output should be:
{"Relationships": [{"A": "green spaces", "B": "air pollution", "Relation": "inhibit"}, {"A": "green spaces", "B": "mental well-being", "VERB": "positively correlate"}]}

Be exhaustive and identify all plausible relationships."""

user_template = "Title: {title}\nAbstract: {abstract}\nEntities: {entities}\nRelationship Types: {relationship_types}"
assistant_template = "Relationships: {relationships}"
keys = ["title", "abstract", "entities", "relationship_types", "relationships"]

generator = FinetuneFileGenerator(
    system_prompt=system_prompt,
    user_template=user_template,
    assistant_template=assistant_template,
    keys=keys,
)

dir_paths = [
    SYNTHETIC_DIR + "/biomedical",
    SYNTHETIC_DIR + "/marine_ecology_2",
    SYNTHETIC_DIR + "genetic",
]
output_file = "relationship_finetune_1.jsonl"
generator.generate_training_file(dir_paths, output_file)


data_check = OpenAIFinetuneDataCheck("relationship_finetune_1.jsonl")
data_check.check_format()
data_check.analyze_data()
data_check.estimate_cost(price_per_million_tokens=8.00)
