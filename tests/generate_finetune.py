from relation_suite.extractors.openai_extractor_baseline import OpenAIExtractorBaseline
from relation_suite.utils.finetune_data_checker import OpenAIFinetuneDataCheck
from relation_suite.models.finetune_file import FinetuneFile
from constants import SYNTHETIC_DIR

system_prompt = OpenAIExtractorBaseline.DEFAULT_PROMPT

user_template = "Title: {title}\nAbstract: {abstract}\nEntities: {entities}\nRelationship Types: {relationship_types}"
assistant_template = '{{"Relationships": {relationships}}}'
keys = ["title", "abstract", "entities", "relationship_types", "relationships"]

train_file = FinetuneFile(
    "data/relationship_finetune_3_train.jsonl",
    [
        SYNTHETIC_DIR + "/biomedical",
        SYNTHETIC_DIR + "/marine_ecology_2",
        SYNTHETIC_DIR + "/genetic",
    ],
    system_prompt=system_prompt,
    user_template=user_template,
    assistant_template=assistant_template,
    keys=keys,
)

val_file = FinetuneFile(
    "data/relationship_finetune_3_val.jsonl",
    [
        SYNTHETIC_DIR + "/desert_ecology",
    ],
    system_prompt=system_prompt,
    user_template=user_template,
    assistant_template=assistant_template,
    keys=keys,
)

for file in [train_file, val_file]:
    file.generate()
    data_check = OpenAIFinetuneDataCheck(file.output_path)
    data_check.check_format()
    data_check.analyze_data()
    data_check.estimate_cost(price_per_million_tokens=8.00)
