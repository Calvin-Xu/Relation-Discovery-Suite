from openai import OpenAI

client = OpenAI()

client.files.create(
    file=open("relationship_finetune_1.jsonl", "rb"), purpose="fine-tune"
)

print(client.files.list())

client.fine_tuning.jobs.create(
    training_file="relationship_finetune_1", model="gpt-3.5-turbo"
)
