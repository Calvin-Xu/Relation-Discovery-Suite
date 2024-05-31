import json
import tiktoken
import numpy as np
from collections import defaultdict


class OpenAIFinetuneDataCheck:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = self.load_data()
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.format_errors = defaultdict(int)
        self.n_missing_system = 0
        self.n_missing_user = 0
        self.n_messages = []
        self.convo_lens = []
        self.assistant_message_lens = []
        self.MAX_TOKENS_PER_EXAMPLE = 4096
        self.TARGET_EPOCHS = 3
        self.MIN_TARGET_EXAMPLES = 100
        self.MAX_TARGET_EXAMPLES = 25000
        self.MIN_DEFAULT_EPOCHS = 1
        self.MAX_DEFAULT_EPOCHS = 25

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def check_format(self):
        for ex in self.dataset:
            if not isinstance(ex, dict):
                self.format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if not messages:
                self.format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    self.format_errors["message_missing_key"] += 1

                if any(
                    k not in ("role", "content", "name", "function_call", "weight")
                    for k in message
                ):
                    self.format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in (
                    "system",
                    "user",
                    "assistant",
                    "function",
                ):
                    self.format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                function_call = message.get("function_call", None)

                if (not content and not function_call) or not isinstance(content, str):
                    self.format_errors["missing_content"] += 1

            if not any(
                message.get("role", None) == "assistant" for message in messages
            ):
                self.format_errors["example_missing_assistant_message"] += 1

        if self.format_errors:
            print("Found errors:")
            for k, v in self.format_errors.items():
                print(f"{k}: {v}")
        else:
            print("No errors found")

    def num_tokens_from_messages(
        self, messages, tokens_per_message=3, tokens_per_name=1
    ):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(self, messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(self.encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(self, values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    def analyze_data(self):
        for ex in self.dataset:
            messages = ex["messages"]
            if not any(message["role"] == "system" for message in messages):
                self.n_missing_system += 1
            if not any(message["role"] == "user" for message in messages):
                self.n_missing_user += 1
            self.n_messages.append(len(messages))
            self.convo_lens.append(self.num_tokens_from_messages(messages))
            self.assistant_message_lens.append(
                self.num_assistant_tokens_from_messages(messages)
            )

        print("Num examples missing system message:", self.n_missing_system)
        print("Num examples missing user message:", self.n_missing_user)
        self.print_distribution(self.n_messages, "num_messages_per_example")
        self.print_distribution(self.convo_lens, "num_total_tokens_per_example")
        self.print_distribution(
            self.assistant_message_lens, "num_assistant_tokens_per_example"
        )
        n_too_long = sum(l > self.MAX_TOKENS_PER_EXAMPLE for l in self.convo_lens)
        print(
            f"\n{n_too_long} examples may be over the {self.MAX_TOKENS_PER_EXAMPLE} token limit, they will be truncated during fine-tuning"
        )

    def estimate_cost(self):
        n_train_examples = len(self.dataset)
        n_epochs = self.TARGET_EPOCHS
        if n_train_examples * self.TARGET_EPOCHS < self.MIN_TARGET_EXAMPLES:
            n_epochs = min(
                self.MAX_DEFAULT_EPOCHS, self.MIN_TARGET_EXAMPLES // n_train_examples
            )
        elif n_train_examples * self.TARGET_EPOCHS > self.MAX_TARGET_EXAMPLES:
            n_epochs = max(
                self.MIN_DEFAULT_EPOCHS, self.MAX_TARGET_EXAMPLES // n_train_examples
            )

        n_billing_tokens_in_dataset = sum(
            min(self.MAX_TOKENS_PER_EXAMPLE, length) for length in self.convo_lens
        )
        print(
            f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
        )
        print(f"By default, you'll train for {n_epochs} epochs on this dataset")
        print(
            f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
        )
