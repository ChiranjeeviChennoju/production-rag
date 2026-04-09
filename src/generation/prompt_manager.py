import yaml
from pathlib import Path


class PromptManager:
    """Load and format versioned prompt templates from YAML files."""

    def __init__(self, prompts_dir: str = "./prompts"):
        self.prompts_dir = Path(prompts_dir)

    def load_prompt(self, name: str, version: str = "v1") -> str:
        """Load a prompt template by name and version, e.g. ('rag_prompt', 'v1')."""
        path = self.prompts_dir / f"{name}_{version}.yaml"
        with open(path) as f:
            config = yaml.safe_load(f)
        return config["template"]

    @staticmethod
    def format_prompt(template: str, context: str, question: str) -> str:
        """Fill in {context} and {question} placeholders."""
        return template.format(context=context, question=question)
