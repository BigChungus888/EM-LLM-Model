import json

from lm_eval import evaluator
from sparse_linear_attention.models import ttt

results = evaluator.simple_evaluate(
    model="hf",
    model_args=(
        "pretrained=models/linear_chunk_64_lr0#1_340M,"
        "tokenizer=fla-hub/delta_net-1.3B-100B,"
        "trust_remote_code=true"
    ),
    tasks=[
        "wikitext", "lambada_openai",
        "piqa", "hellaswag", "winogrande",
        "arc_easy", "arc_challenge", "boolq",
        "social_iqa",
    ],
    batch_size="auto"
)

with open("results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
