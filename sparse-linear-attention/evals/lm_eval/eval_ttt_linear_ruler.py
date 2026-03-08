import json

from lm_eval import evaluator
from sparse_linear_attention.models import ttt

results = evaluator.simple_evaluate(
    model="hf",
    model_args=(
        "pretrained=models/linear_chunk_64_lr0#1_340M,dtype=float32,max_length=16384,trust_remote_code=True,add_bos_token=True,"
        "tokenizer=fla-hub/delta_net-1.3B-100B,"
        "trust_remote_code=true"
    ),
    tasks=[
        "niah_single_1", "niah_single_2", "niah_single_3"
    ],
    batch_size="auto",
    metadata={"max_seq_lengths": [1024, 2048, 4096]},
)

with open("results/linear_chunk_64_lr0#1_340M.json", "w") as f:
    json.dump(results["results"], f, indent=2)
