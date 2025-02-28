#TODO: Add cli arguments for
# Max tokens
# Log file
#TODO: Change prompts for user simulation

from vllm import LLM, SamplingParams
import json
from os import path
import pandas as pd
import argparse

def load_data(data_dir):
    #each row is in json
    queries_path = path.join(data_dir, "queries.json")
    #a csv file with two columns
    sumrels_path = path.join(data_dir, "sumrels.csv")
    #a tsv file with 4 columns
    qrels_path = path.join(data_dir, "qrels.txt")
    #a json file with 7 attributes
    d_infos_path = path.join(data_dir, "documents_info.json")

    queries = []
    with open(queries_path, "r") as f:
        for line in f:
            queries.append(json.loads(line))

    sumrels = pd.read_csv(sumrels_path)
    qrels = pd.read_csv(qrels_path, sep='\t', names=["qid", '0', "did", "1"])

    d_infos = pd.read_json(d_infos_path)

    return queries, sumrels, qrels, d_infos


def load_prompts():
    with open("prompts.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_query_did(query, qrels):
    qid = query["id"]
    did = qrels[qrels['qid'] == qid]["did"].item()
    return did

def has_summary(did, sumrels):
    sumrel = sumrels[sumrels['id'] == did]
    if sumrel["has_summary"].item() == 1:
        return True
    else:
        return False

def get_gt_info(query, sumrels, qrels, d_infos):
    did = get_query_did(query, qrels)
    if not has_summary(did, sumrels):
        return None
    gt_info = d_infos[d_infos['id'] == did]
    return gt_info

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--max_tokens", type=int, default=500)
parser.add_argument("--max_rounds", type=int, default=5)

args = parser.parse_args()

DATA_DIR = args.data_dir
OUT_DIR = args.out_dir
MAX_TOKENS = args.max_tokens
MAX_ROUNDS = args.max_rounds

MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

prompt_formats = load_prompts()
SYSTEM_INSTRUCTION = prompt_formats["system_instruction"]
SYSTEM_PROMPT = prompt_formats['llama3_system_prompt_format']
INTERACTION_PROMPT = prompt_formats['llama3_interaction_prompt_format']

model = LLM(model=MODEL_NAME, tensor_parallel_size=1, trust_remote_code=True)

sp = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=MAX_TOKENS)

def interact(query, gt_info, max_rounds):
    print(f"GT: {gt_info['title'].item()}\n")
    print(f"Initial Question:\n{query['description']}")
    system_prompt = SYSTEM_PROMPT.format(system_instruction=SYSTEM_INSTRUCTION.format(summary=gt_info['summary'].item()))
    print(system_prompt)
    interaction_history = ""
    interaction_history_list = []
    num_rounds = 0
    while(num_rounds < max_rounds):
        user_input = input("Query: ")
        if user_input == "end":
            break
        current_interaction = INTERACTION_PROMPT.format(user_prompt=user_input)
        outputs = model.generate(prompts=[system_prompt + interaction_history + current_interaction], sampling_params=sp)
        outputs = [output.outputs[0].text for output in outputs]
        response = outputs[0]
        print(f"Response:\n{response}")
        interaction_history += current_interaction + response
        interaction_history_list.append(f"Query: {user_input}")
        interaction_history_list.append(f"Response: {response}")
        num_rounds += 1

    print(f"INTERACTION {query['id']} END")
    user_input = input("IS CORRECT: ")
    got_answer = False
    if user_input.lower().strip() in ['y' or 'yes']:
        got_answer = True

    log_path = path.join(OUT_DIR, f"interaction_history_{query['id']}.txt")
    with open(log_path, "w") as f:
        out_obj = {
            "title" : gt_info['title'].item(),
            "init_query" : query['description'],
            "interaction_history" : interaction_history,
            "rounds" : num_rounds,
            "success" : got_answer
            }
        json.dump(out_obj, f)
    return log_path


queries, sumrels, qrels, d_infos = load_data(DATA_DIR)

qids = []
res_paths = []
for query in queries:
    gt_info = get_gt_info(query, sumrels, qrels, d_infos)
    if not gt_info is None:
        log_path = interact(query, gt_info, MAX_ROUNDS)
        qids.append(query['id'])
        res_paths.append(log_path)
    else:
        continue

with open(path.join(OUT_DIR, f"a_meta.csv"), "w") as f:
    f.write("qid,paths\n")
    for id, p in zip(qids, res_paths):
        f.write(f"{id},{p}\n")