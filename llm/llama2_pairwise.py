import sys
import os
from typing import List, Optional
import fire
from llama.llama import Llama, Dialog
from tqdm import tqdm
import json
from collections import defaultdict
import re
import argparse
import random
import time
import copy
from utils import *

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 8,
    resume_eval: bool = False,
    test: bool = False,
    shot: int = 1,
    partition: int = 0,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    if test: 
        output_dir = f"./output/llama2_pair_wise/test"
        if os.path.exists(os.path.join(output_dir, f"log_{partition}.txt")):
            os.remove(os.path.join(output_dir, f"log_{partition}.txt"))
        sys.stdout = open(os.path.join(output_dir, f"log_{partition}.txt"), 'w')
    else:
        output_dir = f"./output/llama2_pair_wise/valid"
        if os.path.exists(os.path.join(output_dir, "log.txt")):
            os.remove(os.path.join(output_dir, "log.txt"))
        sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = "../data/MAVEN_ERE"

    resume_doc_number = 0
    if resume_eval:
        if test:
            with open(os.path.join(output_dir, f"test_prediction_{partition}.jsonl"), "r")as f:
                lines = f.readlines()
                for line in lines:
                    resume_doc_number += 1
        else:
            with open(os.path.join(output_dir, f"test_prediction.jsonl"), "r")as f:
                lines = f.readlines()
                for line in lines:
                    resume_doc_number += 1

    if not test:
        docs = valid_first_10_doc_process(data_dir)
    else:
        docs = test_doc_process(data_dir)
    
    example_doc = example_doc_process()
    if test:
        batch_size = int(len(lines) / 10) + 1
        start = partition * batch_size
        end = start + batch_size if partition != 9 else len(lines)
    else:
        start = 0
        end = len(docs)

    for i, doc in enumerate(tqdm(docs[start:end], desc="Predicting")):
        if resume_eval and i < resume_doc_number:
            continue
        #if i == 1:
        #    exit()
        result = {"id": doc["id"], "coreference": None, "temporal_relations": {}, "causal_relations": {}, "subevent_relations": None}
        result["coreference"] = []
        result["temporal_relations"] = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
        result["causal_relations"] = {"CAUSE": [], "PRECONDITION": []}
        result["subevent_relations"] = []
        doc["coreference_raw_output"] = []
        doc["temporal_raw_output"] = []
        doc["causal_raw_output"] = []
        doc["subevent_raw_output"] = []
        print(example_doc)
        #exit()
        for relation in ["coreference", "temporal", "causal", "subevent"]:
            if relation == "coreference":
                cluster_appeared = []
                for mention_i in doc['event_mentions']:
                    mention_i = mention_i['id_in_number']
                    for mention_j in doc['event_mentions']:
                        #print(mention_i, mention_j)
                        mention_j = mention_j['id_in_number']
                        if mention_i != mention_j:
                            
                            print(mention_i, mention_j)

                            prompt_context = example_doc["context"]
                            prompt_context += f" Do Event_0 and Event_1 refer to the same event?"
                            input_context = doc["context"]
                            input_context += f" Do {mention_i} and {mention_j} refer to the same event?"

                            dialogs: List[Dialog] = [[{"role": "system", "content": "Predict the relation given two events, answer with one word from [Yes, No]"},
                                {"role": "user", "content": f"{prompt_context}"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_0 and Event_3 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_0 and Event_5 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_2 and Event_21 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_21 and Event_26 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_10 and Event_0 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_11 and Event_5 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_12 and Event_21 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_7 and Event_23 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Do Event_8 and Event_6 refer to the same event?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"{input_context} Do not reply using a complete sentence, and only give the answer from: [Yes, No]."}]]

                            results = generator.chat_completion(
                                dialogs,  # type: ignore
                                max_gen_len=max_gen_len,
                                temperature=temperature,
                                top_p=top_p,
                            )
                            pred = results[0]['generation']['content'].strip()

                            print(pred)
                            if pred == "Yes":
                                pair = [mention_i, pred, mention_j]
                                in_cluster = False
                                for k, cluster in enumerate(result["coreference"]):
                                    if pair[0] in cluster and pair[0] in doc["eventnumberid2mention"]:
                                        result["coreference"][k].append(pair[2])
                                        in_cluster = True
                                    elif pair[2] in cluster and pair[2] in doc["eventnumberid2mention"]:
                                        result["coreference"][k].append(pair[0])
                                        in_cluster = True
                                if not in_cluster and pair[0] in doc["eventnumberid2mention"] and pair[2] in doc["eventnumberid2mention"]:
                                    result["coreference"].append([pair[0], pair[2]])
                print(f"Doc #{i} coreference result: {result['coreference']}")
                
            elif relation == "temporal":
                for mention_i in doc['eventnumberid2mention']:
                    for mention_j in doc['eventnumberid2mention']:
                        if mention_i != mention_j:
                            
                            print(mention_i, mention_j)
                            if len(example_doc["temporal_relations"]["BEFORE"]) == 0:
                                example_mention_i = example_doc['event_mentions'][0]['id_in_number']
                                example_mention_j = example_doc['event_mentions'][1]['id_in_number']
                                prompt_answer = "NONE"
                            else:
                                example_mention_i = example_doc["temporal_relations"]["BEFORE"][0][0]
                                example_mention_j = example_doc["temporal_relations"]["BEFORE"][0][1]
                                prompt_answer = "BEFORE"

                            prompt_context = example_doc["context"]
                            prompt_context += f" What is the temporal relation between Event_1 and Event_4?"
                            input_context = doc["context"]
                            input_context += f" What is the temporal relation between {mention_i} and {mention_j}?"

                            dialogs: List[Dialog] = [[{"role": "system", "content": "Predict the temporal relation given two events, answer with one word from [BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS, NONE]"},
                                {"role": "user", "content": f"{prompt_context}."},
                                {"role": "assistant", "content": f"NONE"},
                                {"role": "user", "content": f"What is the temporal relation between Event_7 and Event_0?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between Event_19 and Event_10?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between TIMEX_0 and TIMEX_1?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between TIMEX_0 and Event_3?"},
                                {"role": "assistant", "content": f"CONTAINS"},
                                {"role": "user", "content": f"What is the temporal relation between Event_1 and TIMEX_1?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between Event_5 and Event_1?"},
                                {"role": "assistant", "content": f"OVERLAP"},
                                {"role": "user", "content": f"What is the temporal relation between Event_3 and Event_4?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between Event_5 and Event_10?"},
                                {"role": "assistant", "content": f"ENDS-ON"},
                                {"role": "user", "content": f"What is the temporal relation between Event_20 and Event_0?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between Event_0 and Event_1?"},
                                {"role": "assistant", "content": f"SIMULTANEOUS"},
                                {"role": "user", "content": f"What is the temporal relation between Event_5 and Event_14?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between Event_17 and Event_3?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between Event_10 and TIMEX_1?"},
                                {"role": "assistant", "content": f"BEFORE"},
                                {"role": "user", "content": f"What is the temporal relation between Event_11 and Event_10?"},
                                {"role": "assistant", "content": f"NONE"},
                                {"role": "user", "content": f"{input_context} Do not reply using a complete sentence, and only give the answer from: [BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS, NONE]."}]]

                            results = generator.chat_completion(
                                dialogs,  # type: ignore
                                max_gen_len=max_gen_len,
                                temperature=temperature,
                                top_p=top_p
                            )
                            pred = results[0]['generation']['content'].strip()

                            print(pred)

                            if pred in result["temporal_relations"]:
                                result["temporal_relations"][pred].append([mention_i, mention_j])
                print(f"Doc #{i} temporal result: {result['temporal_relations']}")
                
            elif relation == "causal":
                for mention_i in doc['event_mentions']:
                    mention_i = mention_i['id_in_number']
                    for mention_j in doc['event_mentions']:
                        mention_j = mention_j['id_in_number']
                        if mention_i != mention_j:
                            
                            print(mention_i, mention_j)

                            prompt_context = example_doc["context"]
                            prompt_context += f" What is the relation between Event_2 and Event_5?"
                            input_context = doc["context"]
                            input_context += f" What is the relation between {mention_i} and {mention_j}?"

                            dialogs: List[Dialog] = [[{"role": "system", "content": "Predict the relation given two events, answer with one word from [CAUSE, PRECONDITION, NONE]"},
                                {"role": "user", "content": f"{prompt_context}"},
                                {"role": "assistant", "content": f"NONE"},
                                {"role": "user", "content": f"What is the relation between Event_5 and Event_4?"},
                                {"role": "assistant", "content": f"NONE"},
                                {"role": "user", "content": f"What is the relation between Event_1 and Event_2?"},
                                {"role": "assistant", "content": f"NONE"},
                                {"role": "user", "content": f"What is the relation between Event_21 and Event_0?"},
                                {"role": "assistant", "content": f"PRECONDITION"},
                                {"role": "user", "content": f"What is the relation between Event_21 and Event_2?"},
                                {"role": "assistant", "content": f"PRECONDITION"},
                                {"role": "user", "content": f"What is the relation between Event_3 and Event_4?"},
                                {"role": "assistant", "content": f"PRECONDITION"},
                                {"role": "user", "content": f"What is the relation between Event_11 and Event_12?"},
                                {"role": "assistant", "content": f"PRECONDITION"},
                                {"role": "user", "content": f"What is the relation between Event_17 and Event_15?"},
                                {"role": "assistant", "content": f"PRECONDITION"},
                                {"role": "user", "content": f"What is the relation between Event_0 and Event_25?"},
                                {"role": "assistant", "content": f"PRECONDITION"},
                                {"role": "user", "content": f"What is the relation between Event_21 and Event_27?"},
                                {"role": "assistant", "content": f"PRECONDITION"},
                                {"role": "user", "content": f"{input_context} Do not reply using a complete sentence, and only give the answer from: [PRECONDITION, CAUSE, NONE]."}]]

                            results = generator.chat_completion(
                                dialogs,  # type: ignore
                                max_gen_len=max_gen_len,
                                temperature=temperature,
                                top_p=top_p,
                            )
                            pred = results[0]['generation']['content'].strip()

                            print(pred)

                            if pred in result["causal_relations"]:
                                result["causal_relations"][pred].append([mention_i, mention_j])
                print(f"Doc #{i} causal result: {result['causal_relations']}")
                
            elif relation == "subevent":
                for mention_i in doc['event_mentions']:
                    mention_i = mention_i['id_in_number']
                    for mention_j in doc['event_mentions']:
                        mention_j = mention_j['id_in_number']
                        if mention_i != mention_j:
                            
                            print(mention_i, mention_j)

                            prompt_context = example_doc["context"]
                            prompt_context += f" Is Event_0 a subevent of Event_4?"
                            input_context = doc["context"]
                            input_context += f" Is {mention_i} a subevent of {mention_j}?"


                            dialogs: List[Dialog] = [[{"role": "system", "content": "Predict the subevent relation given two events, answer with one word from [Yes, No]"},
                                {"role": "user", "content": f"{prompt_context}"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Is Event_0 a subevent of Event_6?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Is Event_1 a subevent of Event_3?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Is Event_0 a subevent of Event_2?"},
                                {"role": "assistant", "content": f"Yes"},
                                {"role": "user", "content": f"Is Event_1 a subevent of Event_4?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Is Event_18 a subevent of Event_5?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Is Event_7 a subevent of Event_14?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Is Event_3 a subevent of Event_11?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Is Event_6 a subevent of Event_5?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"Is Event_19 a subevent of Event_16?"},
                                {"role": "assistant", "content": f"No"},
                                {"role": "user", "content": f"{input_context} Do not reply using a complete sentence, and only give the answer from: [Yes, No]."}]]

                            results = generator.chat_completion(
                                dialogs,  # type: ignore
                                max_gen_len=max_gen_len,
                                temperature=temperature,
                                top_p=top_p,
                            )
                            pred = results[0]['generation']['content'].strip()

                            print(pred)

                            if pred == "Yes":
                                result["subevent_relations"].append([mention_i, mention_j])
                print(f"Doc #{i} subevent result: {result['subevent_relations']}")
                
        with open(os.path.join(output_dir, "test_prediction.jsonl"), "a")as f:
            f.write(json.dumps(result))
            f.write("\n")
if __name__ == "__main__":
    fire.Fire(main)