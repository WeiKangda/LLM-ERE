import os
import openai
#from .data import *
from tqdm import tqdm
import sys
import json
from collections import defaultdict
import re
import argparse
import random
import time
import copy
from .utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    docs = []
    if args.test: 
        output_dir = "./llm/output/gpt4_pair_wise/test"
    else:
        output_dir = "./llm/output/gpt4_pair_wise/valid"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = "./data/MAVEN_ERE"

    if os.path.exists(os.path.join(output_dir, "log.txt")):
        os.remove(os.path.join(output_dir, "log.txt"))
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')

    resume_doc_number = 0
    if args.resume_eval:
        with open(os.path.join(output_dir, f"test_prediction.jsonl"), "r")as f:
            lines = f.readlines()
            for line in lines:
                resume_doc_number += 1

    if not args.test:
        docs = train_or_valid_doc_process(data_dir, args.resume_eval , resume_doc_number)
    else:
        docs = test_doc_process(data_dir, args.resume_eval , resume_doc_number)
    
    example_doc = example_doc_process()

    for i, doc in enumerate(tqdm(docs, desc="Predicting")):
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
        for relation in ["coreference", "temporal", "causal", "subevent"]:
            if relation == "coreference":
                cluster_appeared = []
                for mention_i in doc['event_mentions']:
                    for mention_j in doc['event_mentions']:
                        mention_i = mention_i['id_in_number']
                        mention_j = mention_j['id_in_number']
                        if mention_i != mention_j:
                            
                            print(mention_i, mention_j)

                            prompt_context = example_doc["context"]
                            have_cluster = False
                            for cluster in example_doc['coreference']:
                                if len(cluster) > 1:
                                    have_cluster = True
                                    example_cluster = cluster
                            if not have_cluster:
                                example_mention_i = example_doc['event_mentions'][0]['id_in_number']
                                example_mention_j = example_doc['event_mentions'][1]['id_in_number']
                                prompt_answer = "NONE"
                            else:
                                example_mention_i = example_cluster[0]
                                example_mention_j = example_cluster[1]
                                prompt_answer = "COREFERENCE"
                            prompt_context += f" What is the coreference relation between {example_mention_i} and {example_mention_j}?"
                            input_context = doc["context"]
                            input_context += f" What is the coreference relation between {mention_i} and {mention_j}?"

                            print(prompt_context)
                            print(prompt_answer)
                            print(input_context)

                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "Predict the coreference relation given two events, answer with one word from [COREFERENCE, NONE]"},
                                        {"role": "user", "content": f"{prompt_context}."},
                                        {"role": "assistant", "content": f"{prompt_answer}"},
                                        {"role": "user", "content": f"{input_context}"}
                                    ])
                                    
                                    pred = completion.choices[0].message["content"]
                                    print(pred)
                                    if pred == "COREFERENCE":
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
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue                            
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
                            prompt_context += f" What is the temporal relation between {example_mention_i} and {example_mention_j}?"
                            input_context = doc["context"]
                            input_context += f" What is the temporal relation between {mention_i} and {mention_j}?"

                            print(prompt_context)
                            print(prompt_answer)
                            print(input_context)

                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "Predict the temporal relation given two events, answer with one word from [BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS, NONE]"},
                                        {"role": "user", "content": f"{prompt_context}."},
                                        {"role": "assistant", "content": f"{prompt_answer}"},
                                        {"role": "user", "content": f"{input_context}"}
                                    ])
                                    
                                    pred = completion.choices[0].message["content"]
                                    print(pred)
                                    if pred in result["temporal_relations"]:
                                        result["temporal_relations"][pred].append([mention_i, mention_j])
                                    break
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue
                print(f"Doc #{i} temporal result: {result['temporal_relations']}")
                
            elif relation == "causal":
                for mention_i in doc['event_mentions']:
                    for mention_j in doc['event_mentions']:
                        mention_i = mention_i['id_in_number']
                        mention_j = mention_j['id_in_number']
                        if mention_i != mention_j:
                            
                            print(mention_i, mention_j)

                            prompt_context = example_doc["context"]
                            if len(example_doc["causal_relations"]["CAUSE"]) == 0:
                                example_mention_i = example_doc['event_mentions'][0]['id_in_number']
                                example_mention_j = example_doc['event_mentions'][1]['id_in_number']
                                prompt_answer = "NONE"
                            else:
                                example_mention_i = example_doc["causal_relations"]["CAUSE"][0][0]
                                example_mention_j = example_doc["causal_relations"]["CAUSE"][0][1]
                                prompt_answer = "CAUSE"
                            prompt_context += f" What is the causal relation between {example_mention_i} and {example_mention_j}?"
                            input_context = doc["context"]
                            input_context += f" What is the causal relation between {mention_i} and {mention_j}?"

                            print(prompt_context)
                            print(prompt_answer)
                            print(input_context)
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "Predict the causal relation given two events, answer with one word from [CAUSE, PRECONDITION, NONE]"},
                                        {"role": "user", "content": f"{prompt_context}."},
                                        {"role": "assistant", "content": f"{prompt_answer}"},
                                        {"role": "user", "content": f"{input_context}"}
                                ])
                                    
                                    pred = completion.choices[0].message["content"]
                                    print(pred)
                                    if pred in result["causal_relations"]:
                                        result["causal_relations"][pred].append([mention_i, mention_j])
                                    break
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue
                            
                print(f"Doc #{i} causal result: {result['causal_relations']}")
                
            elif relation == "subevent":
                for mention_i in doc['event_mentions']:
                    for mention_j in doc['event_mentions']:
                        mention_i = mention_i['id_in_number']
                        mention_j = mention_j['id_in_number']
                        if mention_i != mention_j:
                            
                            print(mention_i, mention_j)

                            prompt_context = example_doc["context"]
                            if len(example_doc["subevent_relations"]["SUBEVENT"]) == 0:
                                example_mention_i = example_doc['event_mentions'][0]['id_in_number']
                                example_mention_j = example_doc['event_mentions'][1]['id_in_number']
                                prompt_answer = "NONE"
                            else:
                                example_mention_i = example_doc["subevent_relations"]["SUBEVENT"][0][0]
                                example_mention_j = example_doc["subevent_relations"]["SUBEVENT"][0][1]
                                prompt_answer = "SUBEVENT"
                            prompt_context += f" What is the subevent relation between {example_mention_i} and {example_mention_j}?"
                            input_context = doc["context"]
                            input_context += f" What is the subevent relation between {mention_i} and {mention_j}?"

                            print(prompt_context)
                            print(prompt_answer)
                            print(input_context)

                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "Predict the subevent relation given two events, answer with one word from [SUBEVENT, NONE]"},
                                        {"role": "user", "content": f"{prompt_context}."},
                                        {"role": "assistant", "content": f"{prompt_answer}"},
                                        {"role": "user", "content": f"{input_context}"}
                                ])
                                    
                                    pred = completion.choices[0].message["content"]
                                    print(pred)
                                    if pred in result["subevent_relations"]:
                                        result["subevent_relations"][pred].append([mention_i, mention_j])
                                    break
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue
                print(f"Doc #{i} subevent result: {result['subevent_relations']}")
        
        with open(os.path.join(output_dir, "test_prediction.jsonl"), "a")as f:
            f.write(json.dumps(result))
            f.write("\n")