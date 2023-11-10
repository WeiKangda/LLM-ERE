import os
import openai
#from .data import *
from tqdm import tqdm
import sys
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

docs = []
output_dir = "./"
data_dir = "../data/MAVEN_ERE"

if os.path.exists(os.path.join(output_dir, "log.txt")):
    os.remove(os.path.join(output_dir, "log.txt"))
sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')

if os.path.exists(os.path.join(output_dir, "test_prediction.jsonl")):
    os.remove(os.path.join(output_dir, "test_prediction.jsonl"))

with open(os.path.join(data_dir, f"test.jsonl"))as f:
    lines = f.readlines()
    for line in tqdm(lines, desc="Loading test data"):
        doc = json.loads(line.strip())
        #doc = Document(data, ignore_nonetype=False)
        docs.append(doc)

example_doc = None
with open(os.path.join(f"./example.jsonl"))as f:
    lines = f.readlines()
    example_doc = json.loads(lines[0].strip())
i = 0
for doc in tqdm(docs, desc="Predicting"):
    i += 1
    result = {"id": doc["id"], "coreference": None, "causal_relations": {}, "temporal_relations": {}, "subevent_relations": None}
    for relation in ["coreference", "temporal", "causal", "subevent"]:
        if relation == "coreference":
            gold_clusters = []
            for event in example_doc["events"]:
                gold_cluster = []
                for mention in event["mention"]:
                    gold_cluster.append(mention["id"])
                gold_clusters.append(gold_cluster)
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given. For temporal relations, there are 7 types: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS, and NONE. For causal relations, there are 3 types: CAUSE, PRECONDITION, and NONE. For subevent relations, there are 2 types: SUBEVENT and NONE. For coreference relations, there are 2 types: COREF and NONE. Note that the order of the events matter. SIMULTANEOUS and BEGINS-ON are bidirectional relations. Hint: the dataset is unbalanced with more BEFORE and CONTAINS relation pairs for temporal relations."},
                {"role": "user", "content": f"This is the content of the document: {example_doc['sentences']}."},
                {"role": "user", "content": f"These are the event mentions of the document: {example_doc['events']}."},
                {"role": "user", "content": f"These are the TIMEX mentions of the document: {example_doc['TIMEX']}."},
                {"role": "user", "content": f"List all the coreference clusters."},
                {"role": "assistant", "content": f"{gold_clusters}"},
                {"role": "user", "content": f"This is the content of the document: {doc['sentences']}."},
                {"role": "user", "content": f"These are the event mentions of the document: {doc['event_mentions']}."},
                {"role": "user", "content": f"These are the TIMEX mentions of the document: {doc['TIMEX']}."},
                {"role": "user", "content": f"List all the coreference clusters."},
            ])
                
            result["coreference"] = completion.choices[0].message
            print(f"coreference predictions: {completion.choices[0].message}")

        elif relation == "temporal":
            for sub_relation in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given. For temporal relations, there are 7 types: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS, and NONE. For causal relations, there are 3 types: CAUSE, PRECONDITION, and NONE. For subevent relations, there are 2 types: SUBEVENT and NONE. For coreference relations, there are 2 types: COREF and NONE. Note that the order of the events matter. SIMULTANEOUS and BEGINS-ON are bidirectional relations. Hint: the dataset is unbalanced with more BEFORE and CONTAINS relation pairs for temporal relations."},
                    {"role": "user", "content": f"This is the content of the document: {example_doc['sentences']}."},
                    {"role": "user", "content": f"These are the event mentions of the document: {example_doc['events']}."},
                    {"role": "user", "content": f"These are the TIMEX mentions of the document: {example_doc['TIMEX']}."},
                    {"role": "user", "content": f"List all the mention pairs that are {sub_relation} in {relation} relations."},
                    {"role": "assistant", "content": f"{example_doc['temporal_relations'][sub_relation]}"},
                    {"role": "user", "content": f"This is the content of the document: {doc['sentences']}."},
                    {"role": "user", "content": f"These are the event mentions of the document: {doc['event_mentions']}."},
                    {"role": "user", "content": f"These are the TIMEX mentions of the document: {doc['TIMEX']}."},
                    {"role": "user", "content": f"List all the mention pairs that are {sub_relation} in {relation} relations."},
                ])
                
                result["temporal_relations"][sub_relation] = completion.choices[0].message
                print(f"temporal predictions: {completion.choices[0].message}")

        elif relation == "causal":
            for sub_relation in ["CAUSE", "PRECONDITION"]:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given. For temporal relations, there are 7 types: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS, and NONE. For causal relations, there are 3 types: CAUSE, PRECONDITION, and NONE. For subevent relations, there are 2 types: SUBEVENT and NONE. For coreference relations, there are 2 types: COREF and NONE. Note that the order of the events matter. SIMULTANEOUS and BEGINS-ON are bidirectional relations. Hint: the dataset is unbalanced with more BEFORE and CONTAINS relation pairs for temporal relations."},
                    {"role": "user", "content": f"This is the content of the document: {example_doc['sentences']}."},
                    {"role": "user", "content": f"These are the event mentions of the document: {example_doc['events']}."},
                    {"role": "user", "content": f"These are the TIMEX mentions of the document: {example_doc['TIMEX']}."},
                    {"role": "user", "content": f"List all the mention pairs that are {sub_relation} in {relation} relations."},
                    {"role": "assistant", "content": f"{example_doc['causal_relations'][sub_relation]}"},
                    {"role": "user", "content": f"This is the content of the document: {doc['sentences']}."},
                    {"role": "user", "content": f"These are the event mentions of the document: {doc['event_mentions']}."},
                    {"role": "user", "content": f"These are the TIMEX mentions of the document: {doc['TIMEX']}."},
                    {"role": "user", "content": f"List all the mention pairs that are {sub_relation} in {relation} relations."},
                ])
                
                result["causal_relations"][sub_relation] = completion.choices[0].message
                print(f"causal predictions: {completion.choices[0].message}")

        elif relation == "subevent":
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given. For temporal relations, there are 7 types: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS, and NONE. For causal relations, there are 3 types: CAUSE, PRECONDITION, and NONE. For subevent relations, there are 2 types: SUBEVENT and NONE. For coreference relations, there are 2 types: COREF and NONE. Note that the order of the events matter. SIMULTANEOUS and BEGINS-ON are bidirectional relations. Hint: the dataset is unbalanced with more BEFORE and CONTAINS relation pairs for temporal relations."},
                {"role": "user", "content": f"This is the content of the document: {example_doc['sentences']}."},
                {"role": "user", "content": f"These are the event mentions of the document: {example_doc['events']}."},
                {"role": "user", "content": f"These are the TIMEX mentions of the document: {example_doc['TIMEX']}."},
                {"role": "user", "content": f"List all the mention pairs that are SUBEVENT in {relation} relations."},
                {"role": "assistant", "content": f"{example_doc['subevent_relations']}"},
                {"role": "user", "content": f"This is the content of the document: {doc['sentences']}."},
                {"role": "user", "content": f"These are the event mentions of the document: {doc['event_mentions']}."},
                {"role": "user", "content": f"These are the TIMEX mentions of the document: {doc['TIMEX']}."},
                {"role": "user", "content": f"List all the mention pairs that are SUBEVENT in {relation} relations."},
            ])
                
            result["subevent_relations"] = completion.choices[0].message
            print(f"subevent predictions: {completion.choices[0].message}")

    with open(os.path.join(output_dir, "test_prediction.jsonl"), "w")as f:
        f.write(result)
        f.write("\n")
    if i == 2:
        exit()