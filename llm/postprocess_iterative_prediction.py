import json
import os

def postprocess(raw_prediction):
    with open(os.path.join(output_dir, "eventnumberid2mentionid.jsonl"), "r")as f:
        lines = f.readlines()
        eventnumberid2mentionid = json.loads(lines[0].strip())
    docs = {}
    with open(os.path.join(data_dir, "test.jsonl"), "r")as f:
        lines = f.readlines()
        for line in lines:
            doc = json.loads(line.strip())
            docs[doc["id"]] = doc
    processed_prediction = {"id": None, "causal_relations": {"CAUSE": None, "PRECONDITION": None}, "temporal_relations": {"BEFORE": None, "CONTAINS": None, "OVERLAP": None, "BEGINS-ON": None, "ENDS-ON": None, "SIMULTANEOUS": None}, "subevent_relations": None}
    processed_prediction["id"] = raw_prediction["id"]

    eventnumberid2mentionid = eventnumberid2mentionid[processed_prediction["id"]]

    coref_temp = []
    appeared = {}
    for cluster in raw_prediction["coreference"]:
        cluster = list(set(cluster))
        new_cluster = []
        for i, event in enumerate(cluster):
            if event in appeared:
                continue
            if event not in eventnumberid2mentionid:
                continue
            if event.startswith("TIME"):
                continue
            new_cluster.append(eventnumberid2mentionid[event])
            appeared[event] = eventnumberid2mentionid[event]
        coref_temp.append(new_cluster)

    '''gold_len = 0
    mentions = []
    for event in docs[processed_prediction["id"]]['events']:
        for mention in event['mention']:
            gold_len += 1
            mentions.append(mention["id"])'''

    for mention in eventnumberid2mentionid:
        #if mention in appeared:
        #        continue
        already_in_cluster = False
        for cluster in raw_prediction["coreference"]:
            if mention in cluster or mention.startswith("TIME"):
                already_in_cluster = True
        if not already_in_cluster:
            coref_temp.append([eventnumberid2mentionid[mention]])
            appeared[mention] = eventnumberid2mentionid[mention]
    processed_prediction["coreference"] = coref_temp

    #rint(len(docs[processed_prediction["id"]]["event_mentions"]))
    #print(gold_len)
    #print(mentions)
    #print(len(appeared))
    #print(appeared)
    #assert len(appeared) == len(docs[processed_prediction["id"]]["event_mentions"])
    #assert len(appeared) ==gold_len

    temporal_temp = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
    for temporal_type in raw_prediction["temporal_relations"]:
        pred_array = raw_prediction["temporal_relations"][temporal_type]
        pred_array = list(set(tuple(pair) for pair in pred_array))

        for pair in pred_array:
            new_pair = [eventnumberid2mentionid[pair[0]], eventnumberid2mentionid[pair[1]]]
            if eventnumberid2mentionid[pair[0]] == eventnumberid2mentionid[pair[1]]:
                continue
            temporal_temp[temporal_type].append(new_pair)
    processed_prediction["temporal_relations"] = temporal_temp
    
    causal_temp = {"CAUSE": [], "PRECONDITION": []}
    for causal_type in raw_prediction["causal_relations"]:
        pred_array = raw_prediction["causal_relations"][causal_type]
        pred_array = list(set(tuple(pair) for pair in pred_array))

        for pair in pred_array:
            if pair[0].startswith("TIME"):
                continue
            if pair[1].startswith("TIME"):
                continue
            new_pair = [eventnumberid2mentionid[pair[0]], eventnumberid2mentionid[pair[1]]]
            if eventnumberid2mentionid[pair[0]] == eventnumberid2mentionid[pair[1]]:
                continue
            causal_temp[causal_type].append(new_pair)
    processed_prediction["causal_relations"] = causal_temp
    
    subevent_temp = []
    pred_array = raw_prediction["causal_relations"][causal_type]
    pred_array = list(set(tuple(pair) for pair in pred_array))

    for pair in pred_array:
        if pair[0].startswith("TIME"):
                continue
        if pair[1].startswith("TIME"):
            continue
        new_pair = [eventnumberid2mentionid[pair[0]], eventnumberid2mentionid[pair[1]]]
        if eventnumberid2mentionid[pair[0]] == eventnumberid2mentionid[pair[1]]:
                continue
        subevent_temp.append(new_pair)
    processed_prediction["subevent_relations"] = subevent_temp

    #print(processed_prediction)
    return processed_prediction

if __name__ == "__main__":
    output_dir = "./llm/output/llama2_iterative_prediction/valid"
    data_dir = "./data/MAVEN_ERE"

    processed_predictions = []
    with open(os.path.join(output_dir, "test_prediction.jsonl"), "r")as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(f"Doc #{i}")
            raw_prediction = json.loads(line.strip())
            processed_predictions.append(postprocess(raw_prediction))
    
    if os.path.exists(os.path.join(output_dir, "processed_test_prediction.jsonl")):
        os.remove(os.path.join(output_dir, "processed_test_prediction.jsonl"))

    with open(os.path.join(output_dir, "processed_test_prediction.jsonl"), "w")as f:
        f.writelines("\n".join([json.dumps(processed_prediction) for processed_prediction in processed_predictions]))