import json
import os

def postprocess(raw_prediction):
    with open(os.path.join(output_dir, "eventnumberid2mentionid.jsonl"), "r")as f:
        lines = f.readlines()
        eventnumberid2mentionid = json.loads(lines[0].strip())
    processed_prediction = {"id": None, "coreference": None, "causal_relations": {"CAUSE": None, "PRECONDITION": None}, "temporal_relations": {"BEFORE": None, "CONTAINS": None, "OVERLAP": None, "BEGINS-ON": None, "ENDS-ON": None, "SIMULTANEOUS": None}, "subevent_relations": None}
    processed_prediction["id"] = raw_prediction["id"]

    for relation_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
        if relation_type not in raw_prediction["temporal_relations"]:
            processed_prediction["temporal_relations"][relation_type] = []
    
    for relation_type in ["CAUSE", "PRECONDITION"]:
        if relation_type not in raw_prediction["causal_relations"]:
            processed_prediction["causal_relations"][relation_type] = []

    eventnumberid2mentionid = eventnumberid2mentionid[processed_prediction["id"]]

    temporal_temp = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
    for temporal_type in raw_prediction["temporal_relations"]:
        for pair in raw_prediction["temporal_relations"][temporal_type]:
            if pair[0] == pair[1]:
                continue
            new_pair = []
            for event in pair:
                if event not in eventnumberid2mentionid:
                    continue
                new_pair.append(eventnumberid2mentionid[event])
            if len(new_pair) != 2:
                continue
            temporal_temp[temporal_type].append(new_pair)
    processed_prediction["temporal_relations"] = temporal_temp

    causal_temp = {"CAUSE": [], "PRECONDITION": []}
    for causal_type in raw_prediction["causal_relations"]:
        for pair in raw_prediction["causal_relations"][causal_type]:
            if pair[0] == pair[1]:
                continue
            new_pair = []
            for event in pair:
                if event not in eventnumberid2mentionid:
                    continue
                new_pair.append(eventnumberid2mentionid[event])
            if len(new_pair) != 2:
                continue
            causal_temp[causal_type].append(new_pair)
    processed_prediction["causal_relations"] = causal_temp
    
    subevent_temp = []
    for pair in raw_prediction["subevent_relations"]:
        if pair[0] == pair[1]:
            continue
        new_pair = []
        for event in pair:
            if event not in eventnumberid2mentionid:
                continue
            new_pair.append(eventnumberid2mentionid[event])
        if len(new_pair) != 2:
            continue
        subevent_temp.append(new_pair)
    processed_prediction["subevent_relations"] = subevent_temp

    coref_temp = []
    appeared = {}
    for pair in raw_prediction["coreference"]:
        #cluster = list(set(cluster))
        new_cluster = []
        for i, event in enumerate(pair):
            if event in appeared:
                for j, cluster in enumerate(coref_temp):
                    if event in cluster:
                        coref_temp[j].append(eventnumberid2mentionid(pair[0]))
                        coref_temp[j].append(eventnumberid2mentionid(pair[1]))
                        appeared[event] = eventnumberid2mentionid[pair[0]]
                        appeared[event] = eventnumberid2mentionid[pair[1]]
                continue
            if event not in eventnumberid2mentionid:
                continue
            if event.startswith("TIME"):
                continue
            new_cluster.append(eventnumberid2mentionid[event])
            appeared[event] = eventnumberid2mentionid[event]
        coref_temp.append(new_cluster)

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
    for i, cluster in enumerate(coref_temp):
        cluster = list(set(cluster))
        coref_temp[i] = cluster
    processed_prediction["coreference"] = coref_temp
    print(coref_temp)

    return processed_prediction

if __name__ == "__main__":
    output_dir = "./llm/output/valid"

    processed_predictions = []
    with open(os.path.join(output_dir, "test_prediction.jsonl"), "r")as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(f"Doc #{i}")
            #line = json.dumps(line)
            #print(type(line))
            line = line.replace("\"[", "[")
            line = line.replace("]\"", "]")
            line = line.replace("\"{", "{")
            line = line.replace("}\"", "}")
            line = line.replace("\'", "\"")
            #print(line)
            raw_prediction = json.loads(line.strip())
            #print(type(line))
            processed_predictions.append(postprocess(raw_prediction))
    
    if os.path.exists(os.path.join(output_dir, "processed_test_prediction.jsonl")):
        os.remove(os.path.join(output_dir, "processed_test_prediction.jsonl"))

    with open(os.path.join(output_dir, "processed_test_prediction.jsonl"), "w")as f:
        f.writelines("\n".join([json.dumps(processed_prediction) for processed_prediction in processed_predictions]))