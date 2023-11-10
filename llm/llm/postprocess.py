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
    for pair in raw_prediction["coreference"]:
        new_pair = []
        for event in pair:
            if event not in eventnumberid2mentionid or "TIMEX" in event:
                    continue
            new_pair.append(eventnumberid2mentionid[event])
        if len(new_pair) != 2:
            continue
        coref_temp.append(new_pair)
    
    processed_prediction["coreference"] = coref_temp

    #print(raw_prediction["coreference"])
    #print(coref_temp)

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
            line = line.replace("\"", "")
            line = line.replace("\'", "\"")
            #print(line)
            raw_prediction = json.loads(line.strip())
            #print(type(line))
            processed_predictions.append(postprocess(raw_prediction))
    
    if os.path.exists(os.path.join(output_dir, "processed_test_prediction.jsonl")):
        os.remove(os.path.join(output_dir, "processed_test_prediction.jsonl"))

    #for key in processed_predictions:
        #print(key)
        #print(processed_predictions[key])
    with open(os.path.join(output_dir, "processed_test_prediction.jsonl"), "w")as f:
        f.writelines("\n".join([json.dumps(processed_prediction) for processed_prediction in processed_predictions]))