import json


def remove_types(json_obj):
    json_dict = {}
    if type(json_obj) == dict:
        for k, v in json_obj.items():
            if str.startswith(k, "$"):
                if "Int" in k or "Long" in k:
                    return int(v)
                elif "Double" in k:
                    return float(v)
                else:
                    return v
            if k == "_id":
                continue
            json_dict[k] = remove_types(v)
        return json_dict
    elif type(json_obj) == list:
        return [remove_types(k) for k in json_obj]
    else:
        return json_obj


with open("./black_hist_bson.json", "r", encoding="utf-8") as infile:
    with open("./json_output.json", "w", encoding="utf-8") as outfile:
        for line in infile:
            tweet = remove_types(json.loads(line))
            outfile.write(json.dumps(tweet)+"\n")

