import json
from collections import defaultdict

def dump_grammar():
    mapping = defaultdict(set)
    
    with open('workspace/oare_epigraphies.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except:
                continue
            units = data.get('units', [])
            for u in units:
                pi = u.get('parseInfo')
                if pi:
                    for item in pi:
                        k = item.get('variableName')
                        v = item.get('value')
                        if k and v:
                            kn = k.lower().replace(' ', '_')
                            vn = v.lower()
                            mapping[kn].add(vn)
                            
    res = {k: list(v) for k, v in mapping.items()}
    with open('workspace/outputs/published_texts/grammar_keys_dump.json', 'w', encoding='utf-8') as out:
        json.dump(res, out, indent=2, ensure_ascii=False)
    print("Dumped unique grammar keys and values to workspace/outputs/published_texts/grammar_keys_dump.json")

if __name__ == '__main__':
    dump_grammar()
