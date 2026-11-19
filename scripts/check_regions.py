import sys, collections, yaml

yaml_path = sys.argv[1]
counts_path = sys.argv[2] if len(sys.argv) > 2 else 'data/interim/p27_values_attrs.txt'

doc = yaml.safe_load(open(yaml_path))
print('top-level keys:', list(doc))
for k in ['scheme', 'regions', 'source_audit', 'decision_counts', 'region_counts']:
    print('---', k)
    print(doc.get(k))
print('---')

cfg = doc['entities']
print('countries in YAML:', len(cfg))
print('statuses:', collections.Counter(v.get('status') for v in cfg.values()))
print('region codes:', collections.Counter(str(v.get('region_code')) for v in cfg.values()))
print('non-string region_code:', [q for q, v in cfg.items() if not isinstance(v.get('region_code'), str)][:20])

counts = {}
for line in open(counts_path):
    parts = line.split()
    if len(parts) == 2:
        counts[parts[1]] = int(parts[0])
total = sum(counts.values())
unmapped = {q: c for q, c in counts.items() if q not in cfg}
print('P27 values in data:', len(counts), '| rows:', total)
print('unmapped values:', len(unmapped), '| rows:', sum(unmapped.values()),
      '| share: %.4f' % (sum(unmapped.values()) / total))
print('top unmapped:', sorted(unmapped.items(), key=lambda x: -x[1])[:10])
mism = [(q, counts.get(q, 0), v.get('p27_occurrences')) for q, v in cfg.items()
        if counts.get(q, 0) != v.get('p27_occurrences')]
print('occurrence mismatches:', len(mism), mism[:10])
print('YAML entries absent from data:', sum(1 for q in cfg if q not in counts))
for q in ['Q30', 'Q145', 'Q16', 'Q142', 'Q174193']:
    print(q, cfg.get(q))
