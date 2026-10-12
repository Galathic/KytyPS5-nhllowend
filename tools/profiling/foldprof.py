"""Summarize sampler.exe output ("count a;b;...;leaf" lines).
foldprof.py <file> [--within SUBSTR] [--callees SUBSTR] [--top N] [--width W]
  inclusive and self percentages per function; --within keeps samples whose stack contains SUBSTR
  (percentages are then of all samples); --callees lists what the first frame matching SUBSTR calls."""
import re, sys
from collections import Counter
args = sys.argv[1:]
path = args[0]
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
within = opt('--within'); callees = opt('--callees')
top = int(opt('--top', '40')); width = int(opt('--width', '110'))
def clean(name):
    name = re.sub(r'\b(class|struct|enum) ', '', name)
    name = name.replace('Libs::Graphics::', '').replace('Libs::', '').replace('ShaderRecompiler::IR::', 'IR::')
    prev = None
    while prev != name:
        prev = name; name = re.sub(r'<[^<>]*>', '', name)
    return name[:width]
total = 0; incl = Counter(); leaf = Counter(); child = Counter(); kept = 0
for line in open(path, encoding='utf-8', errors='replace'):
    if line.startswith('#'): continue
    count, _, stack = line.rstrip('\n').partition(' ')
    count = int(count); total += count
    frames = [clean(f) for f in stack.split(';')]
    if within and not any(within in f for f in frames): continue
    kept += count
    for f in set(frames): incl[f] += count
    leaf[frames[-1]] += count
    if callees:
        for i, f in enumerate(frames):
            if callees in f:
                child[frames[i + 1] if i + 1 < len(frames) else '<self>'] += count
                break
pct = lambda v: 100.0 * v / max(total, 1)
print(f'samples {total} kept {kept} ({pct(kept):.1f}%)')
if callees:
    print(f'-- callees of {callees}')
    for k, v in child.most_common(top): print(f'{v:6d} {pct(v):5.1f}% {k}')
    sys.exit()
print('-- inclusive')
for k, v in incl.most_common(top): print(f'{v:6d} {pct(v):5.1f}% {k}')
print('-- self')
for k, v in leaf.most_common(top): print(f'{v:6d} {pct(v):5.1f}% {k}')
