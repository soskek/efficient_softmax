import argparse
import collections
import io
import json
import sys

# python construct_vocab.py --data datasets/wikitext-103/wiki.train.tokens -t 50 -s datasets/wikitext-103/vocab.t50.json # NOQA

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', required=True)
parser.add_argument('--threshold', '-t', type=int, default=5)
parser.add_argument('--save', '-s', default='vocab.json')
parser.add_argument('--merge', '-m')
args = parser.parse_args()

if args.merge:
    #merged_vocab = json.load(open(args.merge))
    merged_vocab = set(json.load(open(args.merge)).keys())
    """
    for l in io.open(args.merge, encoding='utf-8', errors='ignore'):
        sp = l.rstrip().split('\t')
        if len(sp) == 2:
            merged_vocab.add(sp[0])
    """
    print('vocab to be merged', len(merged_vocab))

count = collections.defaultdict(int)
with io.open(args.data, encoding='utf-8') as f:
    for line in f:
        words = line.split() + ['<eos>']
        for word in words:
            count[word] += 1

vocab = {'<eos>': 0, '<unk>': 1}
for w, c in sorted(count.items(), key=lambda x: (-x[1], x[0])):
    if c < args.threshold:
        continue
    if w not in vocab:
        vocab[w] = len(vocab)

print('# of words: {}'.format(len(vocab)))

if args.merge:
    for w in merged_vocab:
        if w not in vocab:
            if w in count:
                vocab[w] = len(vocab)

print('# of words: {} after merge'.format(len(vocab)))
if args.merge:
    print('# of refected words: {}'.format(
        len(merged_vocab - set(vocab.keys()))))
json.dump(dict(vocab), open(args.save, 'w'))
json.dump(dict(count), open(args.save + '.count', 'w'))
