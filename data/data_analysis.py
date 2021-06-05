from tqdm import tqdm

from construct_four_way_parallel_corpus import get_unique_en_doc_ids, corpus_path

language_pairs = ['en-ja', 'en-fr', 'en-es']

train_doc_count = get_unique_en_doc_ids('en-ja', 'four_way_parallel_corpus/train')
print('Train data statistics:')
print(f'Documents: {len(train_doc_count)}')
for pair in language_pairs:
    with open(corpus_path(pair, 'en', 'four_way_parallel_corpus/train'), 'r') as f:
        print(f'{pair} has {len(f.readlines())} sentences.')

print('')
print('')

eval_doc_count = get_unique_en_doc_ids('en-ja', 'four_way_parallel_corpus/eval')
print('Eval data statistics:')
print(f'Documents: {len(eval_doc_count)}')
for pair in language_pairs:
    with open(corpus_path(pair, 'en', 'four_way_parallel_corpus/eval'), 'r') as f:
        print(f'{pair} has {len(f.readlines())} sentences.')

print('')
print('')
with open(corpus_path('en-ja', 'en', 'four_way_parallel_corpus/train'), 'r') as en_ja_f, \
     open(corpus_path('en-fr', 'en', 'four_way_parallel_corpus/train'), 'r') as en_fr_f, \
     open(corpus_path('en-es', 'en', 'four_way_parallel_corpus/train'), 'r') as en_es_f: \
    
    en_ja_lines = en_ja_f.readlines()
    en_fr_lines = en_fr_f.readlines()
    en_es_lines = en_es_f.readlines()

    total = 0
    shared = 0
    for en_line in tqdm(en_es_lines):
        total += 1
        if en_line in en_ja_lines and en_line in en_fr_lines:
            shared += 1

print(f'Of the {total} train lines in en-es, {shared} also occur in en-fr and en-ja')

