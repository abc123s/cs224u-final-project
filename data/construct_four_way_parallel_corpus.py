import os
import json
from sklearn.model_selection import train_test_split

# helper function to construct path to corpus files
def corpus_path(language_pair, file_type, corpus_path = 'raw'):
    return f'./{corpus_path}/{language_pair}/OpenSubtitles.{language_pair}.{file_type}'

# helper function to get the english document id
# given a en-[x] corpus id file
def get_en_doc_id(line):
    lang_1_doc_id, lang_2_doc_id, *_ = line.split()
    en_doc_id = lang_1_doc_id if lang_1_doc_id[:2] == 'en' else lang_2_doc_id

    return en_doc_id

# function to get unique english document ids
# for a particular language pair corpus
def get_unique_en_doc_ids(language_pair, corpus = 'raw'):
    with open(corpus_path(language_pair, 'ids', corpus), 'r') as id_file:
        unique_en_doc_ids = set()

        for line in id_file:
            unique_en_doc_ids.add(get_en_doc_id(line))

    print(f'{language_pair} corpus has {len(unique_en_doc_ids)} unique documents.')

    return unique_en_doc_ids

# function to identify document ids that are
# shared across multiple language pair corpuses;
# used to construct a multi-language parallel corpus
def identify_shared_doc_ids(language_pairs):
    language_pair_doc_ids = [
        get_unique_en_doc_ids(language_pair) 
        for language_pair in language_pairs
    ]

    shared_doc_ids = language_pair_doc_ids[0].intersection(*language_pair_doc_ids[1:])

    print(f'{language_pairs} corpuses have {len(shared_doc_ids)} shared documents.')

    return list(shared_doc_ids)

# function to construct a subset of a parallel corpus, containing
# only the specified document ids
def construct_corpus_subset(language_pair, doc_ids, subset_path):
    lang_1, lang_2 = language_pair.split('-')

    if not os.path.isdir(f'./{subset_path}'):
        os.mkdir(f'./{subset_path}')

    if not os.path.isdir(f'./{subset_path}/{language_pair}'):
        os.mkdir(f'./{subset_path}/{language_pair}')

    with open(corpus_path(language_pair, 'ids'), 'r') as orig_id_file, \
         open(corpus_path(language_pair, lang_1), 'r') as orig_lang_1_file, \
         open(corpus_path(language_pair, lang_2), 'r') as orig_lang_2_file, \
         open(corpus_path(language_pair, 'ids', subset_path), 'w') as subset_id_file, \
         open(corpus_path(language_pair, lang_1, subset_path), 'w') as subset_lang_1_file, \
         open(corpus_path(language_pair, lang_2, subset_path), 'w') as subset_lang_2_file:

        for id_line, lang_1_line, lang_2_line in zip(orig_id_file, orig_lang_1_file, orig_lang_2_file):
            en_doc_id = get_en_doc_id(id_line)

            if en_doc_id in doc_ids:
                subset_id_file.write(id_line)
                subset_lang_1_file.write(lang_1_line)
                subset_lang_2_file.write(lang_2_line)

# construct a parallel corpus that includes identical content for all
# language_pairs, one for each language pair (since the same document
# may be split slightly differently for each pair), and split into
# train and eval sets
def construct_multi_language_parallel_corpus(language_pairs, corpus_name):
    if not os.path.isfile(f'./{corpus_name}/train_doc_ids.json') \
       or not os.path.isfile(f'./{corpus_name}/eval_doc_ids.json'):
        print('No train / eval doc id split provided, identifying docs shared by all language pairs and making new split.')
        
        # select only documents that exist in all language pairs
        corpus_doc_ids = identify_shared_doc_ids(language_pairs)

        # split docs into train and eval sets, write them down
        train_corpus_doc_ids, eval_corpus_doc_ids = train_test_split(corpus_doc_ids, test_size = 0.2)

        if not os.path.isdir(f'./{corpus_name}'):
            os.mkdir(f'./{corpus_name}')

        with open(f'./{corpus_name}/train_doc_ids.json', 'w') as train_id_file:
            json.dump(train_corpus_doc_ids, train_id_file)

        with open(f'./{corpus_name}/eval_doc_ids.json', 'w') as eval_id_file:
            json.dump(eval_corpus_doc_ids, eval_id_file)
    else:
        print('Using existing train / eval doc id split (train_doc_ids.json, eval_doc_ids.json)')
        
        with open(f'./{corpus_name}/train_doc_ids.json', 'r') as train_id_file:
            train_corpus_doc_ids = json.load(train_id_file)
        with open(f'./{corpus_name}/eval_doc_ids.json', 'r') as eval_id_file:
            eval_corpus_doc_ids = json.load(eval_id_file)

    # construct actual train and eval corpuses for each language pair
    for language_pair in language_pairs:
        print(f'Constructing train corpus for {language_pair}...')
        construct_corpus_subset(language_pair, train_corpus_doc_ids, f'{corpus_name}/train')
        print(f'Constructing eval corpus for {language_pair}...')
        construct_corpus_subset(language_pair, eval_corpus_doc_ids, f'{corpus_name}/eval')


if __name__ == "__main__":
    construct_multi_language_parallel_corpus(['en-ja', 'en-es', 'en-fr'], 'four_way_parallel_corpus')
