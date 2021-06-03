import os
import random
import pandas as pd

# helper function to construct path to corpus files
def corpus_path(corpus, language_pair, split, file_type):
    return f'../data/{corpus}/{split}/{language_pair}/OpenSubtitles.{language_pair}.{file_type}'

def preprocess(corpus, language_pair, split, context_type, break_token):
    lang_1, lang_2 = language_pair.split('-')
    if context_type == '2-to-2':
        with open(corpus_path(corpus, language_pair, split, lang_1)) as lang_1_corpus, \
             open(corpus_path(corpus, language_pair, split, lang_2)) as lang_2_corpus, \
             open(corpus_path(corpus, language_pair, split, 'ids')) as doc_ids:
            data = []
            prev_lang_1_line = ''
            prev_lang_2_line = ''
            prev_doc_id = None
            for raw_lang_1_line, raw_lang_2_line, doc_id_line in zip(lang_1_corpus, lang_2_corpus, doc_ids):
                lang_1_line = raw_lang_1_line.strip()
                lang_2_line = raw_lang_2_line.strip()
                doc_id = doc_id_line.split()[0]

                # clear out prior context if moving to a new document
                if doc_id != prev_doc_id:
                    prev_lang_1_line = ''
                    prev_lang_2_line = ''

                lang_1_example = f'{prev_lang_1_line} {break_token} {lang_1_line}'
                lang_2_example = f'{prev_lang_2_line} {break_token} {lang_2_line}'

                data.append(["", lang_1_example, lang_2_example])

                prev_lang_1_line = lang_1_line
                prev_lang_2_line = lang_2_line
                prev_doc_id = doc_id

        return pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])
    if context_type == 'no-context-with-break':
        with open(corpus_path(corpus, language_pair, split, lang_1)) as lang_1_corpus, \
             open(corpus_path(corpus, language_pair, split, lang_2)) as lang_2_corpus:
            data = []
            for raw_lang_1_line, raw_lang_2_line in zip(lang_1_corpus, lang_2_corpus):
                lang_1_line = raw_lang_1_line.strip()
                lang_2_line = raw_lang_2_line.strip()

                lang_1_example = f' {break_token} {lang_1_line}'
                lang_2_example = f' {break_token} {lang_2_line}'

                data.append(["", lang_1_example, lang_2_example])
        
        return pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])
    
    if context_type == 'no-context-without-break':
        with open(corpus_path(corpus, language_pair, split, lang_1)) as lang_1_corpus, \
             open(corpus_path(corpus, language_pair, split, lang_2)) as lang_2_corpus:
            data = []
            for raw_lang_1_line, raw_lang_2_line in zip(lang_1_corpus, lang_2_corpus):
                lang_1_line = raw_lang_1_line.strip()
                lang_2_line = raw_lang_2_line.strip()

                lang_1_example = f'{lang_1_line}'
                lang_2_example = f'{lang_2_line}'

                data.append(["", lang_1_example, lang_2_example])

        return pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])

    if context_type == 'random-context':
        with open(corpus_path(corpus, language_pair, split, lang_1)) as lang_1_corpus, \
             open(corpus_path(corpus, language_pair, split, lang_2)) as lang_2_corpus:
            lang_1_corpus_lines = [
                raw_lang_1_line.strip() 
                for raw_lang_1_line in lang_1_corpus.readlines()
            ]
            lang_2_corpus_lines = [
                raw_lang_2_line.strip() 
                for raw_lang_2_line in lang_2_corpus.readlines()
            ]

            data = []
            for lang_1_line, lang_2_line in zip(lang_1_corpus_lines, lang_2_corpus_lines):
                # select random context
                random_index = random.randint(0, len(lang_1_corpus_lines) - 1)
                random_lang_1_line = lang_1_corpus_lines[random_index]
                random_lang_2_line = lang_2_corpus_lines[random_index]

                lang_1_example = f'{random_lang_1_line} {break_token} {lang_1_line}'
                lang_2_example = f'{random_lang_2_line} {break_token} {lang_2_line}'

                data.append(["", lang_1_example, lang_2_example])
        
        return pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])

    raise NotImplementedError