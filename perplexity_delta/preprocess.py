# helper function to construct path to corpus files
def corpus_path(corpus, language_pair, split, file_type):
    return f'../data/{corpus}/{split}/{language_pair}/OpenSubtitles.{language_pair}.{file_type}'

def preprocess(corpus_name, language, split):
    language_pair = f'en-{language}' if language != 'en' else 'en-ja'

    with open(corpus_path(corpus_name, language_pair, split, language)) as corpus, \
         open(corpus_path(corpus_name, language_pair, split, 'ids')) as doc_ids:
        data = []
        prev_line = ''
        prev_doc_id = None
        for raw_line, doc_id_line in zip(corpus, doc_ids):
            line = raw_line.strip()
            doc_id = doc_id_line.split()[0]

            # clear out prior context if moving to a new document
            if doc_id != prev_doc_id:
                prev_line = ''

            data.append([prev_line, line])

            prev_line = line
            prev_doc_id = doc_id

    return data
