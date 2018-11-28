# Lex.py is a utility tool for pre-processing collections and generating TF
# files in SVMlight format -- with support to also dump the vocabulary file.
# It is intended to replace the current scripts relying on Rainbow.

import argparse
import numpy as np
import re
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer

# The following are constants used in the CountVectorizer. Please refer to its
# documentation for the parameter `ngram_range` for more details.
UNIGRAMS=(1, 1)
BIGRAMS=(2, 2)
UNIGRAMS_AND_BIGRAMS=(1, 2)

ENCODING='utf-8'

# The following contains a list of patterns and the string they should be
# replaced with
replace_patterns = [
    ('<[^>]*>', ''),                                    # remove HTML tags
    ('(\D)\d\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D)\d\d\d\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D\D)\d\d\d\D\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D)\d\d\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedZipcodePlusFour \\2'),
    ('(\D)\d(\D)', '\\1ParsedOneDigit\\2'),
    ('(\D)\d\d(\D)', '\\1ParsedTwoDigits\\2'),
    ('(\D)\d\d\d(\D)', '\\1ParsedThreeDigits\\2'),
    ('(\D)\d\d\d\d(\D)', '\\1ParsedFourDigits\\2'),
    ('(\D)\d\d\d\d\d(\D)', '\\1ParsedFiveDigits\\2'),
    ('(\D)\d\d\d\d\d\d(\D)', '\\1ParsedSixDigits\\2'),
    ('\d+', 'ParsedDigits')
]

# Compiles the replace patterns and returns a function that applies them
# (and lower-casing) to text.
def generate_preprocessor(replace_patterns):
    compiled_replace_patterns = [(re.compile(p[0]), p[1])
                                 for p in replace_patterns]
    def preprocessor(text):
        # For each pattern, replace it with the appropriate string
        for pattern, replace in compiled_replace_patterns:
            text = re.sub(pattern, replace, text)
        text = text.lower()
        return text
    return preprocessor

def parse_collection(doc_class_fn, doc_data_fn):
    # Read each line of doc_class_fn and doc_data_fn, yiealding a tuple
    # (class, doc) that corresponds to each line
    with open(doc_class_fn) as doc_class_f:
        with open(doc_data_fn) as doc_data_f:
            doc_class = doc_class_f.readline()
            doc_data = doc_data_f.readline()
            while doc_class != "" and doc_data != "":
                doc_class = doc_class.strip()
                yield (doc_class, doc_data)
                doc_class = doc_class_f.readline()
                doc_data = doc_data_f.readline()

# Returns the fitted count_vectorizer and the transformed collection.
def process_collection(count_vectorizer, collection):
    classes, docs = zip(*collection)
    X = count_vectorizer.fit_transform(docs)
    y = np.array(classes, dtype=np.float32)
    return count_vectorizer, X, y

# Uses analyzer to return the documents after pre-processing. For instance:
# [(0, 'Foo Bar Baz'), (1, 'foo 123 bar')]
#     becomes => ['0 foo bar baz', '1 foo parsedthreedigits bar']
def generate_raw(count_vectorizer, collection):
    analyzer = count_vectorizer.build_analyzer()
    classes, docs = zip(*collection)
    return [' '.join(class_doc)
            for class_doc in zip(classes, [' '.join(analyzer(doc))
                                           for doc in docs])]

# Writes the TF file in SVMlight format
def dump_tf(X, y, out_filename):
    with open(out_filename, 'wb') as out_f:
        dump_svmlight_file(X, y, out_f)

# Writes the vocabulary file
def dump_vocab(count_vectorizer, vocab_filename):
    with open(vocab_filename, 'wt') as vocab_f:
        vocab_f.write(('\n'.join(count_vectorizer.get_feature_names()) + '\n')
                           .encode(ENCODING))

# Writes the raw collection after pre-processing
def dump_raw(raw_collection, raw_filename):
    with open(raw_filename, 'wb') as raw_f:
        raw_f.write(('\n'.join(raw_collection) + '\n').encode(ENCODING))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset parsing and processing utility.')
    parser.add_argument('classes',
        help='File containing document classes. One class per line.')
    parser.add_argument('data',
        help='File containing the collection. One document per line.')
    parser.add_argument('-mindf','--min-df', default=2, type=int,
        help="""When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts. This is passed to the underlying (sklearn) CountVectorizer""")
    parser.add_argument('-r', '--raw-filename',
        help="""Output file to save the raw collection (after pre-processing).
                If missing, the raw collection file is not generated.""")
    parser.add_argument('-t', '--tf-filename',
        help="""Output file to save the TF file in SVMlight format.
                If missing, the TF file is not generated.""")
    parser.add_argument('-v', '--vocab-filename',
        help="""Output file to save the vocabulary.
                If missing, the vocabulary file is not generated.""")
    args = parser.parse_args()
    if not (args.tf_filename or args.raw_filename or args.vocab_filename):
        raise Exception("No file is being generated. Please specify at least "
                        "one file to be generated.")

    count_vectorizer = \
        CountVectorizer(stop_words='english', min_df=args.min_df,
                        ngram_range=UNIGRAMS, encoding=ENCODING,
                        decode_error='ignore',
                        preprocessor=generate_preprocessor(replace_patterns))
    analyzer = count_vectorizer.build_analyzer()
    collection = list(parse_collection(args.classes, args.data))
    count_vectorizer, X, y = process_collection(count_vectorizer, collection)

    if args.raw_filename:
        dump_raw(generate_raw(count_vectorizer, collection), args.raw_filename)
    if args.tf_filename:
        dump_tf(X, y, args.tf_filename)
    if args.vocab_filename:
        dump_vocab(count_vectorizer, args.vocab_filename)
