from os import remove
from os.path import join
import re
from tqdm import tqdm
import pandas as pd
from textblob import TextBlob
from multiprocessing.pool import Pool
import nltk
import os
import codecs
from nltk.tag.perceptron import PerceptronTagger
from nltk.stem.wordnet import WordNetLemmatizer

_CONCEPT_NET_RELATIONS = {
    "/r/Antonym": "ConceptNet_antonym.csv",
    "/r/Synonym": "ConceptNet_synonym.csv",
    "/r/FormOf": "ConceptNet_form_of.csv",
    "/r/IsA": "ConceptNet_is_a.csv",
    "/r/PartOf": "ConceptNet_part_of.csv",
    "/r/HasA": "ConceptNet_has_a.csv",
    "/r/MannerOf": "ConceptNet_manner_of.csv",
    "/r/DerivedFrom": "ConceptNet_derived_from.csv",
    "/r/HasProperty": "ConceptNet_has_property.csv",
}

_CONCEPT_NET_PREFIX_ENGLISH = "/c/en/"
_CONCEPT_NET_POSTFIX_NOUN = "/n"
_CONCEPT_NET_POSTFIX_VERB = "/v"
_CONCEPT_NET_POSTFIX_ADJECTIVE = "/a"
_CONCEPT_NET_POSTFIX_R = "/r"


def extract_concept_net(file_path, out_path, chunk_size=1000):
    """Extracts ConceptNet relations from `file_path` and saves them to the `out_path` folder.
    The input file is read in `chunk_size` chunks at a time."""
    relations = {uri: join(out_path, file_name)
                 for uri, file_name in _CONCEPT_NET_RELATIONS.items()}

    # Remove the previous output files if they exists
    for out_path in relations.values():
        try:
            remove(out_path)
        except OSError:
            pass

    # Read the CSV in chunks, take only nouns and remove language tags

    for chunk in tqdm(pd.read_csv(file_path, sep="\t", chunksize=chunk_size,
                                  header=None, usecols=[1, 2, 3], dtype=str,
                                  encoding="utf-8")):
        noun_rows = chunk.loc[(chunk[2].str.startswith(_CONCEPT_NET_PREFIX_ENGLISH)) &
                              (chunk[3].str.startswith(_CONCEPT_NET_PREFIX_ENGLISH)) &
                              (~chunk[2].str.endswith(_CONCEPT_NET_POSTFIX_VERB)) &
                              (~chunk[2].str.endswith(_CONCEPT_NET_POSTFIX_ADJECTIVE)) &
                              (~chunk[2].str.endswith(_CONCEPT_NET_POSTFIX_R)) &
                              (~chunk[3].str.endswith(_CONCEPT_NET_POSTFIX_VERB)) &
                              (~chunk[3].str.endswith(_CONCEPT_NET_POSTFIX_ADJECTIVE)) &
                              (~chunk[3].str.endswith(_CONCEPT_NET_POSTFIX_R))]

        for rel_uri, rel_file in relations.items():
            relevant_rows = noun_rows.loc[chunk[1] == rel_uri]
            relevant_rows = (relevant_rows.replace(r"^/c/en/", "", regex=True)
                             .replace(r"/n$", "", regex=True))
            relevant_rows.to_csv(rel_file, header=False, columns=[2, 3], mode="a",
                                 encoding="utf-8", index=False, sep=" ")


_YAGO_RELATIONS = {
    "rdf:type": "YAGO_type.csv",
    "rdfs:subClassOf": "YAGO_subClassOf.csv",
}


def extract_yago(file_path, out_path, chunk_size=1000):
    """Extracts YAGO relations from `file_path` and saves them to the `out_path` folder.
    The input file is read in `chunk_size` chunks at a time."""
    relations = {uri: join(out_path, file_name)
                 for uri, file_name in _YAGO_RELATIONS.items()}

    print("Extracting relations from", file_path)
    for chunk in tqdm(pd.read_csv(file_path, sep="\t", chunksize=chunk_size,
                                  header=None, usecols=[1, 2, 3], dtype=str,
                                  encoding="utf-8", skiprows=1)):
        for rel_uri, rel_file in relations.items():
            # Find rows with rel_uri as category
            relevant_rows = chunk.loc[chunk[2] == rel_uri]

            # Isolate the words
            relevant_rows = (relevant_rows
                             .replace(r"^<", "", regex=True)
                             .replace(r">$", "", regex=True)
                             .replace(r"^wordnet_", "", regex=True)
                             .replace(r"^wikicat_", "", regex=True)
                             .replace(r"_\d+$", "", regex=True)
                             .replace("\"", ""))

            relevant_rows.to_csv(rel_file, header=False, columns=[1, 3], mode="a",
                                 encoding="utf-8", index=False, sep=" ")


# Match years like (2017), 1950s, 1785, 19th, 15th century
_YEAR_REGEX = r"^\(?((1|2)\d{3}s?|(\d?1st|\d?2nd|\d?\dth)((\s|-)century)?)\)?$"


def _clean_word(text):
    word_blob = TextBlob(text.replace("_", " "))

    parenthesis_opened_count = 0

    cleaned_word = []
    for word, tag in word_blob.tags:
        parenthesis_opened_count += word.count("(")
        parenthesis_opened_count -= word.count(")")

        # Skip while we're in a parenthesis
        if parenthesis_opened_count > 0:
            continue

        # Cut at prepositions or verb
        if tag == "IN" or tag.startswith("VB"):
            break

        # Ignore year numbers
        if not re.search(_YEAR_REGEX, word):
            word = word.split("/", )
            cleaned_word.append(word)

    # Convert to lower case
    return "_".join(cleaned_word).lower()


def _clean_row(row):
    try:
        word_a = str(row[0])
        word_b = str(row[1])
        word_a_cleaned = _clean_word(row[0])
        word_b_cleaned = _clean_word(row[1])

        if word_a_cleaned != word_b_cleaned:
            return word_a_cleaned, word_b_cleaned

    except:
        # Surround print in try since row could contain invalid characters
        try:
            print("Error in row", row)
        except:
            print("Error in row")

    return (None, None)


def clean_relation_words(file_path, out_path, chunk_size=1000):
    temp_out = "temp_relations.csv"

    print("Cleaning relations")
    with open(temp_out, "w", encoding="utf-8") as out_file:
        with Pool() as pool:
            for chunk in tqdm(pd.read_csv(file_path, sep=" ", chunksize=chunk_size,
                                          header=None, dtype=str,
                                          encoding="utf-8", skiprows=0)):
                word_pairs = [(row[1], row[2]) for row in chunk.itertuples()]
                cleaned_rows = pool.map(_clean_row, word_pairs)
                for word_a_cleaned, word_b_cleaned in cleaned_rows:
                    if word_a_cleaned is not None and word_b_cleaned is not None and word_a_cleaned != "" and word_b_cleaned != "":
                        out_file.write("%s %s\n" %
                                       (word_a_cleaned, word_b_cleaned))

    print("Removing duplicate relations")
    relation_df = pd.read_csv(temp_out, sep=" ")
    relation_df.drop_duplicates(subset=None, inplace=True)
    relation_df.to_csv(out_path, header=False,
                       encoding="utf-8", index=False, sep=" ")
    remove(temp_out)



# Refined version form hearst patterns extraction https://github.com/mmichelsonIF/hearst_patterns_python
# supporting mult word terms and filtering suth , other as adjectives for the noun phrases
# cleaning the list of hyponyms from stopwords, duplicates, siguralization,......

class HearstPatterns(object):
    def __init__(self):
        self.__chunk_patterns = r""" #  helps us find noun phrase chunks
         				NP: {<DT|PP\$>?<JJ>*<NN>+}
 					{<NNP>+}
 					{<NNS>+}
        		"""
        self.__np_chunker = nltk.RegexpParser(
            self.__chunk_patterns)  # create a chunk parser

        # now define the Hearst patterns
        # format is <hearst-pattern>, <general-term>
        # so, what this means is that if you apply the first pattern, the firsr Noun Phrase (NP)
        # is the general one, and the rest are specific NPs
        self.__hearst_patterns = [
            ("(NP_\w+ (, )?such as (NP_\w+ ? (, )?((and |or )NP_\w+)?)+)", "first"),  #
            ("(such NP_\w+ (, )?as (NP_\w+ ?(, )?(and |or )?)+)", "first"),
            ("((NP_\w+ ?(, )?)+(and |or )?other NP_\w+)", "last"),
            ("(NP_\w+ (, )?including (NP_\w+ ?(, )?(and |or )?)+)", "first"),
            ("(NP_\w+ (, )?especially (NP_\w+ ?(, )?(and |or )?)+)", "first"),
        ]

        self.__pos_tagger = PerceptronTagger()

    # divid text into sentences, tokenze the setnences and add par of speech tagging
    def prepare(self, rawtext):
        # NLTK default sentence segmenter
        sentences = nltk.sent_tokenize(rawtext.strip())
        sentences = [nltk.word_tokenize(sent)
                     for sent in sentences]  # NLTK word tokenizer
        sentences = [self.__pos_tagger.tag(sent)
                     for sent in sentences]  # NLTK POS tagger

        return sentences

    # apply chunking step on the setences using the defined grammer for extracting nounphrases
    def chunk(self, rawtext):
        sentences = self.prepare(rawtext.strip())

        all_chunks = []

        for sentence in sentences:
            chunks = self.__np_chunker.parse(
                sentence)  # parse the example sentence
            all_chunks.append(self.prepare_chunks(chunks))
        return all_chunks

    # annotate the np with a prefiy NP_ also exclude the other and such from NP to be used in the patterns
    def prepare_chunks(self, chunks):
        # basically, if the chunk is NP, keep it as a string that starts w/ NP and replace " " with _
        # otherwise, keep the word.
        # remove punct
        # this is all done to make it super easy to apply the Hearst patterns...

        terms = []
        for chunk in chunks:
            label = None
            try:  # gross hack to see if the chunk is simply a word or a NP, as we want. But non-NP fail on this method call
                label = chunk.label()
            except:
                pass

            if label is None:  # means one word...
                token = chunk[0]
                pos = chunk[1]
                if pos in ['.', ':', '-', '_']:
                    continue
                terms.append(token)
            else:
                if chunk[0][0] == 'such':
                    np = "such NP_" + "_".join([a[0] for a in chunk[1:]])
                elif chunk[0][0] == 'other':
                    np = "other NP_" + "_".join([a[0] for a in chunk[1:]])
                else:
                    # This makes it easy to apply the Hearst patterns later
                    np = "NP_" + "_".join([a[0] for a in chunk])
                terms.append(np)
        return ' '.join(terms)

    # main method for extracting hyponym relations based on hearst patterns
    def find_hyponyms(self, folderpath, stopWord):
        hyponyms = []

        print("Finding files in", folderpath)
        filelist = os.listdir(folderpath)
        print("Found", len(filelist), "files")
        for filePath in tqdm(filelist):
            full_path = os.path.join(folderpath, filePath)
            if not os.path.isfile(full_path):
                continue

            #print("processing file ", "........................", filePath)
            file = codecs.open(full_path, "r", encoding='utf-8', errors='ignore')
            lines = file.readlines()
            rawtext = (''.join(lines))
            rawtext = rawtext.lower()
            np_tagged_sentences = self.chunk(rawtext)

            #print("Processing tagged sentences")
            for raw_sentence in np_tagged_sentences:
                # two or more NPs next to each other should be merged into a single NP, it's a chunk error
                # find any N consecutive NP_ and merge them into one...
                # So, something like: "NP_foo NP_bar blah blah" becomes "NP_foo_bar blah blah"
                sentence = re.sub(
                    r"(NP_\w+ NP_\w+)+", lambda m: m.expand(r'\1').replace(" NP_", "_"), raw_sentence)
                # print  sentence
                for (hearst_pattern, parser) in self.__hearst_patterns:
                    matches = re.search(hearst_pattern, sentence)

                    if matches:
                        # print sentence
                        match_str = matches.group(1)
                        nps = [a for a in match_str.split()
                               if a.startswith("NP_")]
                        if parser == "first":
                            general = nps[0]
                            specifics = nps[1:]
                        else:
                            general = nps[-1]
                            specifics = nps[:-1]
                        for i in range(len(specifics)):
                            # print "%s, %s" % (general, specifics[i])
                            hyponyms.append((self.clean_hyponym_term(
                                general), self.clean_hyponym_term(specifics[i])))
            file.close()

        return self.refine_hyponym_term(hyponyms, stopWord)

    def clean_hyponym_term(self, term):
        return term.replace("NP_", "").replace("_", " ")

    # remove stopwprds and sniguralize specfic and general concepts
    def refine_hyponym_term(self, hyponyms, stopWord):
        wnl = WordNetLemmatizer()

        cleanedHyponyms = []
        with open(stopWord) as f:
            stopWords = f.read().splitlines()

        for hyponym in hyponyms:
            #print(hyponym)
            specific = ' '.join([i for i in hyponym[1].split(
                ' ') if not any(w == i.lower() for w in stopWords)])
            general = ' '.join([i for i in hyponym[0].split(
                ' ') if not any(w == i.lower() for w in stopWords)])
            if specific == '' or general == '':
                #print('skipped relation: ', hyponym[1], 'is a ', hyponym[0])
                continue
            cleanedHyponyms.append(
                (wnl.lemmatize(general), wnl.lemmatize(specific)))

        cleanedHyponyms.sort()
        # return self.remove_duplicates(cleanedHyponyms)
        return self.get_occurence_dict(cleanedHyponyms)

    # remove duplicates in hyonym list
    def remove_duplicates(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def get_occurence_dict(self, seq):
        unique_words = set(seq)
        return {word: seq.count(word) for word in unique_words}

def extract_relations_hearst(corpus_path, stop_words_path):
    h = hearstPatterns.HearstPatterns()
    return h.find_hyponyms(corpus_path, stop_words_path)
