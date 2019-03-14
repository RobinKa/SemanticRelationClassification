import re
import os
from glob import glob
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk import download
from tqdm import tqdm

def download_preprocessing_prerequisites():
    """Downloads the NLTK prerequisites needed for other functions"""
    download("averaged_perceptron_tagger")  # POS Tags
    download("stopwords")  # Stop words
    download("brown")  # Noun phrases
    download("punkt")  # Noun phrases


def text_blob_from_file(file_path):
    """Loads a `TextBlob` from `file_path`"""
    with open(file_path, "r", encoding="utf-8") as text_file:
        return TextBlob(text_file.read())


def remove_stop_words(text_blob):
    """Removes all stop words from `text_blob` and returns the resulting `TextBlob`"""
    # Get words from original text, remove the stop words and combine the
    # words again
    words = text_blob.words

    stop_words = [stop_word.lower()
                  for stop_word in stopwords.words("english")]

    words = filter(lambda word: not word.lower() in stop_words, words)

    return TextBlob(" ".join(words))


def find_noun_phrases(text_blob):
    """Returns all noun phrases found in `text_blob`"""
    tags = text_blob.tags

    noun_phrases = []
    current_noun_phrase = []
    current_noun_phrase_pos = []

    # Find the noun phrases sequentially based on the POS tags

    for (word, pos) in tags:
        if re.match("^[a-zA-Z]*$", word):
            if current_noun_phrase == [] or current_noun_phrase_pos[-1] == "JJ":
                if pos in ["NN", "NNS", "NP", "NPS", "JJ"]:
                    current_noun_phrase.append(word)
                    current_noun_phrase_pos.append(pos)
            else:
                if pos in ["NN", "NNS", "NP", "NPS"]:
                    current_noun_phrase.append(word)
                    current_noun_phrase_pos.append(pos)
                else:
                    if ((len(current_noun_phrase) == 1 and not current_noun_phrase_pos[0] == "JJ")
                            or len(current_noun_phrase) > 1):
                        noun_phrases.append(" ".join(current_noun_phrase))
                    current_noun_phrase = []
                    current_noun_phrase_pos = []

    return noun_phrases


def link_noun_phrases(text_blob):
    """Returns a `TextBlob` with all noun phrases in `text_blob` linked by underscores"""
    #noun_phrases = text_blob.noun_phrases
    noun_phrases = find_noun_phrases(text_blob)

    # Sort the noun phrases by occurences of spaces so we replace those first
    noun_phrases = sorted(noun_phrases, reverse=True,
                          key=lambda np: np.count(" "))

    # Select only noun phrases that don't consist of single words (ie. at least a space or hyphen)
    # Replace all spaces with underscores and remove hyphens

    replacements = [(np, np.replace(" ", "_").replace("-", "")) for np in
                    filter(lambda word: word.count(" ") > 0 or word.count("-") > 0, noun_phrases)]

    text_blob_str = str(text_blob)

    for noun_phrase, joined_noun_phrase in replacements:
        text_blob_str = text_blob_str.replace(noun_phrase, joined_noun_phrase)

    return TextBlob(text_blob_str)


def convert_wiki_dump(wiki_dump_path, out_path, wiki_extractor_path):
    """Converts a wikipedia dump at `wiki_dump_path` to multiple text files
    saved to `out_path` using the WikiExtractor.py script at `wiki_extractor_path`"""
    print("Extracting data from wikidump")
    #os.system("python %s %s -b 1000M -q -o %s" %
    #          (wiki_extractor_path, wiki_dump_path, out_path))

    print("Converting xml to text files")
    _split_wiki_articles(out_path, out_path)


def _get_wiki_article_title(article):
    """This function finds the article name for an Wikipedia article"""
    title = re.findall(r"(title=\")(.+?)(\")", article)
    if len(title) == 0 or len(title[0]) <= 1:
        return None

    return title[0][1]


def _split_wiki_articles(raw_article_file_path, article_out_path):
    """This script is used to split Wikipedia articles extracted from a Wikipedia
    dump into seperate files for every article"""
    wiki_files = glob(os.path.join(raw_article_file_path, "AA", "wiki_*"))
    print("Found", len(wiki_files), "files to process")
    for raw_file_path in wiki_files:
        print("Processing", raw_file_path)
        with open(raw_file_path, "r") as raw_file:
            articles = re.split("<doc", raw_file.read())[2:]
            for article in tqdm(articles):
                title = _get_wiki_article_title(article)
                if title is not None and "/" not in title:
                    article_path = os.path.join(article_out_path, title + ".txt")
                    with open(article_path, "w") as out_file:
                        out_file.writelines(
                            "\n".join(article.split("\n")[3:-3]).lower())
