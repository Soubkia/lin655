import conllu
import enum
import functools
import nltk
import pathlib
from dataclasses import dataclass
from operator import itemgetter


def trigrams(text):
    for snt in nltk.tokenize.sent_tokenize(text):
        words = [word.lower() for word in nltk.tokenize.word_tokenize(snt) if word.isalpha()]
        for trigram in nltk.ngrams(words, 3):
            yield trigram


@functools.lru_cache
def get_unimorph(with_lemmas=True):
    path = pathlib.Path(
        __file__
    ).resolve().parent.parent.joinpath(
        "libs", "eng", "eng"
    ).as_posix()

    unimorph = {}
    with open(path) as fd:
        for line in fd:
            if not line.strip().split():
                continue  # Skip empty lines
            lemma, word, tags = line.strip().split()
            if with_lemmas:
                unimorph[word] = lemma, tuple(tags.split(";"))
            else:
                unimorph[word] = tuple(tags.split(";"))
    return unimorph


@dataclass(frozen=True)
class Frame:
    left: str
    right: str
    
    @property
    def is_lexical(self):
        return isinstance(self.left, str) and isinstance(self.right, str)
    
    @property
    def is_partial(self):
        return (
            (isinstance(self.left, Label) and isinstance(self.right, str)) or
            (isinstance(self.left, str) and isinstance(self.right, Label))
        )
    
    @property
    def is_categorical(self):
        return isinstance(self.left, Label) and isinstance(self.right, Label)


class Label(enum.Enum):
    ADJ = "ADJ"
    ADV = "ADV"
    NOUN = "NOUN"
    PRO = "PRO"
    DET = "DET"
    CONJ = "CONJ"
    PREP = "PREP"
    VERB = "VERB"
    #  VERB_PAST = "VERB_PAST"
    #  VERB_PROG = "VERB_PROG"
    #  NOUN_PLRL = "NOUN_PLRL"


SEEDS = [
    ("you", Label.PRO),
    ("we", Label.PRO),
    ("me", Label.PRO),
    ("come", Label.VERB),
    ("play", Label.VERB),
    ("put", Label.VERB),
    ("on", Label.PREP),
    ("out", Label.PREP),
    ("in", Label.PREP),
    ("this", Label.DET),
    ("these", Label.DET),
    ("baby", Label.NOUN),
    ("car", Label.NOUN),
    ("train", Label.NOUN),
    ("box", Label.NOUN),
    ("house", Label.NOUN),
    ("boy", Label.NOUN),
    ("man", Label.NOUN),
    ("book", Label.NOUN),
    ("big", Label.ADJ),
    ("silly", Label.ADJ),
    ("green", Label.ADJ),
    ("well", Label.ADV),
    ("very", Label.ADV),
    ("now", Label.ADV),
    ("and", Label.CONJ),
    ("or", Label.CONJ),
    ("but", Label.CONJ),
    ("used", Label.VERB),
    ("looked", Label.VERB),
    ("called", Label.VERB),
    ("made", Label.VERB),
    #  ("being", Label.VERB),
    ("going", Label.VERB),
    ("playing", Label.VERB),
    #  ("days", Label.NOUN_PLRL),
    #  ("boys", Label.NOUN_PLRL),
    #  ("words", Label.NOUN_PLRL)
]
