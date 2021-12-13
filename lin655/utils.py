import conllu
import enum
import nltk
import pathlib
from dataclasses import dataclass
from operator import itemgetter


def trigrams(text):
    for snt in nltk.tokenize.sent_tokenize(text):
        words = [word.lower() for word in nltk.tokenize.word_tokenize(snt) if word.isalpha()]
        for trigram in nltk.ngrams(words, 3):
            yield trigram


def corpus():
    path = pathlib.Path(
        __file__
    ).resolve().parent.parent.joinpath(
        "libs", "UD_English-GUM", "en_gum-ud-train.conllu"
    ).as_posix()

    with open(path) as fd:
        return conllu.parse(fd.read())


def corpus_raw():
    return " ".join([" ".join(map(itemgetter("form"), snt)) for snt in corpus()])


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
    VERB_PAST = "VERB_PAST"
    VERB_PROG = "VERB_PROG"
    NOUN_PLRL = "NOUN_PLRL"


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
    ("used", Label.VERB_PAST),
    ("looked", Label.VERB_PAST),
    ("called", Label.VERB_PAST),
    ("made", Label.VERB_PAST),
    ("being", Label.VERB_PROG),
    ("going", Label.VERB_PROG),
    ("playing", Label.VERB_PROG),
    ("days", Label.NOUN_PLRL),
    ("boys", Label.NOUN_PLRL),
    ("words", Label.NOUN_PLRL)
]
