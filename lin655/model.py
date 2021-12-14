import math
import pathlib
import sys
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from operator import attrgetter, itemgetter

from .utils import SEEDS, Frame, Label, trigrams, get_unimorph


sys.path.append(pathlib.Path(__file__).resolve().parent.parent.joinpath("libs", "ATP-morphology", "src").as_posix())


@dataclass
class Model:
    frames: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))  # frame -> label -> score
    lexicon: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int))) # word  -> label -> score
    fthresh: int = field(default=5)
    wthresh: int = field(default=5)
    use_atp: bool = field(default=False)

    def __post_init__(self):
        for word, lbl in SEEDS:
            self.lexicon[word][lbl] = math.inf
        if self.use_atp:
            self.generalize()

    @property
    def trusted_lexicon(self):
        ret = defaultdict(lambda: defaultdict(int))
        for word in self.lexicon:
            for lbl in self.lexicon[word]: 
                if self.lexicon[word][lbl] > self.wthresh:
                    ret[word][lbl] = self.lexicon[word][lbl]
        return ret

    @property
    def trusted_frames(self):
        ret = defaultdict(lambda: defaultdict(int))
        for frame in self.frames:
            for lbl in self.frames[frame]:
                if self.frames[frame][lbl] > self.fthresh:
                    ret[frame][lbl] = self.frames[frame][lbl]
        return ret

    def train(self, text):
        import tqdm
        counter = 0
        for left, target, right in tqdm.tqdm(trigrams(text)):
            frame = Frame(left, right)
            wlabel, flabel = self.wlabel(target), self.flabel(frame)
            # The target word is part of the trusted lexicon.
            if wlabel:
                counter += 1
                # Update frame labels.
                self.frames[frame][wlabel] += 1
                for lbl in self.frames[frame]:
                    if lbl != wlabel:
                        self.frames[frame][lbl] -= 1
            # The frame is a trusted context.
            if flabel:
                # Update word labels.
                self.lexicon[target][flabel] += 1
                for lbl in self.lexicon[target]:
                    if lbl != flabel:
                        self.lexicon[target][lbl] -= 0.75
            if self.use_atp and counter % (self.wthresh * 1000) == 0:
                self.generalize()

    def generalize(self):
        from nltk.stem import WordNetLemmatizer
        from atp import ATP

        if not self.use_atp:
            return
        unimorph = get_unimorph()
        feature_space = set()
        pairs = []
        for word in self.trusted_lexicon:
            label, score = max(self.lexicon[word].items(), key=itemgetter(1))
            if label == Label.VERB:
                try:
                    lemma, tags = unimorph[word]
                except KeyError:
                    continue
                feature_space |= set(tags)
                pairs.append((lemma, word, tags))
        atp = ATP(feature_space=feature_space)
        atp.train(pairs)
        for word in self.trusted_lexicon:
            label, score = max(self.lexicon[word].items(), key=itemgetter(1))
            if label == Label.VERB:
                try:
                    lemma, _ = unimorph[word]
                except KeyError:
                    continue
                tags = [("V", "PST"), ("V", "V.PTCP", "PRS"), ("V", "3", "SG", "PRS")]
                for tag in tags:
                    inflection = atp.inflect(lemma, tag)
                    self.lexicon[inflection][Label.VERB] = max(20, self.lexicon[inflection][Label.VERB])


        #  lemmatizer = WordNetLemmatizer()  # XXX: Use universal dependencies.
        #  feature_space = (Label.VERB_PAST, Label.VERB_PROG, Label.NOUN_PLRL)
        #  pairs = []
        #  for word in self.trusted_lexicon:
        #      label, score = max(self.lexicon[word].items(), key=itemgetter(1))
        #      if label in (Label.VERB_PAST, Label.VERB_PROG):
        #          pairs.append((lemmatizer.lemmatize(word, pos="v"), word, (label.name,)))
        #      if label == Label.NOUN_PLRL:
        #          pairs.append((lemmatizer.lemmatize(word, pos="n"), word, (label.name,)))
        #  atp = ATP(feature_space=list(map(attrgetter("name"), feature_space)))
        #  atp.train(pairs)
        #  for word in self.trusted_lexicon:
        #      label, score = max(self.lexicon[word].items(), key=itemgetter(1))
        #      if label == Label.VERB:
        #          self.lexicon[atp.inflect(word, (Label.VERB_PAST.name,))][Label.VERB_PAST] = 20
        #          self.lexicon[atp.inflect(word, (Label.VERB_PROG.name,))][Label.VERB_PROG] = 20
        #      if label == Label.NOUN:
        #          self.lexicon[atp.inflect(word, (Label.NOUN_PLRL.name,))][Label.NOUN_PLRL] = 20

    # TODO: This should not be taking the "max"
    def wlabel(self, word):
        # Retrieve the highest scoring label for the word.
        if word not in self.lexicon:
            return None
        label, score = max(self.lexicon[word].items(), key=itemgetter(1))
        if score <= self.wthresh:
            return None
        return label

    def flabel(self, frame):
        if frame not in self.frames:
            return None
        label, score = max(self.frames[frame].items(), key=itemgetter(1))
        if score <= self.fthresh:
            return None
        return label


    def get_lexicon_df(self):
        import nltk
        from pandas import DataFrame

        dct = defaultdict(set)
        for tree in nltk.corpus.semcor.tagged_chunks():
            if len(tree.leaves()) != 1:
                continue  # Skip complicated tags.
            for word in tree.leaves():
                if word.isalpha():
                    dct[word.lower()].add(tree.label())
        unimorph = get_unimorph(with_lemmas=False)

        data = []
        for word in self.lexicon:
            label, score = max(self.lexicon[word].items(), key=itemgetter(1))
            correct = False
            # XXX: This is the worst thing I have ever written.
            if label == Label.NOUN:
                parts = ("NN", "NNP", "NNS", "NNPS")
                if any([pos in dct[word] for pos in parts]):
                    correct = True
            #  elif label == Label.NOUN_PLRL:
            #      parts = ("NNS", "NNPS")
            #      if any([pos in dct[word] for pos in parts]):
            #          correct = True
            elif label == Label.ADJ:
                parts = ("JJ", "JJR", "JJS")
                if any([pos in dct[word] for pos in parts]):
                    correct = True
            elif label == Label.ADV:
                parts = ("RB", "RBR", "RBS", "WRB")
                if any([pos in dct[word] for pos in parts]):
                    correct = True
            elif label == Label.CONJ:
                parts = ("CC",)
                if any([pos in dct[word] for pos in parts]):
                    correct = True
            elif label == Label.DET:
                parts = ("DT", "PDT", "WDT")
                if any([pos in dct[word] for pos in parts]):
                    correct = True
            elif label == Label.PRO:
                parts = ("PRP", "PRP$", "WP", "WP$")
                if any([pos in dct[word] for pos in parts]):
                    correct = True
            elif label == Label.PREP:
                parts = ("IN",)
                if any([pos in dct[word] for pos in parts]):
                    correct = True
            elif label == Label.VERB:
                parts = ("VB", "VBG", "VBP", "VBZ", "VBD", "VBN")
                if word in unimorph:
                    correct = True
                elif any([pos in dct[word] for pos in parts]):
                    correct = True
            #  if label == Label.VERB_PAST:
            #      if all([ftr in unimorph.get(word, []) for ftr in ("V", "PST")]):
            #          correct = True
            #  elif label == Label.VERB_PROG:
            #      if word.endswith("ing"):
            #          correct = True  # XXX: Not perfect probably.
            data.append({
                "word": word,
                "label": label.name,
                "score": score,
                "correct": correct
            })
        return DataFrame(data).set_index("word") 
