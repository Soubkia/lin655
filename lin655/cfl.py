import math
import nltk
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from operator import itemgetter

from .utils import SEEDS, Label, trigrams


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


@dataclass
class Model:
    frames: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))  # frame -> label -> score
    lexicon: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int))) # word  -> label -> score
    fthresh: int = field(default=15)
    wthresh: int = field(default=15)

    def __post_init__(self):
        for word, lbl in SEEDS:
            self.lexicon[word][lbl] = math.inf

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

    def best_frame(self, lexical_frame, label):
        if label in self.frames[lexical_frame]:
            return lexical_frame
        left_label = self.wlabel(lexical_frame.left)
        right_label = self.wlabel(lexical_frame.right)
        left_partial_frame = Frame(left_label, lexical_frame.right)
        right_partial_frame = Frame(lexical_frame.left, right_label)
        categorical_frame = Frame(left_label, right_label)
        if label in self.frames[left_partial_frame]:
            return left_partial_frame
        if label in self.frames[right_partial_frame]:
            return right_partial_frame
        if label in self.frames[categorical_frame]:
            return categorical_frame
        # No good frames exist so make a new lexical frame.
        return lexical_frame


    def train(self, text):
        for left, target, right in trigrams(text):
            frame = Frame(left, right)
            wlabel, flabel = self.wlabel(target), self.flabel(frame)
            # The target word is part of the trusted lexicon.
            if wlabel:
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

    def generalize(self):
        for lbl in Label:
            # Get relevant subset and compute tolerance principle.
            subset = defaultdict(lambda: defaultdict(int))
            for frame in self.frames:
                if not frame.is_lexical:
                    continue # Skip non-lexical frames.
                if self.frames[frame][lbl] > self.fthresh:
                    subset[frame][lbl] = self.frames[frame][lbl]
            N = len(subset)
            if N < 2:
                continue  # Need more data.
            threshold = N - (N / math.log(N))
            # Create all candidate frames.
            candidates = []
            for frame in subset:
                left_label, right_label = self.wlabel(frame.left), self.wlabel(frame.right)
                if left_label:  # Consider left partial frame.
                    candidates.append(Frame(left_label, frame.right))
                if right_label:  # Consider right partial frame.
                    candidates.append(Frame(frame.left, right_label))
                if left_label and right_label:  # Consider categorical frame.
                    candidates.append(Frame(left_label, right_label))
            candidates = nltk.FreqDist(candidates)
            # Create generalize frames.
            for frame, freq in candidates.items():
                if lbl in self.frames[frame]:
                    continue  # Already created this frame.
                if freq > threshold:
                    self.frames[frame][lbl] = 1

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
