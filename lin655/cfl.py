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

    def best_labeled_frame(self, lexical_frame, label):
        if lexical_frame in self.frames and label in self.frames[lexical_frame]:
            return lexical_frame
        left_label = self.wlabel(lexical_frame.left)
        right_label = self.wlabel(lexical_frame.right)
        left_partial_frame = Frame(left_label, lexical_frame.right)
        right_partial_frame = Frame(lexical_frame.left, right_label)
        categorical_frame = Frame(left_label, right_label)
        if left_partial_frame in self.frames and label in self.frames[left_partial_frame]:
            return left_partial_frame
        if right_partial_frame in self.frames and label in self.frames[right_partial_frame]:
            return right_partial_frame
        if categorical_frame in self.frames and label in self.frames[categorical_frame]:
            return categorical_frame
        # No good frames exist so make a new lexical frame.
        return lexical_frame

    def applicable_frames(self, lexical_frame):
        ret = [lexical_frame]
        left_label = self.wlabel(lexical_frame.left)
        right_label = self.wlabel(lexical_frame.right)
        left_partial_frame = Frame(left_label, lexical_frame.right)
        right_partial_frame = Frame(lexical_frame.left, right_label)
        categorical_frame = Frame(left_label, right_label)
        if left_partial_frame in self.frames:
            ret.append(left_partial_frame)
        if right_partial_frame in self.frames:
            ret.append(right_partial_frame)
        if categorical_frame in self.frames:
            ret.append(categorical_frame)
        return ret

    def best_trusted_frame(self, lexical_frame):
        # TODO: Just make this part of the class.
        # Order frames by most specific.
        def most_specific(frame):
            if frame.is_lexical:
                return 0
            elif frame.is_partial:
                return 1
            elif frame.is_categorical:
                return 2
        
        frames = sorted(
            list(
                set(self.applicable_frames(lexical_frame)) & set(self.trusted_frames.keys())
            ),
            key=most_specific
        )
        if frames:
            return frames[0]
        return None

    def train(self, text):
        import tqdm  # XXX: Delete me?

        for left, target, right in tqdm.tqdm(trigrams(text)):
            frame = Frame(left, right)
            wlabel, flabel = self.wlabel(target), self.flabel(frame)
            trusted_frame = self.best_trusted_frame(frame)
            # The target word is part of the trusted lexicon.
            if wlabel:
                best_frame = self.best_labeled_frame(frame, wlabel)
                # Update frame labels.
                self.frames[best_frame][wlabel] += 1
                for frm in self.applicable_frames(frame):
                    if frm == best_frame:
                        continue  # Do not punish good boys.
                    if frm in self.frames:
                        for lbl in self.frames[frm]:
                            self.frames[frm][lbl] -= 1
            # The frame is a trusted context.
            if trusted_frame:
                flabel, fscore = max(self.frames[trusted_frame].items(), key=itemgetter(1))
                self.lexicon[target][flabel] += 1
                for lbl in self.lexicon[target]:
                    if lbl != flabel:
                        self.lexicon[target][lbl] -= 0.75
            # If learning happened try to generalize.
            if wlabel or trusted_frame:
                self.generalize()

    def update_frames(self, frame):
        pass

    def update_lexicon(self, frame):
        pass

    def generalize(self):
        for lbl in Label:
            # Get relevant subset and compute tolerance principle.
            subset = defaultdict(lambda: defaultdict(int))
            for frame in self.frames:
                if not frame.is_lexical:
                    continue # Skip non-lexical frames.
                if lbl in self.frames[frame] and self.frames[frame][lbl] > self.fthresh:
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
                if frame in self.frames and lbl in self.frames[frame]:
                    continue  # Already created this frame.
                if freq > threshold:
                    self.frames[frame][lbl] = 1

    # TODO: Should this be taking the max?
    def wlabel(self, word):
        # Retrieve the highest scoring label for the word.
        if word not in self.lexicon:
            return None
        label, score = max(self.lexicon[word].items(), key=itemgetter(1))
        if score <= self.wthresh:
            return None
        return label

    # XXX: Delete me.
    def flabel(self, frame):
        if frame not in self.frames:
            return None
        label, score = max(self.frames[frame].items(), key=itemgetter(1))
        if score <= self.fthresh:
            return None
        return label
