import logging
import re
import sys
from pathlib import Path
# from fastnumbers import check_float
from collections import defaultdict
import math
import subprocess

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)


class Lexicon:
    def __init__(self, files) -> None:
        self.lexicon = defaultdict(list)
        for f in files:
            try:
                ans = self.read_lexicon(f)
            except:
                logging.info(f"Problem reading {f}")

            for w in ans.keys():
                if w not in self.lexicon:
                    self.lexicon[w] = ans[w]


    def read_lexicon(self, filename: str, ans = None):
        """Read a lexicon from `filename`.

        Each line in the lexicon contains "word p1 p2 p3 ...".
        That is, the first field is a word and the remaining
        fields are tokens. Fields are separated by space(s).

        Args:
        filename:
            Path to the lexicon.txt

        Returns:
        A dictionary, e.g., 
        {
            'w0': [(s00, ['p1', 'p2']), (s01, ['p1', 'p3'])], 
            'w1': [(s10, ['p3, 'p4'])],
            ...
        }
        """
        if ans is None:
            ans = defaultdict(list)

        with open(filename, "r", encoding="utf-8") as fin:
            # whitespace = re.compile("[ \t]+")
            for line in fin:
                a = line.strip()
                # a = line.strip().split()
                if len(a) == 0:
                    continue
                a = a.split("\t")

                if len(a) != 2 and len(a) != 6:
                    logging.info(f"Found bad line {line} in lexicon file {filename}")
                    logging.info("Every line is expected to contain at least 2 fields")
                    sys.exit(1)
                word = a[0]
                if word == "<eps>":
                    logging.info(f"Found bad line {line} in lexicon file {filename}")
                    logging.info("<eps> should not be a valid word")
                    sys.exit(1)

                # tokens = a[1:]
                # prob = 1.0  # probability
                # if check_float(tokens[0]):
                #     prob = float(tokens[0])
                #     tokens = tokens[1:]
                
                if len(a) == 2:
                    tokens = a[1].split()
                    prob = 1.0
                elif len(a) == 6:
                    tokens = a[5].split()
                    prob = float(a[1])
                
                ans[word].append([prob, tokens])
        
        # Normalization
        for word, pron_list in ans.items():
            total = sum([prob for prob, tokens in pron_list])
            for entry in pron_list:
                entry[0] /= total 
            pron_list.sort(key=lambda x: x[0], reverse=True)

        return ans

    def get_pron(self, word, limit=None):
        pron = self.lexicon.get(word, None)

        if pron is None:  # TODO: g2p online
            return None
    
        if limit is not None:
            pron = pron[:limit]
        
        return pron
