import logging
import argparse
import buckeye
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type
import typing
import collections
from Bio import pairwise2
import functools
import glob
import dataclassy
import yaml
import textgrid
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm_loggable.auto import tqdm
except:
    from tqdm import tqdm

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

def parse_opts():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ali-dir', type=str, default=None, help='')
    parser.add_argument('--ali-pattern', type=str, default=None, help='')
    parser.add_argument('--ref-dir', type=str, default="/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/temp3/", help='')
    parser.add_argument('--out-dir', type=str, default="audio_ruizhe/mfa/temptemp/", help='')
    parser.add_argument('--word-level', action='store_true', default=False, help='')
    parser.add_argument('--silence-phone', type=str, default="-", help='')

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


# https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/montreal_forced_aligner/data.py#L1697
@dataclassy.dataclass(slots=True)
class CtmInterval:
    """
    Data class for intervals derived from CTM files

    Parameters
    ----------
    begin: float
        Start time of interval
    end: float
        End time of interval
    label: str
        Text of interval
    confidence: float, optional
        Confidence score of the interval
    """

    begin: float
    end: float
    label: typing.Union[int, str]
    confidence: typing.Optional[float] = None

    def __lt__(self, other):
        """Sorting function for CtmIntervals"""
        return self.begin < other.begin

    def __add__(self, other):
        if isinstance(other, str):
            return self.label + other
        else:
            self.begin += other
            self.end += other

    def __post_init__(self) -> None:
        """
        Check on data validity

        Raises
        ------
        :class:`~montreal_forced_aligner.exceptions.CtmError`
            If begin or end are not valid
        """
        if self.end < -1 or self.begin == 1000000:
            raise ValueError(self)


# https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/main/montreal_forced_aligner/helper.py#L491
def compare_labels(
    ref: str, test: str, silence_phone: str, mapping: Optional[Dict[str, str]] = None
) -> int:
    """

    Parameters
    ----------
    ref: str
    test: str
    mapping: Optional[dict[str, str]]

    Returns
    -------
    int
        0 if labels match or they're in mapping, 2 otherwise
    """
    if ref == test:
        return 0
    if ref == silence_phone or test == silence_phone:
        return 10
    if mapping is not None and test in mapping:
        if isinstance(mapping[test], str):
            if mapping[test] == ref:
                return 0
        elif ref in mapping[test]:
            return 0
    ref = ref.lower()
    test = test.lower()
    if ref == test:
        return 0
    return 2


def overlap_scoring(
    first_element: CtmInterval,
    second_element: CtmInterval,
    silence_phone: str,
    mapping: Optional[Dict[str, str]] = None,
) -> float:
    r"""
    Method to calculate overlap scoring

    .. math::

       Score = -(\lvert begin_{1} - begin_{2} \rvert + \lvert end_{1} - end_{2} \rvert + \begin{cases}
                0, & if label_{1} = label_{2} \\
                2, & otherwise
                \end{cases})

    See Also
    --------
    `Blog post <https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html>`_
        For a detailed example that using this metric

    Parameters
    ----------
    first_element: :class:`~montreal_forced_aligner.data.CtmInterval`
        First CTM interval to compare
    second_element: :class:`~montreal_forced_aligner.data.CtmInterval`
        Second CTM interval
    mapping: Optional[dict[str, str]]
        Optional mapping of phones to treat as matches even if they have different symbols

    Returns
    -------
    float
        Score calculated as the negative sum of the absolute different in begin timestamps, absolute difference in end
        timestamps and the label score
    """
    begin_diff = abs(first_element.begin - second_element.begin)
    end_diff = abs(first_element.end - second_element.end)
    label_diff = compare_labels(first_element.label, second_element.label, silence_phone, mapping)
    return -1 * (begin_diff + end_diff + label_diff)


def align_phones(
    ref: List[CtmInterval],
    test: List[CtmInterval],
    silence_phone: str,
    ignored_phones: typing.Set[str] = None,
    custom_mapping: Optional[Dict[str, str]] = None,
    debug: bool = False,
    stats: dict = {},
) -> Tuple[float, float, Dict[Tuple[str, str], int]]:
    """
    Align phones based on how much they overlap and their phone label, with the ability to specify a custom mapping for
    different phone labels to be scored as if they're the same phone

    Parameters
    ----------
    ref: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals as reference
    test: list[:class:`~montreal_forced_aligner.data.CtmInterval`]
        List of CTM intervals to compare to reference
    silence_phone: str
        Silence phone (these are ignored in the final calculation)
    custom_mapping: dict[str, str], optional
        Mapping of phones to treat as matches even if they have different symbols
    debug: bool, optional
        Flag for logging extra information about alignments

    Returns
    -------
    float
        Score based on the average amount of overlap in phone intervals
    float
        Phone error rate
    dict[tuple[str, str], int]
        Dictionary of error pairs with their counts
    """

    if ignored_phones is None:
        ignored_phones = set()
    if custom_mapping is None:
        score_func = functools.partial(
            overlap_scoring, silence_phone=silence_phone
        )
    else:
        score_func = functools.partial(
            overlap_scoring, silence_phone=silence_phone, mapping=custom_mapping
        )

    alignments = pairwise2.align.globalcs(
        ref, test, score_func, -2, -2, gap_char=["-"], one_alignment_only=True
    )
    overlap_count = 0
    overlap_sum = 0
    overlap_sum_begin = 0
    overlap_sum_end = 0
    overlap_sum_durA = 0
    overlap_sum_durB = 0
    num_insertions = 0
    num_deletions = 0
    num_substitutions = 0

    pos_cnt_begin = 0
    zero_cnt_begin = 0
    neg_cnt_begin = 0
    pos_cnt_end = 0
    zero_cnt_end = 0
    neg_cnt_end = 0

    my_stats = dict()

    errors = collections.Counter()
    for a in alignments:
        for i, sa in enumerate(a.seqA):
            sb = a.seqB[i]
            if sa == "-":
                if sb.label not in ignored_phones:
                    errors[(sa, sb.label)] += 1
                    num_insertions += 1
                else:
                    continue
            elif sb == "-":
                if sa.label not in ignored_phones:
                    errors[(sa.label, sb)] += 1
                    num_deletions += 1
                else:
                    continue
            else:
                if sa.label in ignored_phones:
                    continue
                overlap_sum += (abs(sa.begin - sb.begin) + abs(sa.end - sb.end)) / 2
                overlap_sum_begin += abs(sa.begin - sb.begin)
                overlap_sum_end += abs(sa.end - sb.end)
                
                if sa.begin - sb.begin > 0:
                    pos_cnt_begin += 1
                elif sa.begin - sb.begin == 0:
                    zero_cnt_begin += 1
                else:
                    neg_cnt_begin += 1
                
                if sa.end - sb.end > 0:
                    pos_cnt_end += 1
                elif sa.end - sb.end == 0:
                    zero_cnt_end += 1
                else:
                    neg_cnt_end += 1
                
                stats["phones_distribution_begin"][sa.label].append(sb.begin - sa.begin)
                stats["phones_distribution_end"][sa.label].append(sb.end - sa.end)
                stats["phones_distribution_mid"][sa.label].append((sb.begin + sb.end)/2 - (sa.begin + sa.end)/2)
                    
                overlap_sum_durA += abs(sa.end - sa.begin)
                overlap_sum_durB += abs(sb.end - sb.begin)
                overlap_count += 1
                if compare_labels(sa.label, sb.label, silence_phone, mapping=custom_mapping) > 0:
                    num_substitutions += 1
                    errors[(sa.label, sb.label)] += 1
    if debug:
        logging.debug(pairwise2.format_alignment(*alignments[0]))
    if overlap_count:
        score = overlap_sum / overlap_count
        score_begin = overlap_sum_begin / overlap_count
        score_end = overlap_sum_end / overlap_count
    else:
        logging.warning(f"overlap_count={overlap_count}: {ref}")
        score = 0
        score_begin = 0
        score_end = 0
    
    phone_error_rate = (num_insertions + num_deletions + (2 * num_substitutions)) / len(ref)
    
    assert pos_cnt_begin + zero_cnt_begin + neg_cnt_begin + pos_cnt_end + zero_cnt_end + neg_cnt_end == overlap_count * 2
    pos = (pos_cnt_begin, zero_cnt_begin, neg_cnt_begin, pos_cnt_end, zero_cnt_end, neg_cnt_end)
    
    stats["alignments"] = alignments

    return score, score_begin, score_end, phone_error_rate, errors, overlap_count, overlap_sum_durA, overlap_sum_durB, pos


def read_time_file(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()

    start_time = float(lines[0].strip().split()[1])
    end_time = float(lines[1].strip().split()[1])

    words = lines[3].strip().split()
    words_start = list(map(float, lines[4].strip().split()))
    words_end = list(map(float, lines[5].strip().split()))
    assert len(words) == len(words_start) and len(words) == len(words_end)

    phones = lines[7].strip().split()
    phones_start = list(map(float, lines[8].strip().split()))
    phones_end = list(map(float, lines[9].strip().split()))
    assert len(phones) == len(phones_start) and len(phones) == len(phones_end)

    return start_time, end_time, words, words_start, words_end, phones, phones_start, phones_end


def read_textgrid_file(filename):
    tg = textgrid.TextGrid.fromFile(filename)
    assert tg[0].name == "words"
    assert tg[1].name == "phones"

    start_time, end_time = tg.minTime, tg.maxTime

    words = []
    words_start = []
    words_end = []
    for inteval in tg[0]:
        word = inteval.mark
        if len(word) == 0:
            # word = "<eps>"
            continue
        words.append(word)
        words_start.append(inteval.minTime)
        words_end.append(inteval.maxTime)
    
    phones = []
    phones_start = []
    phones_end = []
    for inteval in tg[1]:
        phone = inteval.mark
        if len(phone) == 0:
            # phone = "<eps>"
            continue
        phones.append(phone)
        phones_start.append(inteval.minTime)
        phones_end.append(inteval.maxTime)
    
    return start_time, end_time, words, words_start, words_end, phones, phones_start, phones_end


def read_pt_file(filename):
    # ali = torch.load(filename)
    with open(filename, 'rb') as file:      
        # Call load method to deserialze
        ali = pickle.load(file)

    ret = dict()
    for (utter_id, rs) in ali:
        # if utter_id.startswith("s13"): continue
        ret[utter_id] = rs
        # time_start, time_end, words_text, word_times_start, word_times_end, phones_text, phones_beg_time, phones_end_time = rs
    return ret


def get_ctm_list(filename):
    if isinstance(filename, tuple):
        start_time, end_time, words, words_start, words_end, phones, phones_start, phones_end = filename
    elif filename.endswith(".time"):
        start_time, end_time, words, words_start, words_end, phones, phones_start, phones_end = \
            read_time_file(filename)
    else:
        start_time, end_time, words, words_start, words_end, phones, phones_start, phones_end = \
            read_textgrid_file(filename)
        
    words_ctm_list = [
        CtmInterval(begin=beg, end=end, label=w.lower().replace("▁", "")) for w, beg, end in zip(words, words_start, words_end)
    ]

    phones_ctm_list = [
        CtmInterval(begin=beg, end=end, label=p) for p, beg, end in zip(phones, phones_start, phones_end)
    ]

    return words_ctm_list, phones_ctm_list

def plot_histogram(mylist, path='test.pdf'):
    plt.set_loglevel("info")

    mylist = np.asarray(mylist)

    binwidth = 0.01
    plt.hist(mylist, bins=int(mylist.max()/binwidth))
    # plt.show() 

    plt.savefig(path, bbox_inches='tight')


def plot_boxplot(mylists, path="test.pdf"):
    # https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/
    pass


def evaluate_alignments(
    mapping: Optional[Dict[str, str]] = None,
    output_directory: Optional[str] = None,
    ali_dir: str = None,
    ref_dir: str = None,
    ali_file_pattern = None,
    silence_phone="-",
    ignored_phones=None,
    debug=False,
    word_level=False,
) -> None:
    logging.info(f"ali_dir={ali_dir}")
    logging.info(f"ali_file_pattern={ali_file_pattern}")

    if ali_file_pattern.endswith(".TextGrid"):  # MFA
        ali_files = glob.glob(f"{ali_dir}/{ali_file_pattern}", recursive=True)  # list(Path(ali_dir).rglob("*.TextGrid"))
        ali_files = {f.stem: str(f) for f in map(Path, ali_files)}
    elif ali_file_pattern.endswith(".pkl"):  # TorchAudio
        ali_files = {}
        fns = glob.glob(f"{ali_dir}/{ali_file_pattern}", recursive=True)
        logging.info(f"ali_files: {fns}")
        for f in fns:
            ali_files.update(read_pt_file(f))
    else:
        raise NotImplementedError

    # ref_files = glob.glob(f"{ref_dir}/*/*.time", recursive=True)
    ref_files = glob.glob(f"{ref_dir}/**/*.time", recursive=True)
    # ref_files = {f.stem: str(f) for f in map(Path, ref_files)}
    ref_files = {"_".join(str(f)[:-5].split("/")[-4:]): str(f) for f in map(Path, ref_files)}

    logging.info(f"ali_files: {len(ali_files)}")
    logging.info(f"ref_files: {len(ref_files)}")

    score_count = 0
    score_sum = 0
    score_sum_begin = 0
    score_sum_end = 0
    dur_sum_A = 0
    dur_sum_B = 0
    phone_edit_sum = 0
    phone_length_sum = 0
    phone_confusions = collections.Counter()
    overlap_count_sum = 0

    pos_cnt_begin_sum, zero_cnt_begin_sum, neg_cnt_begin_sum, pos_cnt_end_sum, zero_cnt_end_sum, neg_cnt_end_sum = 0,0,0,0,0,0

    # ali_files = {k: v for k, v in ali_files.items() if k.startswith("s03") or k.startswith("s13")}
    ali_files = {k: v for k, v in ali_files.items() if not k.startswith("s35")}
    stats = defaultdict(lambda: defaultdict(list))

    for fname, f_ali in tqdm(ali_files.items()):
        if fname not in ref_files: continue
        f_ref = ref_files[fname]

        # if fname == "s1301a-18":
        #     import pdb; pdb.set_trace()

        try:
            words_ali, phones_ali = get_ctm_list(f_ali)
            words_ref, phones_ref = get_ctm_list(f_ref)
        except:
            logging.warning(f"Problem reading files: {fname}")
            continue

        if word_level:
            ali = words_ali
            ref = words_ref
        else:
            ali = phones_ali
            ref = phones_ref

        if len(ref) == 0:
            # import pdb; pdb.set_trace()
            logging.warning(f"Empty ref: {fname}")
            continue

        phone_boundary_error, pbe_begin, pbe_end, phone_error_rate, errors, overlap_count, overlap_sum_durA, overlap_sum_durB, pos = align_phones(
            ref=ref, 
            test=ali, 
            silence_phone=silence_phone,
            ignored_phones=ignored_phones,
            custom_mapping=mapping,
            debug=debug,
            stats=stats,
        )
        pos_cnt_begin, zero_cnt_begin, neg_cnt_begin, pos_cnt_end, zero_cnt_end, neg_cnt_end = pos
        # import pdb; pdb.set_trace()

        stats["pbe"][fname] = phone_boundary_error
        stats["pbe_begin"][fname] = pbe_begin
        stats["pbe_end"][fname] = pbe_end
        stats["per"][fname] = phone_error_rate

        # print(phone_boundary_error, phone_error_rate, errors)
        # break

        reference_phone_count = len(ref)
        phone_confusions.update(errors)
        score_count += 1
        score_sum += phone_boundary_error
        score_sum_begin += pbe_begin
        score_sum_end += pbe_end
        phone_edit_sum += int(phone_error_rate * reference_phone_count)
        phone_length_sum += reference_phone_count
        overlap_count_sum += overlap_count
        dur_sum_A += overlap_sum_durA
        dur_sum_B += overlap_sum_durB

        pos_cnt_begin_sum += pos_cnt_begin
        zero_cnt_begin_sum += zero_cnt_begin
        neg_cnt_begin_sum += neg_cnt_begin
        pos_cnt_end_sum += pos_cnt_end
        zero_cnt_end_sum += zero_cnt_end
        neg_cnt_end_sum += neg_cnt_end

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    logging.info(f"output_directory: {output_directory}")

    utt2pbe = stats["pbe"]
    utt2pbe_sorted = [(k, v) for k, v in sorted(utt2pbe.items(), reverse=True, key=lambda item: item[1])]
    print("utt_id", "phone boundary error")
    for k, v in utt2pbe_sorted[:10]:
        print(k, v)
    print()
    plot_histogram(list(utt2pbe.values()), path=f"{output_directory}/pbe_dist.pdf")
    with open(f"{output_directory}/utt_pbe.txt", "w") as fout:
        print("utt_id", "phone boundary error", file=fout)
        for k, v in utt2pbe_sorted:
            fname = ref_files[k]
            if v >= 0.05:
                try:
                    words_ref, phones_ref = get_ctm_list(fname)
                except:
                    logging.warning(f"Problem reading files: {fname}")
                text = [w.label for w in words_ref]
                print(k, v, len(text), text, file=fout)
            else:
                print(k, v, file=fout)

    with open(f"{output_directory}/confusion.txt", "w") as fout:
        print("reference,hypothesis,count\n", file=fout)
        for k, v in sorted(phone_confusions.items(), key=lambda x: -x[1]):
            print(f"{k[0]},{k[1]},{v}", file=fout)
    logging.info(f"Phone boundary error: {score_sum/score_count} = {score_sum} / {score_count}")
    logging.info(f"Phone boundary error (begin): {score_sum_begin/score_count}")
    logging.info(f"Phone boundary error (end): {score_sum_end/score_count}")
    logging.info("")
    logging.info(f"Phone average duration (ref): {dur_sum_A/overlap_count_sum}")
    logging.info(f"Phone average duration (ali): {dur_sum_B/overlap_count_sum}")
    logging.info("")
    logging.info(f"Average phone error rate: {phone_edit_sum/phone_length_sum}")
    logging.info(f"Average phone overlap rate: {overlap_count_sum/phone_length_sum}")

    logging.info("")
    logging.info(f"begin_early_cnt: {pos_cnt_begin_sum}")
    logging.info(f"begin_correct_cnt: {zero_cnt_begin_sum}")
    logging.info(f"begin_late_cnt: {neg_cnt_begin_sum}")
    logging.info("")
    logging.info(f"end_early_cnt: {pos_cnt_end_sum}")
    logging.info(f"end_correct_cnt: {zero_cnt_end_sum}")
    logging.info(f"end_late_cnt: {neg_cnt_end_sum}")
    logging.info("")
    logging.info(f"overlap_count_sum: {overlap_count_sum}")

    return score_sum/score_count

def main(opts):
    # mapping_file = "/fsx/users/huangruizhe/mfa-models/scripts/alignment_benchmarks/mapping_files/mfa_buckeye_mapping.yaml"
    # mapping_file = "/fsx/users/huangruizhe/audio_ruizhe/mfa/mfa_buckeye_mapping.yaml"
    # mapping_file = "/exp/rhuang/meta/audio_ruizhe/mfa/mfa_buckeye_mapping.yaml"
    mapping_file = "/exp/rhuang/meta/audio_ruizhe/mfa/mfa_timit_mapping.yaml"
    with open(mapping_file, 'r') as fin:
        mapping = yaml.safe_load(fin)

    evaluate_alignments(
        mapping=mapping,
        output_directory=opts.out_dir,
        ali_dir=opts.ali_dir,
        ali_file_pattern=opts.ali_pattern,
        ref_dir=opts.ref_dir,
        silence_phone=opts.silence_phone,
        ignored_phones=None,
        debug=False,
        word_level=opts.word_level,
    )

def main1(opts):
    # mapping_file = "/fsx/users/huangruizhe/mfa-models/scripts/alignment_benchmarks/mapping_files/mfa_buckeye_mapping.yaml"
    mapping_file = "/fsx/users/huangruizhe/audio_ruizhe/mfa/mfa_buckeye_mapping.yaml"
    with open(mapping_file, 'r') as fin:
        mapping = yaml.safe_load(fin)

    evaluate_alignments(
        mapping=mapping,
        output_directory="/fsx/users/huangruizhe/audio_ruizhe/mfa/temptemp/",
        # ali_dir="/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/temp3/aligned_mfa/*/",  # _finetune
        # ali_dir="/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/temp4/aligned_mfa_finetune/*/",  # _finetune
        # ali_file_pattern="*.TextGrid",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_00/ali/",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_stride4_intra0.0_inter0.0/ali/",
        # ali_file_pattern="ali_epoch=37-step*.pkl",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_stride1_intra0.0_inter0.0/ali/",
        # ali_file_pattern="ali_epoch=6-step=22881*.pkl",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_stride2_intra0.8_inter0.3/ali/",
        # ali_file_pattern="ali_epoch=9-step=32687*.pkl",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_stride2_intra0.0_inter0.0_finetune1/ali/",
        # ali_file_pattern="ali_epoch=55*.pkl",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/blstm1_ttbtbtbf2_ctc_phoneme_stride2_intra0.0_inter0.0/ali/",
        # ali_file_pattern="ali_epoch=14-step*.pkl",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/conformer_4_ctc_aug_phoneme/ali/",
        # ali_file_pattern="ali_epoch=189-step*.pkl",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_20230803_12/ali/",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_20230804_blstm_5/ali/",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn_larger_context3/ali/",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_priors3_0.3_dnn_stride2_533_nospecaug/multi_task3/ali/",
        # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_priors3_0.3_dnn/_ft_bk/ali/",
        # ali_file_pattern="ali_epoch=84-*.pkl",
        ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/wav2vec_mms/ali/",
        ali_file_pattern="ali_wav2vec*.pkl",
        ref_dir="/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/temp3/",  # s03
        # ref_dir="/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/temp4/",  # s03
        silence_phone="-",
        ignored_phones=None,
        debug=False,
        word_level=opts.word_level,
    )

def main2(opts):
    # mapping_file = "/fsx/users/huangruizhe/mfa-models/scripts/alignment_benchmarks/mapping_files/mfa_buckeye_mapping.yaml"
    mapping_file = "/fsx/users/huangruizhe/audio_ruizhe/mfa/mfa_buckeye_mapping.yaml"
    with open(mapping_file, 'r') as fin:
        mapping = yaml.safe_load(fin)

    rs = []
    for i in range(26):
        pbe = evaluate_alignments(
            mapping=mapping,
            output_directory="/fsx/users/huangruizhe/audio_ruizhe/mfa/temp/",
            # ali_dir="/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/temp3/aligned_mfa/*/",
            # ali_file_pattern="*.TextGrid",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_00/ali/",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_stride4_intra0.0_inter0.0/ali/",
            # ali_file_pattern="ali_epoch=37-step*.pkl",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_stride1_intra0.0_inter0.0/ali/",
            # ali_file_pattern="ali_epoch=6-step=22881*.pkl",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_stride2_intra0.8_inter0.3/ali/",
            # ali_file_pattern="ali_epoch=9-step=32687*.pkl",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn-5_ffn4_ctc_phoneme_stride2_intra0.0_inter0.0_finetune1/ali/",
            # ali_file_pattern="ali_epoch=55*.pkl",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/blstm1_ttbtbtbf2_ctc_phoneme_stride2_intra0.0_inter0.0/ali/",
            # ali_file_pattern="ali_epoch=14-step*.pkl",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/conformer_4_ctc_aug_phoneme/ali/",
            # ali_file_pattern="ali_epoch=189-step*.pkl",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_20230803_12/ali/",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_20230804_blstm_5/ali/",
            # ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/tdnn_larger_context3/ali/",
            ali_dir="/fsx/users/huangruizhe/audio/examples/asr/librispeech_conformer_ctc/experiments/exp_note_1/ali/",
            ali_file_pattern=f"ali_epoch={i}-step=*.pkl",
            ref_dir="/fsx/users/huangruizhe/datasets/Buckeye_Corpus2/temp3/",  # s03
            silence_phone="-",
            ignored_phones=None,
            debug=False,
            word_level=opts.word_level,
        )
        rs.append(pbe)
    print(*rs, sep="\n")

if __name__ == '__main__':
    opts = parse_opts()

    main(opts)
    # main1(opts)
    # main2(opts)

# TODO:
# 1. seperate start time and end time evaluation
# 2. test other models/archs
# 3. word boundary
# 4. specaug: no time aug or no specaug at all
# 5. data augmentation
# 6. fine-tune: (1) low-res -> hi-res; (2) hmm weights; (3) buckeye
# 7. delete unused checkpoints
# 8. tdnn-blstm, conformer


