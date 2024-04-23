import k2
import logging
import torch
from k2_icefall_utils import (
    get_best_paths,
    get_texts_with_timestamp,
)
from factor_transducer import make_factor_transducer_word_level_index_with_skip
import itertools
import lis  # https://github.com/huangruizhe/lis
from dataclasses import dataclass
from typing import Union
from tqdm import tqdm


logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=logfmt, force=True)


def uniform_segmentation_with_overlap(
    m, 
    segment_size, 
    overlap, 
    shortest_segment_size=0
):
    '''
    This function cuts the input matrix `m` into overlapping segments.
    `m` can be, e.g., the feature matrix of the input audio or the emission matrix.

    Args:
        m: 3-D tensor of shape (1, T, D).
        segment_size: an integer, the size of each segment.
        overlap: an integer, the number of frames to overlap between segments.
        shortest_segment_size: an integer, the minimum size of the last segment. If the last segment is shorter than this value, it will be discarded.
    Returns:
        m_segmented: 3-D tensor of shape (L, segment_size, D), where L is the number of segments.
        segment_lengths: 1-D tensor of shape (L,). It contains the length of each segment. 
        segment_offsets: 1-D tensor of shape (L,). It contains the offset of each segment.
    '''
    assert m.ndim == 3  # (1, T, D)
    assert m.size(0) == 1

    step = segment_size - overlap
    if (m.size(1) - segment_size) % step == 0:
        n_segments = (m.size(1) - segment_size) // step + 1
        padding_size = 0
    else:
        n_segments = (m.size(1) - segment_size) // step + 2
        padding_size = (n_segments - 1) * step + segment_size - m.size(1)
    
    m_padded = torch.nn.functional.pad(m, (0, 0, 0, padding_size))  # Pad the tensor with zeros
    m_segmented = m_padded.unfold(dimension=1, size=segment_size, step=step)
    m_segmented = m_segmented.permute(0, 1, 3, 2)  # `m_segmented` is of shape (1, L, segment_size, D), where L is the number of segments.
    m_segmented = m_segmented[0]  # (L, segment_size, D)
    
    segment_lengths = [segment_size] * m_segmented.size(0)
    segment_lengths[-1] -= padding_size
    segment_lengths = torch.tensor(segment_lengths)

    segment_offsets = torch.arange(0, segment_lengths.size(0) * step, step)

    assert len(segment_lengths) == n_segments
    assert len(segment_lengths) == m_segmented.size(0)
    assert len(segment_lengths) == len(segment_offsets)

    # Discard the last chunk if it is too short
    if segment_lengths[-1] < shortest_segment_size:
        m_segmented = m_segmented[:-1]
        segment_lengths = segment_lengths[:-1]
        segment_offsets = segment_offsets[:-1]
    
    return m_segmented, segment_lengths, segment_offsets


@dataclass
class AlignedToken:
    token_id: Union[str, int]
    timestamp: int
    attr: dict

@dataclass
class AlignedWord:
    word: int
    start_time: int
    end_time: int
    phones: list

def align_segments(
    emissions,
    decoding_graph,
    segment_lengths,
):
    '''
    This function does alignment for a batch of segments using k2 library.

    Args:
        emissions: 3-D tensor of shape (L, T, C), where L is the number of segments, T is the number of frames.
        decoding_graph: k2.Fsa, the decoding graph.
        segment_lengths: 1-D tensor of shape (L,). It contains the length of each segment. 
    Returns:
        hyps: a list of lists. Each sublist contains the output labels in the best path.
        timestamps: a list of lists. Each sublist contains the timestamps corresponding to the labels.
    '''

    # Use the graph's device
    device = decoding_graph.device
    emissions = emissions.to(device)

    # Find best alignment paths using k2 library and the input WFST graph
    best_paths = get_best_paths(emissions, segment_lengths, decoding_graph)
    best_paths = best_paths.detach().to('cpu')

    # Get the output labels and timestamps from the best paths
    decoding_results = get_texts_with_timestamp(best_paths)
    hyps = decoding_results["hyps"]
    timestamps = decoding_results["timestamps"]

    # # There can be empty result in `token_ids_indices`. 
    # # We put [1,1] as a placeholder for the ease of future processing
    # token_ids_indices = [tkid if len(tkid) > 0 else [1,1] for tkid in token_ids_indices]
    # token_ids_indices = [list(map(lambda x: x - 1, rg)) for rg in token_ids_indices]

    results = []
    for hyp, timestamp in zip(hyps, timestamps):
        assert len(hyp) == len(timestamp)  # `hyp` and `timestamp` are both lists of integers
        aligned_tokens = []
        for tid, ts in zip(hyp, timestamp):
            aligned_tokens.append(AlignedToken(tid, ts, {}))
        results.append(aligned_tokens)

    return results



def align(
    waveform_or_features,
    text,
    tokenizer,
    model,
    device,
    do_segmentation=True,  # related to uniform segmentation
    sample_rate=16000,   # how many frames in `waveform_or_features` are there for each second
    segment_size=15.0,  # in seconds
    overlap=2.0,  # in seconds
    shortest_segment_size=0.2,  # in seconds
    batch_size=32,
    model_frame_duration=0.02,  # in seconds, the frame duration of the model output
):
    '''
    This function aligns long audio and noisy text in WFST framework.

    Args:
        waveform_or_features: 3-D tensor of shape (1, T, C), where T is the number of frames. For the moment, we only support one long audio.
        text: a string, the long and noisy text.
        tokenizer: the tokenizer as defined in our library.
        do_segmentation: a boolean, whether to segment the long audio into overlapping segments.
                         It's True by default. But if you have already segmented the input to short segments, 
                         you can set it to False.
        sample_rate: an integer corresponding to how many frames in `waveform_or_features` are there for each second
        segment_size: an integer, the size of each segment in seconds.
        overlap: an integer, the size of overlap between segments in seconds.
        shortest_segment_size: an integer, the minimum size in seconds of the last segment. If the last segment is shorter than this value, it will be discarded.
        batch_size: an integer, the batch size for alignment.
    '''
    assert waveform_or_features.ndim == 3
    assert waveform_or_features.size(0) == 1
    logging.info(f"The input is of shape: {waveform_or_features.shape}")

    #### Step (0): tokenize the text ####
    logging.info("Step (0): tokenize the text")
    tokenized_text = tokenizer.encode(text)
    logging.info(f"There are {len(tokenized_text)} words or {sum([len(l) for l in tokenized_text])} tokens in the text")

    #### Step (1): get the decoding graph ####
    logging.info("Step (1): get the decoding graph with `make_factor_transducer_word_level_index_with_skip`")
    decoding_graph, word_index_sym_tab, token_sym_tab = \
        make_factor_transducer_word_level_index_with_skip(
            tokenized_text,
            blank_penalty=0,  # TODO: we need to expose these arguments to the outside
            skip_penalty=-0.5,
            return_penalty=-18.0
        )
    decoding_graph = decoding_graph.to(device)
    logging.info(f"There are {decoding_graph.shape[0]} nodes and {decoding_graph.num_arcs} arcs in the decoding graph for the text of {len(tokenized_text)} words.")
    logging.info(f"The decoding graph is on device: {decoding_graph.device}")

    #### Step (2): uniform segmentation ####
    if do_segmentation:
        logging.info("Step (2): uniform segmentation")
        segments, segment_lengths, segment_offsets = uniform_segmentation_with_overlap(
            waveform_or_features,
            segment_size=int(segment_size * sample_rate),
            overlap=int(overlap * sample_rate),
            shortest_segment_size=int(shortest_segment_size * sample_rate),
        )
        segments = segments.squeeze(-1)  # Now segments is of shape (N, segment_size) for waveforms or (N, segment_size, D) for features
        logging.info(f"segment.shape={segments.shape}, segment_lengths.shape={segment_lengths.shape}, segment_offsets.shape={segment_offsets.shape}")
    else:
        logging.info("Step (2): skipping uniform segmentation")

    #### Step (3): obtain alignment for segments ####
    logging.info("Step (3): obtain alignment for segments")
    output_frames_offset = segment_offsets // (sample_rate * model_frame_duration)

    alignment_results = list()
    for i in tqdm(range(0, segments.size(0), batch_size)):
        batch_segments = segments[i: i+batch_size].to(device)
        batch_segment_lengths = segment_lengths[i: i+batch_size]
        batch_output_frames_offset = output_frames_offset[i: i+batch_size]

        with torch.inference_mode():
            # Checkout the API of the forward function here: https://github.com/pytorch/audio/blob/main/src/torchaudio/pipelines/_wav2vec2/utils.py#L34
            batch_emissions, batch_emissions_lengths = model(batch_segments.to(device), batch_segment_lengths.to(device))

        # Attach the star dimension manually, see torchaudio issue #3772
        star_dim = torch.empty((batch_emissions.size(0), batch_emissions.size(1), 1), device=batch_emissions.device, dtype=batch_emissions.dtype)
        star_dim[:] = -5.0  # TODO: this is hard-wired for the moment
        batch_emissions = torch.cat((batch_emissions, star_dim), 2)

        # `token_ids` and `timestamps` will each be a list of lists.
        # Each sublist corresponds to a segment in the batch.
        batch_results = align_segments(
            batch_emissions,
            decoding_graph,
            batch_emissions_lengths,
        )

        # The interpretation of `token.token_id` depends on the decoding graph.
        # Here, in this tutorial, `token.token_id` is the key to the `word_index_sym_tab``
        # and `token_sym_tab` dictionaries.
        for aligned_tokens, offset in zip(batch_results, batch_output_frames_offset):
            for token in aligned_tokens:
                token.timestamp += offset  # This will become the absolute frame timestamp in the whole audio
                if token.token_id == tokenizer.blk_id:
                    continue
                if token.token_id in word_index_sym_tab:
                    token.attr["wid"] = word_index_sym_tab[token.token_id]
                if token.token_id in token_sym_tab:
                    token.attr["tk"] = token_sym_tab[token.token_id]

        alignment_results.extend(batch_results)

    #### Step (4): concatenate the alignment results for segments ####
    logging.info("Step (4): concatenate the alignment results for segments")

    # `resolved_alignment_results` is a list of `AlignedToken`
    # `unaligned_text_indices` is a list of (start_word_index, end_word_index)
    #    which corresponds to "holes" in the long text that are not aligned
    resolved_alignment_results, unaligned_text_indices = concat_alignments(
        alignment_results,
        neighborhood_size=5,
    )
    logging.info(f"There are {len(resolved_alignment_results)} tokens aligned, while {len(unaligned_text_indices)} unaligned parts in the long text.")

    #### Step (5): get the final word-level alignments ####
    logging.info("Step (5): get the final word-level alignments")

    # `word_alignment` is a dict of word index in the long text => AlignedWord object
    word_alignment = get_final_word_alignment(resolved_alignment_results, text, tokenizer)
    logging.info(f"There are {len(word_alignment)} words aligned in the long text.")

    return word_alignment, unaligned_text_indices
    
    


# def align_one_batch(
#     batch_features: torch.Tensor,
#     y_long_list,
#     segment_lengths,
#     params: AttributeDict,
#     model: nn.Module,
#     sp: Optional[spm.SentencePieceProcessor],
#     is_standard_forced_alignment: bool = False,
# ):  # -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    
    
#     return token_ids_indices, timestamps


def get_linear_fst(word_ids_list, blank_id=0, max_symbol_id=1000, is_left=True, return_str=False):
    graph = k2.linear_fst(labels=[word_ids_list], aux_labels=[word_ids_list])[0]

    max_symbol_id += 1

    c_str = k2.to_str_simple(graph)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])

    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]

    max_symbol_id = max(max_symbol_id, len(arcs) + 10)

    new_arcs = []
    # left: ground-truth (from arange)
    # right: hypothesis (from alignment)
    for i, (ss, ee, l1, l2, w) in enumerate(arcs):
        if l1 > 0:
            # substitution
            if is_left:
                new_arcs.append([ss, ee, l1, max_symbol_id, -1])
            else:
                new_arcs.append([ss, ee, max_symbol_id, i+1, -1])
            
            # deletion
            new_arcs.append([ss, ee, l1, blank_id, -2])
        
        # insertion
        if is_left:
            new_arcs.append([ss, ss, blank_id, max_symbol_id, -2])
        else:
            new_arcs.append([ss, ss, max_symbol_id, blank_id, -2])
    
    if not is_left:
        arcs = [(ss, ee, l1, i+1 if l2 >= 0 else l2, w) for i, (ss, ee, l1, l2, w) in enumerate(arcs)]

    new_arcs = arcs + new_arcs
    new_arcs.append([final_state])

    new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
    new_arcs = [[str(i) for i in arc] for arc in new_arcs]
    new_arcs = [" ".join(arc) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        return fst


def get_range_without_outliers(my_list, scan_range=100, outlier_threshold=60):
    # remove outliers
    # given a list of integers in my_list in ascending order, find the range without outliers
    # outliers: a number that is outlier_threshold smaller/larger than its neighbors

    # my_list = [150] + list(range(200,1000)) + [1105]
    # print(my_list)
    # get_range_without_outliers(my_list)

    if len(my_list) <= 10:
        return my_list[0], my_list[-1]

    scan_range = min(scan_range, int(len(my_list)/2) - 1)
    
    left = [i+1 for i in range(0, scan_range) if my_list[i+1] - my_list[i] > outlier_threshold]
    right = [i-1 for i in range(-scan_range, 0) if my_list[i] - my_list[i-1] > outlier_threshold]
    left = left[-1] if len(left) > 0 else 0
    right = right[0] if len(right) > 0 else -1
    left = my_list[left]
    right = my_list[right]
    return left, right


def remove_outliers(my_list, scan_range=100, outlier_threshold=60):
    # Given a list of integers in my_list in ascending order, remove outliers
    # such that there is a gap more than outlier_threshold

    if len(my_list) <= 10:
        return my_list
    
    scan_range = min(scan_range, int(len(my_list)/2) - 1)
    left = [i+1 for i in range(0, scan_range) if my_list[i+1] - my_list[i] > outlier_threshold]
    right = [i-1 for i in range(-scan_range, 0) if my_list[i] - my_list[i-1] > outlier_threshold]
    left = left[-1] if len(left) > 0 else 0
    right = right[0]+1 if len(right) > 0 else None

    return my_list[left: right]


def get_lis_alignment(lis_results, alignment_results):
    '''
    This function aligns the longest increasing subsequence (LIS) in `lis_results` to the original `alignment_results`.
    `lis_results` must be a sublist of `alignment_results`.
    We use a heuristic here to ensure some "tightness" of the alignment.
    More specifically, for the first half of `lis_results`, we align them to the last occurrence in `alignment_results`;
    for the last half of `lis_results`, we align them to the first occurrence in `alignment_results`.
    '''    
    midpoint = len(lis_results) // 2

    lis_ali = dict()
    j = 0  # j: index in lis_results
    outer_break = False
    for aligned_tokens in alignment_results:
        for token in aligned_tokens:
            if j < midpoint:
                if lis_results[j] == token.attr.get("wid", None):
                    lis_ali[lis_results[j]] = token
                    j += 1
                    if j == len(lis_results):
                        outer_break = True
                        break
            else:
                if lis_results[j] == token.attr.get("wid", None) and lis_results[j] not in lis_ali:
                    lis_ali[lis_results[j]] = token
                    j += 1
                    if j == len(lis_results):
                        outer_break = True
                        break
        if outer_break:
            break

    for token in lis_ali.values():
        token.attr["lis"] = True

    return alignment_results


class WordCounter: 
    def __init__(self, val=0): 
        self.counter1 = val
    def f1(self): 
        self.counter1 += 1; return self.counter1


def reduce_long_list1(ids_list1, ids_list2):
    # break it into parts and compute overlap

    T = 2000
    start = 0
    end = None

    for i in range(0, len(ids_list1), T):
        ids_list1_tmp = ids_list1[i: i+T]
        nonoverlap = set(ids_list1_tmp) - set(ids_list2)
        if len(nonoverlap) / len(ids_list1_tmp) < 0.3:
            start = max(0, i-1)
            break
    
    for i in reversed(range(0, len(ids_list1), T)):
        ids_list1_tmp = ids_list1[i: i+T]
        if len(ids_list1_tmp) < 0.8 * T:
            continue

        nonoverlap = set(ids_list1_tmp) - set(ids_list2)
        if len(nonoverlap) / len(ids_list1_tmp) < 0.3:
            end = i + 1 + 1
            break
    
    return ids_list1[start: end]



def get_aligned_list(hyps_list, my_hyps_min=None, my_hyps_max=None, device='cpu'):
    my_hyps_ids = sorted([w for hyp in hyps_list for w in hyp])

    _my_hyps_min, _my_hyps_max = get_range_without_outliers(my_hyps_ids, scan_range=100, outlier_threshold=60)
    my_hyps_min = _my_hyps_min if my_hyps_min is None else max(my_hyps_min - 10, 0)
    my_hyps_max = _my_hyps_max if my_hyps_max is None else my_hyps_max + 10
        
    max_symbol_id = max(my_hyps_ids) + 100   # use this symbol to mark the chunk boundaries
    ids_list1 = list(range(my_hyps_min, my_hyps_max + 1))
    ids_list2 = [item for sublist in hyps_list for item in ([max_symbol_id] + sublist)]
    ids_list2 = ids_list2[1:]

    list1_len_thres = 40000
    if len(ids_list1) > list1_len_thres:
        logging.warning(f"Long ids_list1 ({len(ids_list1)}) in range {my_hyps_min} and {my_hyps_max} vs. list2 ({len(ids_list2)})")
        ids_list1 = reduce_long_list1(ids_list1, ids_list2)
        logging.warning(f"Reduced to ({len(ids_list1)})")

    graph1 = get_linear_fst(ids_list1, max_symbol_id=max_symbol_id+1, blank_id=0, is_left=True, return_str=False)
    graph2 = get_linear_fst(ids_list2, max_symbol_id=max_symbol_id+1, blank_id=0, is_left=False, return_str=False)

    # graph1 = graph1.to(device)
    # graph2 = graph2.to(device)

    rs = k2.compose(graph1, graph2, treat_epsilons_specially=True)
    rs = k2.connect(rs)
    rs = k2.remove_epsilon_self_loops(rs)
    # rs = k2.arc_sort(rs)
    rs = k2.top_sort(rs)  # top-sort is needed for shortest path: https://github.com/k2-fsa/k2/issues/746#issuecomment-856503238
    # print("Composed graph size: ", rs.shape, rs.num_arcs)

    rs_vec = k2.create_fsa_vec([rs])
    best_paths = k2.shortest_path(rs_vec, use_double_scores=True)
    # best_paths.shape, best_paths.num_arcs

    best_paths = k2.top_sort(best_paths)
    best_paths = k2.arc_sort(best_paths)

    rs_list1 = best_paths[0].labels.tolist()
    rs_list2 = best_paths[0].aux_labels.tolist()
    rs_list2_ = [ids_list2[i-1] if i > 0 else None for i in rs_list2]
    for l1, l2 in zip(rs_list1, rs_list2_):
        if l1 == l2:
            break
    rs_my_hyps_min = l1
    
    for l1, l2 in zip(reversed(rs_list1), reversed(rs_list2_)):
        if l1 == l2 and l1 > 0:
            break
    rs_my_hyps_max = l1

    # Adjust rs_list2 (which are the indices in ids_list2) for each chunk
    # We need to convert it to the indices in the chunk
    word_counter = WordCounter(-1)
    i_mapping = {word_counter.f1() : i - 1 for sublist in hyps_list for i, item in enumerate([max_symbol_id] + sublist)}
    rs_list2 = [i_mapping[i] if i in i_mapping else None for i in rs_list2]

    # print("Best path range: ", rs_my_hyps_min, rs_my_hyps_max)

    return best_paths, rs_list1, rs_list2, rs_list2_, max_symbol_id, rs_my_hyps_min, rs_my_hyps_max


def handle_failed_groups(no_need_to_realign, alignment_results):
    # Just simply and linearly makeup the timestamp (frame index in the output frames)

    if len(no_need_to_realign) > 0:
        for group in no_need_to_realign:
            ss = group[0]   # this is aligned left end point
            ee = group[-1]  # this is aligned right end point
            num_gaps = ee - ss
            tt1 = alignment_results[ss]
            tt2 = alignment_results[ee]
            for j, i in enumerate(range(ss + 1, ee)):
                alignment_results[i] = int((tt2 - tt1) / num_gaps * (j + 1) + tt1)
    return

# TODO: we may need to do a two pass alignment
def concat_alignments(alignment_results, neighborhood_size=5):
    # The task here is to find the "reliable" aligned parts from the alignment results `alignment_results`
    # Since the alignment results are actually "indices" in the long text, we hope to find the
    # longest increasing subsubsequence (LIS) from the alignment results.

    # solution1: just wfst(k2) to compute edit distance (shortest path from the pruned graph)
    # solution2: just use python's difflib: https://docs.python.org/3/library/difflib.html#differ-example
    #            - https://github.com/lowerquality/gentle/blob/master/gentle/diff_align.py
    # solution3: https://unix.stackexchange.com/questions/2150/diffing-two-big-text-files
    # solution4: Longest Increasing Subsequence (LIS)
    #            - https://en.wikipedia.org/wiki/Longest_increasing_subsequence
    #            - https://www.reddit.com/r/algorithms/comments/c5rerp/given_a_list_of_unsorted_integers_find_the/?onetap_auto=true
    #            - https://leetcode.com/problems/longest-increasing-subsequence/description/
    #            - https://algo.monster/liteproblems/673
    #            - https://python.plainenglish.io/longest-increasing-subsequence-python-e75d028cef7a
    #            - https://www.youtube.com/watch?v=66w10xKzbRM&t=0s
    # solution5: vimdiff: vimdiff <(tr ' ' '\n' <download/LibriSpeechAligned/LibriSpeech/books/ascii/2981/2981.txt) <(tr ' ' '\n' <download/LibriSpeechAligned/LibriSpeech/books/ascii/3600/3600.txt)

    hyps = [[token.attr["wid"] for token in aligned_tokens if "wid" in token.attr] for aligned_tokens in alignment_results]
    timestamps = [[token.timestamp for token in aligned_tokens if "wid" in token.attr] for aligned_tokens in alignment_results]

    # Find the longest increasing subsequence (LIS)
    hyp_list = [i for hyp in hyps for i in hyp]
    lis_results = lis.longestIncreasingSubsequence(hyp_list)
    
    # Post-process1: remove outliers from the LIS results
    lis_results = remove_outliers(lis_results, scan_range=100, outlier_threshold=60)
    if len(lis_results) == 0:
        return dict(), None
    
    # Post-process2: remove isolatedly aligned words
    # Each aligned word should have a neighborhood of at least `neighbor_threshold`` words
    neighbor_threshold = 0.4
    rg_min = lis_results[0]
    assert rg_min == min(lis_results)
    rg_max = lis_results[-1]
    assert rg_max == max(lis_results)
    set_lis_results = set(lis_results)
    for i in range(rg_min, rg_max + 1):
        if i in set_lis_results:
            left_neighbors_in_lis = [j for j in range(i-neighborhood_size, i) if j in set_lis_results]
            right_neighbors_in_lis = [j for j in range(i+1, i+neighborhood_size+1) if j in set_lis_results]
            num_left_neighbors = i - max(i-neighborhood_size, rg_min)
            num_right_neighbors = min(i+neighborhood_size, rg_max) - i
            # only less than 50% of the words in the neighborhood are aligned
            if len(left_neighbors_in_lis) < neighbor_threshold * num_left_neighbors and \
                len(right_neighbors_in_lis) < neighbor_threshold * num_right_neighbors:
                set_lis_results.remove(i)
    lis_results = [i for i in lis_results if i in set_lis_results]

    # Align LIS results with the original `alignment_results`
    alignment_results = get_lis_alignment(lis_results, alignment_results)  # hyp_list and lis_result are both word indices in the long text

    # Keep only the aligned tokens which are in LIS
    resolved_alignment_results = list()
    for aligned_tokens in alignment_results:
        word_start_flag = False
        for token in aligned_tokens:
            if token.attr.get("lis", False):
                resolved_alignment_results.append(token)
                word_start_flag = True
            elif "wid" in token.attr:
                # assert "lis" not in token.attr
                word_start_flag = False
            elif word_start_flag:
                resolved_alignment_results.append(token)
    
    # Find the un-aligned transcript
    # The `unaligned_text_indices` is a list of tuples (s, e) where
    # the words starting from s and ending at e (inclusive) in the original transcript do not appear in the alignment results.
    unaligned_text_indices = find_unaligned_text(rg_min, rg_max, set_lis_results)

    # # For some unaligned parts, we don't need to realign them cos they are too short
    # handle_failed_groups(no_need_to_realign, alignment_results)

    return resolved_alignment_results, unaligned_text_indices


def merge_segments(segments, threshold, is_sorted=True):
    merged = []
    if not is_sorted:
        segments = sorted(segments, key=lambda x: x[0]) 
    for start, end in segments:
        if merged and start - merged[-1][1] <= threshold:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def get_dur_group(group, rg_min, alignment_results):
    ss = rg_min + group[0] - 1
    ee = rg_min + group[-1] + 1
    return alignment_results[ee] - alignment_results[ss]


def get_neighbor_aligned_word(idx, alignment_results, neighbor_range):
    if neighbor_range > 0:
        step = -1
    else:
        step = 1

    idx += neighbor_range
    while idx != 0:
        if idx in alignment_results:
            return idx
        idx += step
    
    return idx


def find_unaligned_text(rg_min, rg_max, set_lis_results):
    '''
    The "unaligned" parts are defined as the "holes" of alignment results in the original transcript.
    E.g., (s1, e2) is a hole if the words starting from s1 and ending at e1 in the original transcript do not appear in the alignment results.
    Note, the holes are not necessarily "align-able", as it may not be spoken in the audio.
    '''

    # Find the indices of all consecutive "False" segments
    holes = [[rg_min+i for i, _ in group] for key, group in itertools.groupby(enumerate(list(range(rg_min, rg_max+1))), key=lambda x: x[1] in set_lis_results) if not key]

    if False:
        # Too few words, e.g., 2 words
        no_need_to_realign1 = [group for group in to_realign if len(group) <= no_need_to_realign_thres1]
        no_need_to_realign1 = [(rg_min + group[0] - 1, rg_min + group[-1] + 1) for group in no_need_to_realign1]
        to_realign = [group for group in to_realign if len(group) > no_need_to_realign_thres1]

        # Too short time, e.g., 10 frames in the final output is 0.4 sec
        no_need_to_realign2 = [group for group in to_realign if get_dur_group(group, rg_min, alignment_results) <= no_need_to_realign_thres2]
        no_need_to_realign2 = [(rg_min + group[0] - 1, rg_min + group[-1] + 1) for group in no_need_to_realign2]
        to_realign = [group for group in to_realign if get_dur_group(group, rg_min, alignment_results) > no_need_to_realign_thres2]

        no_need_to_realign = no_need_to_realign1 + no_need_to_realign2
        no_need_to_realign = sorted(no_need_to_realign, key=lambda x: x[0])
    else:
        # Ok, just add more padding to the segments
        no_need_to_realign = []
        pass

    # Ok, these are the unaligned holes in the transcript, which may need to be realigned
    holes = [(group[0], group[-1]) for group in holes]

    # Merge the unaligned segments if they are close to each other
    holes = merge_segments(holes, threshold=3, is_sorted=True)
    
    # neighbor_range = 3  # add 3 already aligned words to the left and right
    # to_realign = [
    #     (
    #         get_neighbor_aligned_word(rg_min + group[0], alignment_results, -neighbor_range),
    #         get_neighbor_aligned_word(rg_min + group[-1] + 1, alignment_results, neighbor_range)
    #     ) for group in to_realign
    # ]  # each start/end will be an aligned word

    return holes


def get_final_word_alignment(alignment_results, text, tokenizer):
    text_splitted = text.split()

    # A dictionary from: word_index => AlignedWord object
    word_alignment = dict()
    aligned_word = None
    word_idx = None
    assert "wid" in alignment_results[0].attr
    for aligned_token in alignment_results:
        if "wid" in aligned_token.attr:
            if aligned_word is not None:
                word_alignment[word_idx] = aligned_word
            word_idx = aligned_token.attr['wid']
            aligned_word = AlignedWord(
                word=text_splitted[word_idx],
                start_time=aligned_token.timestamp,
                end_time=None,
                phones=[],
            )
        # TODO: 
        # Ideally, we should have a start_time and end_time for each token,
        # just as in the HMM-GMM model.
        aligned_word.phones.append(
            AlignedToken(
                token_id=tokenizer.id2token[aligned_token.attr['tk']],
                timestamp=aligned_token.timestamp,
                attr=None
            )
        )
    return word_alignment


def get_audacity_labels(word_alignment, frame_duration):
    # To audacity: https://manual.audacityteam.org/man/importing_and_exporting_labels.html
    
    alignment_results_ = [(w.word, w.start_time*frame_duration) for _, w in sorted(word_alignment.items())] 
    audacity_labels_str = "\n".join([f"{t:.2f}\t{t:.2f}\t{label}" for label, t in alignment_results_])
    return audacity_labels_str


def get_gentle_visualization(word_alignment, tokenizer, frame_duration, audio_file, text, i_word_start=None, i_word_end=None):
    # align_to_gentle:
    # /exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc_large/alignment/ali_torchaudio.py

    text_splitted = text.split()
    if i_word_start is None:
        i_word_start = 0
    if i_word_end is None:
        i_word_end = len(text_splitted)

    # Get the json file of alignment results
    transcript = " ".join(text_splitted[i_word_start: i_word_end])
    words = []
    char_offet = 0
    for wid in range(i_word_start, i_word_end):
        if wid in word_alignment:
            w = word_alignment[wid]
            
            if len(words) > 0:  # Use the start time of this word as the end time of the previous word
                words[-1]["end"] = f"{w.start_time * frame_duration:.2f}"
            
            words.append({
                "word": w.word,
                "alignedWord": tokenizer.text_normalize(w.word),
                "case": "success",
                "start": f"{w.start_time * frame_duration:.2f}",
                "startOffset": char_offet,
                "end": f"{w.start_time * frame_duration + 0.1:.2f}",
                "endOffset": char_offet + len(w.word) + 1,
                "phones": [
                    {
                        "duration": f"{frame_duration:.2f}",
                        "phone": p.token_id,
                    } for p in w.phones
                ],
            })
        else:
            w = text_splitted[wid]
            w = AlignedWord(
                word=w,
                start_time=None,
                end_time=None,
                phones=[],
            )
            words.append({
                "word": w.word,
                "case": "not-found-in-audio",
                "startOffset": char_offet,
                "endOffset": char_offet + len(w.word) + 1,
            })
        
        char_offet += (len(w.word) + 1)
    
    inline_json = {
        "transcript": transcript,
        "words": words,
    }

    import os
    from pathlib import Path
    import json

    current_file_path = Path(os.path.abspath(__file__))
    html_path = current_file_path.parent / "gentle" / "view_alignment.html"
    audio_path = Path(audio_file).resolve()

    htmltxt = open(html_path).read()
    htmltxt = htmltxt.replace("var INLINE_JSON;", "var INLINE_JSON=%s;" % (json.dumps(inline_json)))
    htmltxt = htmltxt.replace('src="a.wav"', f'src="{audio_path.name}"')
    
    open(audio_path.parent / 'index.html', 'w').write(htmltxt)
    print(
        f"Gentle visualization is saved to: {str(audio_path.parent / 'index.html')}\n" + \
        f"Usage instrunctions:\n" + \
        f"- Download both the audio file and the index.html file to the same folder in your local machine;\n" + \
        f"- Open the index.html file in a browser."
    )
