import k2
import logging
import torch
from k2_icefall_utils import (
    get_best_paths,
    get_texts_with_timestamp,
)


logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=logfmt)


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
    '''

    # Use the graph's device
    device = decoding_graph.device
    emissions = emissions.to(device)

    # find best alignment paths using k2 library and the input WFST graph
    best_paths = get_best_paths(emissions, segment_lengths, decoding_graph)
    best_paths = best_paths.detach().to('cpu')

    decoding_results = get_texts_with_timestamp(best_paths)
    token_ids_indices = decoding_results["hyps"]  # Note, here the "hyps" are actually indices 
    timestamps = decoding_results["timestamps"]

    # There can be empty result in `token_ids_indices`. 
    # We put [1,1] as a placeholder for the ease of future processing
    token_ids_indices = [tkid if len(tkid) > 0 else [1,1] for tkid in token_ids_indices]
    token_ids_indices = [list(map(lambda x: x - 1, rg)) for rg in token_ids_indices]

    return token_ids_indices, timestamps



def align(
    emissions,
    graph,
    segment_size,
    overlap,
    shortest_segment_size=0,
    batch_size=32,
    do_segmentation=True,
):
    '''
    This function does alignment for long audio features using k2 library.

    Args:
        emissions: 3-D tensor of shape (1, T, C), where T is the number of frames.
        graph: k2.Fsa, the decoding graph.
        segment_size: an integer, the size of each segment.
        overlap: an integer, the number of frames to overlap between segments.
        shortest_segment_size: an integer, the minimum size of the last segment. If the last segment is shorter than this value, it will be discarded.
        batch_size: an integer, the batch size for alignment.
    '''

    # Step (1): cut the long audio features into overlapping segments
    if do_segmentation:
        emissions_segmented, segment_lengths, segment_offsets = \
            uniform_segmentation_with_overlap(emissions, segment_size, overlap, shortest_segment_size=shortest_segment_size)
    else:
        assert emissions.size(0) == 1
        emissions_segmented = emissions
        segment_lengths = torch.tensor([emissions.size(1)])
        segment_offsets = torch.tensor([0])
    
    # Use the graph's device
    device = graph.device

    # Step (2): do alignment for batches. 
    # Hopefully, each batch can fit into the GPU memory.
    results_hyps = list()
    results_timestamps = list()
    for i in range(0, emissions_segmented.size(0), batch_size):
        batch_emissions = emissions_segmented[i: i+batch_size].to(device)
        batch_segment_lengths = segment_lengths[i: i+batch_size]

        # find best alignment paths using k2 library and the input WFST graph
        best_paths = get_best_paths(batch_emissions, batch_segment_lengths, decoding_graph)
        best_paths = best_paths.detach().to('cpu')

        decoding_results = get_texts_with_timestamp(best_paths)
        token_ids_indices = decoding_results["hyps"]  # Note, here the "hyps" are actually indices 
        timestamps = decoding_results["timestamps"]

        # There can be empty result in `token_ids_indices`. 
        # We put [1,1] as a placeholder for the ease of future processing
        token_ids_indices = [tkid if len(tkid) > 0 else [1,1] for tkid in token_ids_indices]
        token_ids_indices = [list(map(lambda x: x - 1, rg)) for rg in token_ids_indices]

        results_hyps.extend(token_ids_indices)
        results_timestamps.extend(timestamps)
    
    


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