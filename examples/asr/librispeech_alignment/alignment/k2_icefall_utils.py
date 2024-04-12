import logging
import random
import torch
import k2

from collections import defaultdict


# https://github.com/k2-fsa/icefall/blob/master/icefall/decode.py#L94
def _get_lattice(
    nnet_output: torch.Tensor,
    decoding_graph: k2.Fsa,
    supervision_segments: torch.Tensor,
    search_beam: float,
    output_beam: float,
    min_active_states: int,
    max_active_states: int,
    subsampling_factor: int = 1,
) -> k2.Fsa:
    """Get the decoding lattice from a decoding graph and neural
    network output.
    Args:
      nnet_output:
        It is the output of a neural model of shape `(N, T, C)`.
      decoding_graph:
        An Fsa, the decoding graph. It can be either an HLG
        (see `compile_HLG.py`) or an H (see `k2.ctc_topo`).
      supervision_segments:
        A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
        Each row contains information for a supervision segment. Column 0
        is the `sequence_index` indicating which sequence this segment
        comes from; column 1 specifies the `start_frame` of this segment
        within the sequence; column 2 contains the `duration` of this
        segment.
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      subsampling_factor:
        The subsampling factor of the model.
    Returns:
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=subsampling_factor - 1,
    )

    lattice = k2.intersect_dense_pruned(
        decoding_graph,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice


def get_lattice(
    ctc_output, 
    target_lengths, 
    decoding_graph,
    search_beam=20,
    output_beam=8,
    min_active_states=30,
    max_active_states=10000,
    subsampling_factor=1,
):
    # ctc_output: (N, T, C)
    batch_size = ctc_output.size(0)
    supervision_segments = torch.stack(
        (
            torch.arange(batch_size),
            torch.zeros(batch_size),
            target_lengths.cpu(),
        ),
        1,
    ).to(torch.int32)

    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    if isinstance(decoding_graph, list):
        if len(decoding_graph) > 1:
            decoding_graph = [decoding_graph[i] for i in indices.tolist()]
            decoding_graph = k2.create_fsa_vec(decoding_graph)
        else:
            decoding_graph = decoding_graph[0]
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = decoding_graph.to(ctc_output.device)

    lattice = _get_lattice(
        nnet_output=ctc_output,
        decoding_graph=decoding_graph,
        supervision_segments=supervision_segments,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
        subsampling_factor=subsampling_factor,
    )
    
    return lattice, indices


def get_lattice_and_best_paths(ctc_output, target_lengths, decoding_graph):
    lattice, indices = get_lattice(ctc_output, target_lengths, decoding_graph)

    best_paths = k2.shortest_path(lattice, use_double_scores=True)

    _indices = {i_old : i_new for i_new, i_old in enumerate(indices.tolist())}
    best_paths = [best_paths[_indices[i]] for i in range(len(_indices))]
    best_paths = k2.create_fsa_vec(best_paths)

    lattice = [lattice[_indices[i]] for i in range(len(_indices))]
    lattice = k2.create_fsa_vec(lattice)
    
    # This `lattice` and `best_paths` are in the same order as the original batch
    return lattice, best_paths


def get_best_paths(ctc_output, target_lengths, decoding_graph):
    lattice, best_paths = get_lattice_and_best_paths(ctc_output, target_lengths, decoding_graph)
    return best_paths


# https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py#L359
def get_texts_with_timestamp(
    best_paths: k2.Fsa, return_ragged: bool = False
):
    """Extract the texts (as word IDs) and timestamps (as frame indexes)
    from the best-path FSAs.
    Args:
      best_paths:
        A k2.Fsa with best_paths.arcs.num_axes() == 3, i.e.
        containing multiple FSAs, which is expected to be the result
        of k2.shortest_path (otherwise the returned values won't
        be meaningful).
      return_ragged:
        True to return a ragged tensor with two axes [utt][word_id].
        False to return a list-of-list word IDs.
    Returns:
      Returns a list of lists of int, containing the label sequences we
      decoded.
    """
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        all_aux_shape = (
            best_paths.arcs.shape().remove_axis(1).compose(best_paths.aux_labels.shape)
        )
        all_aux_labels = k2.RaggedTensor(all_aux_shape, best_paths.aux_labels.values)
        # remove 0's and -1's.
        aux_labels = best_paths.aux_labels.remove_values_leq(0)
        # TODO: change arcs.shape() to arcs.shape
        aux_shape = best_paths.arcs.shape().compose(aux_labels.shape)
        # remove the states and arcs axes.
        aux_shape = aux_shape.remove_axis(1)
        aux_shape = aux_shape.remove_axis(1)
        aux_labels = k2.RaggedTensor(aux_shape, aux_labels.values)
    else:
        # remove axis corresponding to states.
        aux_shape = best_paths.arcs.shape().remove_axis(1)
        all_aux_labels = k2.RaggedTensor(aux_shape, best_paths.aux_labels)
        # remove 0's and -1's.
        aux_labels = all_aux_labels.remove_values_leq(0)

    assert aux_labels.num_axes == 2

    timestamps = []
    if isinstance(best_paths.aux_labels, k2.RaggedTensor):
        for p in range(all_aux_labels.dim0):
            time = []
            for i, arc in enumerate(all_aux_labels[p].tolist()):
                if len(arc) == 1 and arc[0] > 0:
                    time.append(i)
            timestamps.append(time)
    else:
        for labels in all_aux_labels.tolist():
            time = [i for i, v in enumerate(labels) if v > 0]
            timestamps.append(time)

    return {
        "timestamps": timestamps,
        "hyps": aux_labels if return_ragged else aux_labels.tolist(),
    }

