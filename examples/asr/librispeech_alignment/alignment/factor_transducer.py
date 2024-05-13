from collections import defaultdict
import k2


# def convert_long_text_to_fst(items, sp, pid, results):
#     libri_long_text_sp = dict()
#     for k, text in tqdm(items, mininterval=2, desc=f"libri_long_text [{pid}]"):
#         # libri_long_text_sp[k] = make_factor_transducer1(sp.encode(text, out_type=int), return_str=True, blank_penalty=0)
#         # libri_long_text_sp[k] = make_factor_transducer2(sp.encode(text, out_type=int), return_str=True, blank_penalty=-12)
#         # libri_long_text_sp[k] = make_factor_transducer3(sp.encode(text, out_type=int), word_start_symbols={i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}, return_str=True, blank_penalty=0)
#         # libri_long_text_sp[k] = make_factor_transducer4(sp.encode(text, out_type=int), word_start_symbols={i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}, return_str=True, blank_penalty=0)
#         libri_long_text_sp[k] = make_factor_transducer4_skip(sp.encode(text, out_type=int), word_start_symbols={i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}, return_str=True, blank_penalty=0, skip_penalty=-0.5, return_penalty=None)
#     results[pid] = libri_long_text_sp
#     return libri_long_text_sp


def flatten_list(lst):
    # return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]
    return [item for w in lst for item in (w[0] if isinstance(w[0], list) else w)]


def make_ctc_graph(word_id_list, return_str=False):
    '''
        This makes ctc graph same as k2.ctc_graph. We just demonstrate how to draw a ctc graph.
        `word_id_list` is the tokenization for a sentence.
        It is of the following format:
        - [[75], [47], [7], [629, 218]]
        - or [[[75]], [[47]], [[7]], [[629, 218], [123, 456]]] in case a word has multiple pronunciations
    '''

    word_id_list_flattened = flatten_list(word_id_list)

    arcs = []
    start_state = 0
    final_state = 2*len(word_id_list_flattened) + 1

    eps = 0
    # arc: (ss, ee, l1, l2, weight)
    arcs.append((start_state, start_state, eps, eps, 0))
    
    prev_blk_state = start_state
    prev_non_blk_state = None
    prev_p = None
    token_cnt = 0
    
    for w in word_id_list:
        # `w` is the list of tokens for this word
        assert len(w) > 0, f"len(w)={len(w)}, w={w}, word_id_list={word_id_list}"

        if isinstance(w[0], list):
            # TODO: for the moment, we only consider `word_id_list` as a list of integers.
            # That is, each word has only one tokenization
            # Otherwise, if the word has multiple tokenizations, we take only the first one
            w = w[0]
        
        # For each word/token, we do the following:
        # 1) link the graph of this word/token to the existing graph
        # 2) draw the graph of this word/token
        
        for p in w:
            # `p` is the token
            cur_non_blk_state = 2*token_cnt + 1
            cur_blk_state = cur_non_blk_state + 1

            # 1) link it to the existing graph
            if prev_non_blk_state is not None:
                if p != prev_p:
                    arcs.append((prev_non_blk_state, cur_non_blk_state, p, p, 0))

            if prev_blk_state is not None:
                arcs.append((prev_blk_state, cur_non_blk_state, p, p, 0))

            # 2) draw the graph of this token
            arcs.append((cur_non_blk_state, cur_non_blk_state, p, eps, 0))
            arcs.append((cur_non_blk_state, cur_blk_state, eps, eps, 0))
            arcs.append((cur_blk_state, cur_blk_state, eps, eps, 0))

            prev_non_blk_state = cur_non_blk_state
            prev_blk_state = cur_blk_state
            prev_p = p
            
            token_cnt += 1

    assert prev_non_blk_state == final_state - 2, f"cur_state={prev_non_blk_state}, final_state={final_state}"
    arcs.append((prev_non_blk_state, final_state, -1, -1, 0))
    arcs.append((prev_blk_state, final_state, -1, -1, 0))
    arcs.append((final_state,))

    new_arcs = sorted(arcs, key=lambda arc: arc[0])
    new_arcs = [" ".join(map(str, arc)) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        return fst


def make_factor_transducer_basic(word_id_list, return_str=False, blank_penalty=0):
    '''
        This is the original, simplest factor transducer for a "linear" fst.
        We can enter at *any non-eps state*, and we can exit at *any state*.
        The blank symbols at the beginning and ending of the graph can be penalized.

        `blank_penalty` should be negative (e.g., blank_penalty=-1.0):
        -- this is because in k2, the weights on the arcs are interpreted as "log-probs"
        -- we prefer the arc weights the larger the better
    '''

    word_id_list_flattened = flatten_list(word_id_list)

    arcs = []
    start_state = 0
    final_blk_state = 2*len(word_id_list_flattened) + 1
    final_state = final_blk_state + 1

    eps = 0
    # arc: (ss, ee, l1, l2, weight)
    arcs.append((start_state, start_state, eps, eps, blank_penalty))
    
    prev_blk_state = start_state
    prev_non_blk_state = None
    prev_p = None
    token_cnt = 0

    # TODO:
    # 1. We may better model disfluencies ['uh', 'um'] as in: https://github.com/lowerquality/gentle/blob/master/align.py#L39C1-L39C33
    
    for w in word_id_list:
        # `w` is the list of tokens for this word
        assert len(w) > 0, f"len(w)={len(w)}, w={w}, word_id_list={word_id_list}"

        if isinstance(w[0], list):
            # TODO: for the moment, we only consider `word_id_list` as a list of integers.
            # That is, each word has only one tokenization
            # Otherwise, if the word has multiple tokenizations, we take only the first one
            w = w[0]
        
        # For each word/token, we do the following:
        # 1) link the graph of this word/token to the existing graph
        # 2) draw the graph of this word/token
        # 3) add factor arcs
        
        for p in w:
            # `p` is the token
            cur_non_blk_state = 2*token_cnt + 1
            cur_blk_state = cur_non_blk_state + 1

            # 1) link it to the existing graph
            if prev_non_blk_state is not None:
                if p != prev_p:
                    arcs.append((prev_non_blk_state, cur_non_blk_state, p, p, 0))

            if prev_blk_state is not None and prev_blk_state > 0:
                arcs.append((prev_blk_state, cur_non_blk_state, p, p, 0))

            # 2) draw the graph of this token
            arcs.append((cur_non_blk_state, cur_non_blk_state, p, eps, 0))
            arcs.append((cur_non_blk_state, cur_blk_state, eps, eps, 0))
            arcs.append((cur_blk_state, cur_blk_state, eps, eps, 0))

            # 3) add factor arcs
            arcs.append((start_state, cur_non_blk_state, p, p, 0))  # in-coming factor arc
            arcs.append((cur_non_blk_state, final_state, -1, -1, 0))  # out-going factor arc 1
            arcs.append((cur_non_blk_state, final_blk_state, eps, eps, 0))  # out-going factor arc 2

            prev_non_blk_state = cur_non_blk_state
            prev_blk_state = cur_blk_state
            prev_p = p
            
            token_cnt += 1

    assert prev_non_blk_state == final_state - 3, f"cur_state={prev_non_blk_state}, final_state={final_state}"
    arcs.append((final_blk_state, final_blk_state, eps, eps, blank_penalty))
    arcs.append((final_blk_state, final_state, -1, -1, 0))
    arcs.append((prev_blk_state, final_state, -1, -1, 0))
    arcs.append((final_state,))

    new_arcs = sorted(arcs, key=lambda arc: arc[0])
    new_arcs = [" ".join(map(str, arc)) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        return fst


def make_factor_transducer_word_level(word_id_list, return_str=False, blank_penalty=0):
    '''
        This is a modification of `make_factor_transducer_basic`, where the factors are on "word-level"
        That is, the words always come as a whole. We can only enter at word-starts and exits at word-ends.

        `blank_penalty` should be negative (e.g., blank_penalty=-1.0):
        -- this is because in k2, the weights on the arcs are interpreted as "log-probs"
        -- we prefer the arc weights the larger the better
    '''

    word_id_list_flattened = flatten_list(word_id_list)

    arcs = []
    start_state = 0
    final_blk_state = 2*len(word_id_list_flattened) + 1
    final_state = final_blk_state + 1

    eps = 0
    # arc: (ss, ee, l1, l2, weight)
    arcs.append((start_state, start_state, eps, eps, blank_penalty))
    
    prev_blk_state = start_state
    prev_non_blk_state = None
    prev_p = None
    token_cnt = 0
    
    for w in word_id_list:
        # `w` is the list of tokens for this word
        assert len(w) > 0, f"len(w)={len(w)}, w={w}, word_id_list={word_id_list}"

        if isinstance(w[0], list):
            # TODO: for the moment, we only consider `word_id_list` as a list of integers.
            # That is, each word has only one tokenization
            # Otherwise, if the word has multiple tokenizations, we take only the first one
            w = w[0]
        
        # For each word/token, we do the following:
        # 1) link the graph of this word/token to the existing graph
        # 2) draw the graph of this word/token
        # 3) add factor arcs
        
        len_w = len(w)
        for i_p, p in enumerate(w):
            # `p` is the token
            cur_non_blk_state = 2*token_cnt + 1
            cur_blk_state = cur_non_blk_state + 1

            # 1) link it to the existing graph
            if prev_non_blk_state is not None:
                if p != prev_p:
                    arcs.append((prev_non_blk_state, cur_non_blk_state, p, p, 0))

            if prev_blk_state is not None and prev_blk_state > 0:
                arcs.append((prev_blk_state, cur_non_blk_state, p, p, 0))

            # 2) draw the graph of this token
            arcs.append((cur_non_blk_state, cur_non_blk_state, p, eps, 0))
            arcs.append((cur_non_blk_state, cur_blk_state, eps, eps, 0))
            arcs.append((cur_blk_state, cur_blk_state, eps, eps, 0))

            # 3) add factor arcs
            if i_p == 0:  # word-start
                arcs.append((start_state, cur_non_blk_state, p, p, 0))  # in-coming factor arc
            if i_p == len_w - 1:  # word-end
                arcs.append((cur_non_blk_state, final_state, -1, -1, 0))  # out-going factor arc 1
                arcs.append((cur_non_blk_state, final_blk_state, eps, eps, 0))  # out-going factor arc 2

            prev_non_blk_state = cur_non_blk_state
            prev_blk_state = cur_blk_state
            prev_p = p
            
            token_cnt += 1

    assert prev_non_blk_state == final_state - 3, f"cur_state={prev_non_blk_state}, final_state={final_state}"
    arcs.append((final_blk_state, final_blk_state, eps, eps, blank_penalty))
    arcs.append((final_blk_state, final_state, -1, -1, 0))
    arcs.append((prev_blk_state, final_state, -1, -1, 0))
    arcs.append((final_state,))

    new_arcs = sorted(arcs, key=lambda arc: arc[0])
    new_arcs = [" ".join(map(str, arc)) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        return fst


def make_factor_transducer_word_level_index(word_id_list, return_str=False, blank_penalty=0):
    '''
        This is a modification of `make_factor_transducer_word_level`, but we output "word indices" instead of "word symbols"

        `blank_penalty` should be negative (e.g., blank_penalty=-1.0):
        -- this is because in k2, the weights on the arcs are interpreted as "log-probs"
        -- we prefer the arc weights the larger the better
    '''

    word_id_list_flattened = flatten_list(word_id_list)

    arcs = []
    start_state = 0
    final_blk_state = 2*len(word_id_list_flattened) + 1
    final_state = final_blk_state + 1

    eps = 0
    # arc: (ss, ee, l1, l2, weight)
    arcs.append((start_state, start_state, eps, eps, blank_penalty))

    word_index_sym_tab = {eps: eps, 1: 0}  # need to handle special cases
    token_sym_tab = {eps: eps}
    sym_id = 1

    prev_blk_state = start_state
    prev_non_blk_state = None
    prev_p = None
    token_cnt = 0
    
    for i_w, w in enumerate(word_id_list):
        # `w` is the list of tokens for this word
        assert len(w) > 0, f"len(w)={len(w)}, w={w}, word_id_list={word_id_list}"

        if isinstance(w[0], list):
            # TODO: for the moment, we only consider `word_id_list` as a list of integers.
            # That is, each word has only one tokenization
            # Otherwise, if the word has multiple tokenizations, we take only the first one
            w = w[0]
        
        # For each word/token, we do the following:
        # 1) link the graph of this word/token to the existing graph
        # 2) draw the graph of this word/token
        # 3) add factor arcs
        
        len_w = len(w)
        for i_p, p in enumerate(w):
            # `p` is the token
            cur_non_blk_state = 2 * token_cnt + 1
            cur_blk_state = cur_non_blk_state + 1
            
            token_sym_tab[sym_id] = p

            # 1) link it to the existing graph
            if prev_non_blk_state is not None:
                if p != prev_p:
                    arcs.append((prev_non_blk_state, cur_non_blk_state, p, sym_id, 0))
                    if i_p == 0:
                        word_index_sym_tab[sym_id] = i_w

            if prev_blk_state is not None and prev_blk_state > 0:
                arcs.append((prev_blk_state, cur_non_blk_state, p, sym_id, 0))
                if i_p == 0:
                    word_index_sym_tab[sym_id] = i_w

            # 2) draw the graph of this token
            arcs.append((cur_non_blk_state, cur_non_blk_state, p, sym_id, 0))
            arcs.append((cur_non_blk_state, cur_blk_state, eps, eps, 0))
            arcs.append((cur_blk_state, cur_blk_state, eps, eps, 0))

            # 3) add factor arcs
            if i_p == 0:  # word-start
                arcs.append((start_state, cur_non_blk_state, p, sym_id, 0))  # in-coming factor arc
            if i_p == len_w - 1:  # word-end
                # arcs.append((cur_non_blk_state, final_state, -1, -1, 0))  # out-going factor arc 1
                arcs.append((cur_non_blk_state, final_blk_state, eps, sym_id + 1, 0))  # out-going factor arc 2
                arcs.append((cur_non_blk_state, final_blk_state, p, sym_id + 1, 0))  # out-going factor arc 3

            prev_non_blk_state = cur_non_blk_state
            prev_blk_state = cur_blk_state
            prev_p = p
            
            token_cnt += 1
            sym_id += 1

    assert prev_non_blk_state == final_state - 3, f"cur_state={prev_non_blk_state}, final_state={final_state}"
    arcs.append((final_blk_state, final_blk_state, eps, eps, blank_penalty))
    arcs.append((final_blk_state, final_state, -1, -1, 0))
    # arcs.append((prev_blk_state, final_state, -1, -1, 0))
    arcs.append((prev_blk_state, final_blk_state, eps, sym_id, 0))
    arcs.append((final_state,))
    word_index_sym_tab[sym_id] = i_w + 1

    new_arcs = sorted(arcs, key=lambda arc: arc[0])
    new_arcs = [" ".join(map(str, arc)) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs, word_index_sym_tab, token_sym_tab
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        return fst, word_index_sym_tab, token_sym_tab


def make_factor_transducer_word_level_index_with_skip(
    word_id_list,
    return_str=False,
    blank_penalty=0, 
    skip_penalty=-0.5, 
    return_penalty=None, 
    noneps_bonus=0.0,
):
    '''
        This is a modification of `make_factor_transducer_word_level_index`, but we allow skip arcs instead of a standard factor transducer
        That means the "factors" do not need to be consecutive words. Some words can be skipped.

        `skip_penalty` is good for breaking ties: it prefers the shorter path without skips.
        That is, the more skips, the bigger the penalty.

        `return_penalty` is to allow paths to return to the start state.
        This is useful when we hope to allow the paths to "regret" and go back into the fst graph

        `blank_penalty` should be negative (e.g., blank_penalty=-1.0):
        -- this is because in k2, the weights on the arcs are interpreted as "log-probs"
        -- we prefer the arc weights the larger the better

        `noneps_bonus` give some prior probability to the non-blank arcs
    '''

    word_id_list_flattened = flatten_list(word_id_list)

    skip_id = 2*len(word_id_list_flattened) + 1
    return_id = skip_id + 1

    arcs = []
    start_state = 0
    final_blk_state = 2*len(word_id_list_flattened) + 1
    final_state = final_blk_state + 1

    eps = 0
    # arc: (ss, ee, l1, l2, weight)
    arcs.append((start_state, start_state, eps, eps, blank_penalty))
    
    # We hope to have two aux labels. So we make a lookup table
    word_index_sym_tab = {eps: eps, 1: 0}  # need to handle corner cases
    token_sym_tab = {eps: eps}
    sym_id = 1
    
    prev_blk_state = start_state
    prev_non_blk_state = None
    prev_p = None
    token_cnt = 0

    blank_states_for_skip = []  # use -1 to seperate the states of different words
    
    for i_w, w in enumerate(word_id_list):
        # `w` is the list of tokens for this word
        assert len(w) > 0, f"len(w)={len(w)}, w={w}, word_id_list={word_id_list}"

        if isinstance(w[0], list):
            # TODO: for the moment, we only consider `word_id_list` as a list of integers.
            # That is, each word has only one tokenization
            # Otherwise, if the word has multiple tokenizations, we take only the first one
            w = w[0]
        
        # For each word/token, we do the following:
        # 1) link the graph of this word/token to the existing graph
        # 2) draw the graph of this word/token
        # 3) add factor arcs
        # 4) add skip arcs
        # 5) add return arcs
        
        len_w = len(w)
        for i_p, p in enumerate(w):
            # `p` is the token
            cur_non_blk_state = 2*token_cnt + 1
            cur_blk_state = cur_non_blk_state + 1

            token_sym_tab[sym_id] = p

            if i_p == 0:
                blank_states_for_skip.append(-1)
            blank_states_for_skip.append(cur_blk_state)

            # 1) link it to the existing graph
            if prev_non_blk_state is not None:
                if p != prev_p:
                    arcs.append((prev_non_blk_state, cur_non_blk_state, p, sym_id, 0 + noneps_bonus))
                    if i_p == 0:
                        word_index_sym_tab[sym_id] = i_w

            if prev_blk_state is not None and prev_blk_state > 0:
                arcs.append((prev_blk_state, cur_non_blk_state, p, sym_id, 0 + noneps_bonus))
                if i_p == 0:
                    word_index_sym_tab[sym_id] = i_w

            # 2) draw the graph of this token
            arcs.append((cur_non_blk_state, cur_non_blk_state, p, eps, 0 + noneps_bonus))
            arcs.append((cur_non_blk_state, cur_blk_state, eps, eps, 0))
            arcs.append((cur_blk_state, cur_blk_state, eps, eps, 0))

            # 3) add factor arcs
            if i_p == 0:  # word-start
                arcs.append((start_state, cur_non_blk_state, p, sym_id, 0 + noneps_bonus))  # in-coming factor arc
            if i_p == len_w - 1:  # word-end
                # arcs.append((cur_non_blk_state, final_state, -1, -1, 0))  # out-going factor arc 1
                arcs.append((cur_non_blk_state, final_blk_state, eps, sym_id + 1, 0))  # out-going factor arc 2
                arcs.append((cur_non_blk_state, final_blk_state, p, sym_id + 1, 0 + noneps_bonus))  # out-going factor arc 3

                # 5) add return arcs
                if return_penalty is not None:
                    arcs.append((cur_non_blk_state, start_state, eps, return_id, return_penalty))

            prev_non_blk_state = cur_non_blk_state
            prev_blk_state = cur_blk_state
            prev_p = p
            
            token_cnt += 1
            sym_id += 1

    # 4) add skip arcs
    # Note: we need to avoid "leakage" of a word to the other words -- i.e., 
    # the phones of this word go to the phones of the other words without outputing word indices
    for s1, s2 in zip(blank_states_for_skip, blank_states_for_skip[1:-1]):
        if s1 == -1 or s2 == -1:  # word boundary
            continue
        arcs.append((s1, s2, eps, skip_id, skip_penalty))

    assert prev_non_blk_state == final_state - 3, f"cur_state={prev_non_blk_state}, final_state={final_state}"
    arcs.append((final_blk_state, final_blk_state, eps, eps, blank_penalty))
    arcs.append((final_blk_state, final_state, -1, -1, 0))
    # arcs.append((prev_blk_state, final_state, -1, -1, 0))
    arcs.append((prev_blk_state, final_blk_state, eps, sym_id, 0))
    arcs.append((final_state,))
    word_index_sym_tab[sym_id] = i_w + 1

    assert skip_id > sym_id, f"Please increase your skip_id={skip_id} to be greater than {sym_id}"
    assert skip_id > i_w + 1, f"Please increase your skip_id={skip_id} to be greater than {i_w + 1}"

    new_arcs = sorted(arcs, key=lambda arc: arc[0])
    new_arcs = [" ".join(map(str, arc)) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs, word_index_sym_tab, token_sym_tab
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        fst.skip_id = int(skip_id)
        fst.return_id = int(return_id)
        return fst, word_index_sym_tab, token_sym_tab


# def make_factor_transducer1(word_id_list, return_str=False, blank_penalty=0):
#     # This is the original, simplest factor transducer for a "linear" fst

#     # word_id_list = flatten_list(word_id_list)

#     fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

#     c_str = k2.to_str_simple(fst_graph)
#     arcs = c_str.strip().split("\n")
#     arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
#     final_state = int(arcs[-1])
    
#     arcs = arcs[:-1]
#     arcs = [tuple(map(int, a.split())) for a in arcs]
#     # ss, ee, l1, l2, w = arc

#     non_eps_nodes = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0)   # if this node has a non-eps in-coming arc
#     arcs += [(0, n, l, l, 0) for n, l in non_eps_nodes if n > 1]

#     arcs += [(n, final_state, -1, -1, 0) for n in range(1, final_state - 2)]

#     new_arcs = arcs
#     new_arcs.append([final_state])

#     new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
#     new_arcs = [[str(i) for i in arc] for arc in new_arcs]
#     new_arcs = [" ".join(arc) for arc in new_arcs]
#     new_arcs = "\n".join(new_arcs)

#     if return_str:
#         return new_arcs
#     else:
#         fst = k2.Fsa.from_str(new_arcs, acceptor=False)
#         return fst


# def make_factor_transducer2(word_id_list, return_str=False, blank_penalty=-1):
#     # This is the factor transducer where blank symbols at the beginning and ending of the graph is penalized
#     # Last resort: use a cheap alignment model to get a subgraph of the big graph first

#     # blank_penalty should be negative:
#     # -- this is because in k2, the weights on the arcs are interpreted as "log-probs"
#     # -- we prefer the arc weights the larger the better

#     # word_id_list = flatten_list(word_id_list)

#     fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

#     c_str = k2.to_str_simple(fst_graph)
#     arcs = c_str.strip().split("\n")
#     arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
#     final_state = int(arcs[-1])
    
#     arcs = arcs[:-1]
#     arcs = [tuple(map(int, a.split())) for a in arcs]
#     # ss, ee, l1, l2, w = arc

#     arc0 = arcs[0]
#     arcs_last = [a for a in arcs[-5:] if a[2] > 0]

#     arcs = [(0, 0, 0, 0, blank_penalty)] + arcs[1:-5] + arcs_last

#     non_eps_nodes = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0)   # if this node has a non-eps in-coming arc
#     arcs += [(0, n, l, l, 0) for n, l in non_eps_nodes if n > 1]

#     # arcs += [(n, final_state, -1, -1, 0) for n in range(1, final_state - 2)]
#     arcs += [(n, final_state - 1, 0, 0, blank_penalty) for n, l in non_eps_nodes]
#     arcs += [(final_state - 1, final_state - 1, 0, 0, blank_penalty)]
#     arcs += [(final_state - 1, final_state, -1, -1, 0)]

#     new_arcs = arcs
#     new_arcs.append([final_state])

#     new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
#     new_arcs = [[str(i) for i in arc] for arc in new_arcs]
#     new_arcs = [" ".join(arc) for arc in new_arcs]
#     new_arcs = "\n".join(new_arcs)

#     if return_str:
#         return new_arcs
#     else:
#         fst = k2.Fsa.from_str(new_arcs, acceptor=False)
#         return fst


# def make_factor_transducer3(word_id_list, word_start_symbols, return_str=False, blank_penalty=0):
#     # This is a modification of make_factor_transducer1, where the factors are on "word-level"
#     # That is, the words always come as a whole. We can only enter at word-starts and exits at word-ends.

#     # word_id_list = flatten_list(word_id_list)

#     fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

#     c_str = k2.to_str_simple(fst_graph)
#     arcs = c_str.strip().split("\n")
#     arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
#     final_state = int(arcs[-1])
    
#     arcs = arcs[:-1]
#     arcs = [tuple(map(int, a.split())) for a in arcs]
#     # ss, ee, l1, l2, w = arc

#     word_start = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start in-coming arc
#     word_end = [w - j for w, l in word_start for j in [1, 2] if w - j > 0]
#     word_end.extend([final_state - 2, final_state - 1])
#     # word_end = [w for w in word_end if self_loops[w] > 0]

#     # print(sorted(word_start)[:100])
#     # print(sorted(word_end)[:100])

#     # assert len(word_start) == len([i for i in word_id_list if i in word_start_symbols])
#     # assert len(word_start) * 2 == len(word_end), f"len(word_start)={len(word_start)}, len(word_end)={len(word_end)}"
 
#     arcs += [(0, n, l, l, 0) for (n, l) in word_start if n > 1]

#     word_end = word_end[:-2]
#     arcs += [(n, final_state, -1, -1, 0) for n in word_end]

#     new_arcs = arcs
#     new_arcs.append([final_state])

#     new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
#     new_arcs = [[str(i) for i in arc] for arc in new_arcs]
#     new_arcs = [" ".join(arc) for arc in new_arcs]
#     new_arcs = "\n".join(new_arcs)

#     if return_str:
#         return new_arcs
#     else:
#         fst = k2.Fsa.from_str(new_arcs, acceptor=False)
#         return fst


# def make_factor_transducer4(word_id_list, word_start_symbols, return_str=False, blank_penalty=0):
#     # This is a modification of make_factor_transducer3, but we only output word indices instead of word symbols

#     # word_id_list = flatten_list(word_id_list)

#     fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

#     c_str = k2.to_str_simple(fst_graph)
#     arcs = c_str.strip().split("\n")
#     arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
#     final_state = int(arcs[-1])
    
#     arcs = arcs[:-1]
#     arcs = [tuple(map(int, a.split())) for a in arcs]
#     # ss, ee, l1, l2, w = arc

#     self_loops = {ss: l1 for ss, ee, l1, l2, w in arcs if ss == ee}
#     word_start = [(s, l) for s, l in self_loops.items() if l > 0 and l in word_start_symbols]   # if this node has a non-eps, word-start in-coming arc
#     word_start_ids = {s: i + 1 for i, (s, l) in enumerate(word_start)}
#     word_end = [(w - j, w) for w, l in word_start for j in [1, 2] if w - j > 0]
#     word_end.extend([(final_state - 2, None), (final_state - 1, None)])
#     word_end = [(w, w_next) for w, w_next in word_end if self_loops[w] > 0]
#     word_start_ids[None] = len(word_start) + 1

#     # assert len(word_start) == len([i for i in word_id_list if i in word_start_symbols])
#     # assert len(word_start) == len(word_end), f"len(word_start)={len(word_start)}, len(word_end)={len(word_end)}"

#     arcs = [arcs[0]] + arcs[2:-5] + [a for a in arcs[-5:] if a[2] > 0]
#     arcs = [[ss, ee, l1, word_start_ids.get(ee, 0) if ss != ee else 0, w] for ss, ee, l1, l2, w in arcs]
#     arcs += [(0, n, l, word_start_ids[n], 0) for n, l in word_start]
#     arcs += [(n, final_state, self_loops[n], word_start_ids[n_next], 0) for n, n_next in word_end]
#     arcs += [(n, final_state, 0, word_start_ids[n_next], 0) for n, n_next in word_end]
#     arcs += [(final_state, final_state, 0, 0, 0), (final_state, final_state + 1, -1, -1, 0)]
    
#     # non_eps_nodes2 = [(n, l) for n, l in non_eps_nodes2 if 0 < n < final_state - 2]
    
#     new_arcs = arcs
#     new_arcs.append([final_state + 1])

#     new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
#     new_arcs = [[str(i) for i in arc] for arc in new_arcs]
#     new_arcs = [" ".join(arc) for arc in new_arcs]
#     new_arcs = "\n".join(new_arcs)

#     # print(new_arcs)

#     if return_str:
#         return new_arcs
#     else:
#         fst = k2.Fsa.from_str(new_arcs, acceptor=False)
#         return fst


# def make_factor_transducer4_bigram(word_id_list, word_start_symbols, return_str=False, blank_penalty=0):
#     # This is a modification of make_factor_transducer4, but we just use a bigram graph instead of a factor transducer
#     # This will be similar to gentle: https://github.com/lowerquality/gentle/blob/master/gentle/language_model.py

#     fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]
#     raise NotImplementedError


# def make_factor_transducer4_skip(word_id_list, word_start_symbols, return_str=False, blank_penalty=0, skip_penalty=-0.5, return_penalty=None, noneps_bonus=0.0):
#     # This is a modification of make_factor_transducer4, but we allow skip arcs instead of a factor transducer
#     # `skip_penalty` is good for breaking ties: it prefers the shorter path without skips

#     # word_id_list = flatten_list(word_id_list)

#     fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

#     c_str = k2.to_str_simple(fst_graph)
#     arcs = c_str.strip().split("\n")
#     arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
#     final_state = int(arcs[-1])

#     arcs = arcs[:-1]
#     arcs = [tuple(map(int, a.split())) for a in arcs]
#     # ss, ee, l1, l2, w = arc

#     self_loops = {ss: l1 for ss, ee, l1, l2, w in arcs if ss == ee}
#     word_start = [(s, l) for s, l in self_loops.items() if l > 0 and l in word_start_symbols]   # if this node has a non-eps, word-start in-coming arc
#     word_start_ids = {s: i + 1 for i, (s, l) in enumerate(word_start)}
#     word_end = [(w - j, w) for w, l in word_start for j in [1, 2] if w - j > 0]
#     word_end.extend([(final_state - 2, None), (final_state - 1, None)])
#     word_end = [(w, w_next) for w, w_next in word_end if self_loops[w] > 0]
#     word_start_ids[None] = len(word_start) + 1

#     # assert len(word_start) == len([i for i in word_id_list if i in word_start_symbols])
#     # assert len(word_start) == len(word_end), f"len(word_start)={len(word_start)}, len(word_end)={len(word_end)}"

#     eps_self_loops = [ss for ss, l1 in self_loops.items() if l1 == 0]
#     eps_self_loops = sorted(eps_self_loops[1:-1]) + [final_state]

#     arcs = [arcs[0]] + arcs[2:-5] + [a for a in arcs[-5:] if a[2] > 0]
#     arcs = [[ss, ee, l1, word_start_ids.get(ee, 0) if ss != ee else 0, w + noneps_bonus if l1 > 0 else w] for ss, ee, l1, l2, w in arcs]

#     # in-coming arcs
#     arcs += [(0, n, l, word_start_ids[n], noneps_bonus if l > 0 else 0) for n, l in word_start]

#     # out-going arcs
#     # arcs += [(n, final_state, self_loops[n], counter.f3(), 0) for n, l in non_eps_nodes2 if self_loops[n] > 0]
#     # arcs += [(final_state - 1, final_state, self_loops[final_state - 1], counter.counter3, 0)]

#     arcs += [(n, final_state, self_loops[n], word_start_ids[n_next], 0) for n, n_next in word_end]
#     arcs += [(n, final_state, 0, word_start_ids[n_next], 0) for n, n_next in word_end]

#     if return_penalty is not None:
#         arcs += [(n, 0, 0, 0, return_penalty) for n, n_next in word_end]

#     # skip arcs
#     if skip_penalty is not None:
#         # arcs += [(n, next_token, self_loops[next_token], 0, 0) for ns, next_token in zip(eps_arcs1, token_starts[2:]) for n in ns]
#         arcs += [(n1, n2, 0, 0, skip_penalty) for n1, n2 in zip(eps_self_loops, eps_self_loops[1:])]

#     # validation: monotonic order
#     if False:
#         node_arcs_in = defaultdict(list)
#         node_arcs_out = defaultdict(list)
#         for a in arcs:
#             ss, ee, l1, l2, w = a
#             node_arcs_out[ss].append(a)
#             node_arcs_in[ee].append(a)
#         cur_edge_id = -1
#         breakpoint()
#         for i in range(1, final_state):
#             for in_arc in node_arcs_in[i]:
#                 if in_arc[0] == 0 and in_arc[3] > 0:
#                     assert in_arc[3] >= cur_edge_id, f"in_arc={in_arc}, cur_edge_id={cur_edge_id}"
#                     cur_edge_id = in_arc[3]
#             for out_arc in node_arcs_out[i]:
#                 if out_arc[1] == final_state and out_arc[3] > 0:
#                     assert out_arc[3] >= cur_edge_id, f"out_arc={out_arc}, cur_edge_id={cur_edge_id}"
#                     cur_edge_id = out_arc[3]
    
#     # validation: same label to the final-1 state or next state
#     if False:
#         word_start = {w[0]: i+1 for i, w in enumerate(word_start)}
#         word_end = [w - j for w, i in word_start.items() for j in [1, 2] if w - j > 0]
#         # word_end.extend([final_state - 2, final_state - 1])
#         word_end = [w for w in word_end if self_loops[w] > 0]
#         # print(sorted(word_start)[:100])
#         # print(sorted(word_end)[:100])

#         assert len(word_start) == len(word_end) + 1
#         # assert len(word_start) == len(book.split())
#         # assert len(word_start) == len(non_eps_nodes2)

#         node_arcs_in = defaultdict(list)
#         node_arcs_out = defaultdict(list)
#         for a in arcs:
#             ss, ee, l1, l2, w = a
#             node_arcs_out[ss].append(a)
#             node_arcs_in[ee].append(a)
#         for i in word_end:
#             out_arc_label = None
#             next_word_label = None
#             for out_arc in node_arcs_out[i]:
#                 if out_arc[1] == final_state:
#                     out_arc_label = out_arc[3]
#                 if out_arc[1] in word_start and out_arc[0] != out_arc[1]:
#                     next_word_label = word_start[out_arc[1]]
#             if next_word_label is None:
#                 next_word_label = word_start.get(i + 2, None)
#             if self_loops[i] > 0:
#                 assert out_arc_label == next_word_label, f"state={i}, out_arc_label={out_arc_label}, next_word_label={next_word_label}"

#     arcs += [(final_state, final_state, 0, 0, 0)]
#     arcs += [(final_state, final_state + 1, -1, -1, 0)]

#     # non_eps_nodes2 = [(n, l) for n, l in non_eps_nodes2 if 0 < n < final_state - 2]

#     new_arcs = arcs
#     new_arcs.append([final_state + 1])

#     new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
#     new_arcs = [[str(i) for i in arc] for arc in new_arcs]
#     new_arcs = [" ".join(arc) for arc in new_arcs]
#     new_arcs = "\n".join(new_arcs)

#     # print(new_arcs)

#     if return_str:
#         return new_arcs
#     else:
#         fst = k2.Fsa.from_str(new_arcs, acceptor=False)
#         return fst


# def get_decoding_graphs(_texts, sp):
#     fst_graph = k2.ctc_graph(sp.encode(_texts, out_type=int), modified=False, device='cpu')
#     return [fst_graph[i] for i in range(fst_graph.shape[0])]


# def get_decoding_graphs_factor_transducer(token_ids):
#     return [make_factor_transducer1(t) for t in token_ids]


# def _make_factor_transducer5(fst_graphs_3, word_start_symbols, two_ends_bonus=1.0):
#     # This is similar to make_factor_transducer3
#     # We concat three graphs, and then add some bonus to the two ends of the graph to encourage alignment to the two ends
#     # We will probably also do non-linear scaling

#     # Compared to make_factor_transducer3, we remove some arcs from the graph
#     # Also, we modify some arcs to add bonus to the two ends of the graph to encourage alignment to the two ends

#     ############ Graph 1 ############
#     fst_graph1 = fst_graphs_3[0]
    
#     c_str = k2.to_str_simple(fst_graph1)
#     arcs = c_str.strip().split("\n")
#     arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
#     final_state = int(arcs[-1])

#     arcs = arcs[:-1]
#     arcs = [tuple(map(int, a.split())) for a in arcs]
#     # ss, ee, l1, l2, w = arc

#     non_eps_nodes = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start in-coming arc
#     arcs += [(0, n, l, l, 0) for n, l in non_eps_nodes if n > 1]

#     # boost the weights
#     arcs = [(ss, ee, l1, l2, w + two_ends_bonus) if (ss>0 and l1>0) else (ss, ee, l1, l2, w + two_ends_bonus * 5) if (ee>0 and l1>0) else (ss, ee, l1, l2, w) for ss, ee, l1, l2, w in arcs]

#     arcs += [(0, final_state, -1, -1, 0)]

#     new_arcs = arcs
#     new_arcs.append([final_state])

#     new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
#     new_arcs = [[str(i) for i in arc] for arc in new_arcs]
#     new_arcs = [" ".join(arc) for arc in new_arcs]
#     new_arcs = "\n".join(new_arcs)

#     fst_graph1 = k2.Fsa.from_str(new_arcs, acceptor=False)

#     ############ Graph 2 ############
#     fst_graph2 = fst_graphs_3[1]

#     ############ Graph 3 ############
#     fst_graph3 = fst_graphs_3[2]

#     c_str = k2.to_str_simple(fst_graph3)
#     arcs = c_str.strip().split("\n")
#     arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
#     final_state = int(arcs[-1])

#     arcs = arcs[:-1]
#     arcs = [tuple(map(int, a.split())) for a in arcs]
#     # ss, ee, l1, l2, w = arc

#     non_eps_nodes2 = set((arc[0], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start out-going arc
#     non_eps_nodes2 = [(n, l) for n, l in non_eps_nodes2 if 0 < n < final_state - 2]
#     arcs += [(n, final_state, -1, -1, 0) for n, l in non_eps_nodes2]

#     # boost the weights
#     arcs = [(ss, ee, l1, l2, w + two_ends_bonus) if (ee!=final_state and l1>0) else (ss, ee, l1, l2, w + two_ends_bonus*5) if l1>0 else (ss, ee, l1, l2, w) for ss, ee, l1, l2, w in arcs]

#     arcs += [(0, final_state, -1, -1, 0)]

#     new_arcs = arcs
#     new_arcs.append([final_state])

#     new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
#     new_arcs = [[str(i) for i in arc] for arc in new_arcs]
#     new_arcs = [" ".join(arc) for arc in new_arcs]
#     new_arcs = "\n".join(new_arcs)

#     fst_graph3 = k2.Fsa.from_str(new_arcs, acceptor=False)

#     # concatenate the three graphs: don't be fooled by `k2.cat`
#     compiler = openfst.Compiler()
#     compiler.write(k2.to_str_simple(fst_graph1, openfst=True))
#     f1 = compiler.compile()
#     compiler.write(k2.to_str_simple(fst_graph2, openfst=True))
#     f2 = compiler.compile()
#     compiler.write(k2.to_str_simple(fst_graph3, openfst=True))
#     f3 = compiler.compile()
#     f1.concat(f2)
#     f1.concat(f3)
#     fst = k2.Fsa.from_openfst(str(f1), acceptor=False)
#     fst = k2.arc_sort(fst)
#     return fst


# def make_factor_transducer5(libri_long_text_str, cut_ids, text_ranges, sp, extension=10, two_ends_bonus=1.0):
#     # This factor transducer enforce some specific factors to be present in the graph

#     text_ranges = [(max(rg[0]-1, 0), rg[-1]-1) for cid, rg in zip(cut_ids, text_ranges)]
#     text_ranges = [
#         (
#             (max(rg[0]-extension, 0), rg[0]),  # left extension
#             (rg[0], rg[-1]),                    # alignment results
#             (rg[-1], rg[-1]+extension)           # right extension
#         ) for cid, rg in zip(cut_ids, text_ranges)
#     ]
#     text_keys = [tuple(get_uid_key(cid)[:2]) for cid in cut_ids]
#     texts = [" ".join(libri_long_text_str[tk][rg[0]: rg[-1]]) for tk, rg3 in zip(text_keys, text_ranges) for rg in rg3]
#     token_ids = sp.encode(texts, out_type=int)
#     fst_graphs = k2.ctc_graph(token_ids, modified=False, device='cpu')

#     # token_lens = [(len(left), len(mid), len(right)) for left, mid, right in zip(token_ids[::3], token_ids[1::3], token_ids[2::3])]
#     # word_id_lists = [left + mid + right for left, mid, right in zip(token_ids[::3], token_ids[1::3], token_ids[2::3])]

#     # breakpoint()

#     fst_graphs = [(fst_graphs[i], fst_graphs[i+1], fst_graphs[i+2]) for i in range(0, fst_graphs.shape[0], 3)]
#     word_start_symbols = {i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}
#     fst_graphs = [_make_factor_transducer5(fst_graphs_3, word_start_symbols, two_ends_bonus=two_ends_bonus) for fst_graphs_3 in fst_graphs]
    
#     return fst_graphs
