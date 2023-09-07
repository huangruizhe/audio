# https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

