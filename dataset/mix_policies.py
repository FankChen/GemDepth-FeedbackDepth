"""Built-in dataset mixing policies."""

from dataset.mix_registry import register


@register("native")
def native(native_ratios):
    """Whatever each source already asks for. The historical behaviour.

    Kept as the default so every run recorded so far stays reproducible: the
    numbers on the dashboards were all measured under this mix.
    """
    return dict(native_ratios)


@register("uniform")
def uniform(counts):
    """Repeat each source until they all contribute about equally.

    This is what the paper's appendix describes -- a uniform sampling
    probability across sources -- and what the native mix is far from. Measured
    from a stage1_lite launch (dataset/dataset_mix.py prints the first four; the
    tartanair branch prints no count, so it is the remainder of the 834172-entry
    list the dataloader reports):

        PointOdyssey    301594 clips x 1  = 301594   36.2%
        TartanAir       295196 clips x 1  = 295196   35.4%
        MVS-Synth         8280 clips x 26 = 215280   25.8%
        VKITTI2          11342 clips x 1  =  11342    1.4%
        DynamicReplica   10760 clips x 1  =  10760    1.3%

    Note VKITTI2 against MVS-Synth: more clips, nineteen times less weight. That
    is not a judgement about the data, it is an omission -- MVS-Synth was given
    RATIO=26 to pull it up towards the big sets, and VKITTI2 was never wired
    into the same mechanism. It matters because VKITTI2 is the only
    driving-domain source we have and KITTI is the benchmark we read first.

    The effect is amplified by how little of the pool a run actually touches: at
    5000 steps and an effective batch of 16, training draws 80k clips from the
    834k-entry list, so a source at 1.4% is seen roughly a thousand times in
    total. The mixing distribution is close to the whole story at that budget.

    Repetition is integer, so "equal" is approximate for sources whose counts do
    not divide the largest one.
    """
    target = max(counts.values())
    return {label: max(1, round(target / n)) for label, n in counts.items()}
