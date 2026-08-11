TRAIN_SCENES = frozenset({"Scene01", "Scene02", "Scene18"})
TEST_SCENES = frozenset({"Scene06", "Scene20"})
VARIATION = "15-deg-left"


def scene_is_selected(scene, variation, mode):
    scenes = TRAIN_SCENES if mode == "train" else TEST_SCENES
    return scene in scenes and variation == VARIATION
