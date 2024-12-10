import numpy as np
import pandas as pd
import pytest

from research.lib import stratified_group_split


@pytest.fixture
def samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": "abcdefg",
            "group": [100, 100, 100, 300, 200, 200, 200],
            "label1": [0, 0, 0, 1, 0, 0, 0],
            "label2": [1, 1, 1, 0, 0, 0, 0],
        }
    )


def test_create_per_group_indices_and_labels(samples: pd.DataFrame) -> None:
    per_group_indices, per_group_labels = stratified_group_split._create_per_group_indices_and_labels(  # noqa: SLF001
        samples[["label1", "label2"]], samples["group"]
    )
    # Groups should be sorted by value and then 0-indexed. 100 -> 0, 200 -> 1, 300 -> 2
    assert per_group_indices == {
        0: (0, 1, 2),
        1: (4, 5, 6),
        2: (3,),
    }
    assert per_group_labels[0].tolist() == [0, 1]
    assert per_group_labels[1].tolist() == [0, 0]
    assert per_group_labels[2].tolist() == [1, 0]


def test_convert_group_indices_to_sample_indices() -> None:
    per_group_indices = {
        0: (0, 1, 2),
        1: (4, 6),
        2: (3,),
        3: (5, 7),
    }
    group_index = np.array([0, 3])
    sample_indices = stratified_group_split._convert_group_indices_to_sample_indices(per_group_indices, group_index)  # noqa: SLF001
    assert sample_indices.tolist() == [0, 1, 2, 5, 7]


rng = np.random.default_rng()


# These random states are generated at import time.
@pytest.mark.parametrize("random_state", rng.integers(2**32, size=20).tolist())
@pytest.mark.parametrize(
    "splitter_class",
    [
        stratified_group_split.MultilabelStratifiedGroupShuffleSplit,
        stratified_group_split.MultilabelStratifiedGroupKFold,
    ],
)
def test_groups_are_disjoint(
    random_state: int,
    splitter_class: type,
    samples: pd.DataFrame,
) -> None:
    if splitter_class is stratified_group_split.MultilabelStratifiedGroupShuffleSplit:
        splitter = splitter_class(
            n_splits=1,
            test_size=0.5,
            random_state=random_state,
        )
    else:
        splitter = splitter_class(
            n_splits=3,
            shuffle=True,
            random_state=random_state,
        )
    for train_index, test_index in splitter.split(
        samples["x"],
        samples[["label1", "label2"]],
        samples["group"],
    ):
        train_groups = set(samples.iloc[train_index]["group"])
        test_groups = set(samples.iloc[test_index]["group"])
        assert not (train_groups & test_groups)
