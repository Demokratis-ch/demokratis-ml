import dataclasses
from collections.abc import Callable, Iterator
from typing import Mapping

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from sklearn.preprocessing import OrdinalEncoder

__all__ = [
    "MultilabelStratifiedGroupShuffleSplit",
    "MultilabelStratifiedGroupKFold",
    "SplitResult",
    "one_simple_split",
]


class MultilabelStratifiedGroupShuffleSplit(MultilabelStratifiedShuffleSplit):
    def split(
        self,
        X,  # noqa: ARG002
        y: pd.DataFrame,
        groups: pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Performs a stratified group shuffle split on a DataFrame of samples; to achieve the grouping,
        the following steps are taken:

        1. Samples are grouped by the values in the ``groups`` Series. This explicitly requires that
           each sample in a group has the same ``y`` labels!
        2. Standard stratified shuffle split is performed on the groups.
        3. Resulting group indices are converted back to sample indices.

        :param X: Ignored. Only for compatibility with sklearn splitter interface.
        :param y: DataFrame of samples with columns as labels.
        :param groups: Series of group identifiers for each sample.
        """
        return _group_split(super().split, y, groups)


class MultilabelStratifiedGroupKFold(MultilabelStratifiedKFold):
    def split(
        self,
        X,  # noqa: ARG002
        y: pd.DataFrame,
        groups: pd.Series,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """See the docstring for :meth:`MultilabelStratifiedGroupShuffleSplit.split`."""
        return _group_split(super().split, y, groups)


@dataclasses.dataclass(frozen=True)
class SplitResult:
    X: np.ndarray
    y: pd.DataFrame
    groups: pd.Series

    def __post_init__(self):
        assert self.X.shape[0] == self.y.shape[0] == self.groups.shape[0]

    @property
    def shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.X.shape, self.y.shape


def one_simple_split(
    splitter: MultilabelStratifiedGroupShuffleSplit,
    X: np.ndarray,
    y: pd.DataFrame,
    groups: pd.Series,
) -> tuple[SplitResult, SplitResult]:
    """Use the configured ``splitter`` to perform a single split on the data and return the train and test
    splits as :class:`SplitResult` instances.

    :returns: (train, test)
    """
    assert splitter.get_n_splits() == 1
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.DataFrame)
    assert isinstance(groups, pd.Series)
    train_index, test_index = next(splitter.split(X, y, groups))
    return (
        SplitResult(
            X=X[train_index],
            y=y.iloc[train_index],
            groups=groups.iloc[train_index],
        ),
        SplitResult(
            X=X[test_index],
            y=y.iloc[test_index],
            groups=groups.iloc[test_index],
        ),
    )


def _group_split(
    original_splitter: Callable[[np.ndarray, np.ndarray], Iterator[tuple[np.ndarray, np.ndarray]]],
    y: pd.DataFrame,
    groups: pd.Series,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The logic of this function is described in the docstring
    of :meth:`MultilabelStratifiedGroupShuffleSplit.split`."""
    # These requirements are not standard for sklearn splitters but we can enforce them here
    # because we know the data coming in, and Series & DataFrame are convenient for us to use.
    assert isinstance(y, pd.DataFrame)
    assert isinstance(groups, pd.Series)

    per_group_sample_indices, per_group_labels = _create_per_group_indices_and_labels(y, groups)
    for groups_train_index, groups_test_index in original_splitter(
        np.zeros(len(per_group_labels)),
        per_group_labels,
    ):
        samples_train_index = _group_indices_to_sample_indices(per_group_sample_indices, groups_train_index)
        samples_test_index = _group_indices_to_sample_indices(per_group_sample_indices, groups_test_index)
        yield samples_train_index, samples_test_index


def _create_per_group_indices_and_labels(
    y: pd.DataFrame,
    groups: pd.Series,
) -> tuple[Mapping[int, tuple[int, ...]], np.ndarray]:
    """:returns: per_group_sample_indices, per_group_labels; where per_group_sample_indices is a dict
    mapping group index to a tuple of indices of samples belonging to that group, and
    per_group_labels is a 2D numpy array where each row corresponds to a group and each column
    corresponds to a label in y.
    """
    assert y.index.equals(groups.index)
    df = y.copy()
    df["sample_index"] = y.reset_index().index
    group_index_encoder = OrdinalEncoder(dtype=int)
    df["group_index"] = group_index_encoder.fit_transform(groups.to_numpy().reshape(-1, 1))

    def group_label_aggregator(group_label_column: pd.Series) -> int:
        # https://pandas.pydata.org/docs/user_guide/cookbook.html#constant-series
        array = group_label_column.to_numpy()
        if (array[0] != array).any():
            raise ValueError(
                "Group is not homogeneous",
                group_label_column.name,
                group_label_column.unique(),
            )
        return group_label_column.iloc[0]

    aggregated = df.groupby("group_index").agg(
        {
            "sample_index": tuple,
            **{y_column: group_label_aggregator for y_column in y.columns},
        }
    )
    per_group_sample_indices = aggregated["sample_index"].to_dict()
    per_group_labels = aggregated[y.columns].to_numpy()
    return per_group_sample_indices, per_group_labels


def _group_indices_to_sample_indices(
    per_group_sample_indices: Mapping[int, tuple[int, ...]],
    group_indices: np.ndarray,
) -> np.ndarray:
    """Unroll the group indices generated by :func:`_create_per_group_indices_and_labels` back into
    sample indices.
    """
    return np.concatenate([per_group_sample_indices[group] for group in group_indices])
