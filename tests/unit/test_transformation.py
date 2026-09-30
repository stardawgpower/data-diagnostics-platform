import pandas as pd
import pytest

from data_diagnostics.transformation import (
    DropColumns,
    InvalidTransformationError,
    RemoveEmptyRows,
    ReplaceValueWithMissing,
    TransformationPlan,
    UnknownColumnError,
    apply_transformation_plan,
)


def test_empty_plan_returns_unchanged_copy() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )

    plan = TransformationPlan()

    result = apply_transformation_plan(
        data,
        plan,
    )

    pd.testing.assert_frame_equal(
        result.data,
        data,
    )

    assert result.data is not data
    assert result.original_shape == (3, 2)
    assert result.final_shape == (3, 2)
    assert result.operation_count == 0


def test_remove_completely_empty_rows() -> None:
    data = pd.DataFrame(
        {
            "a": [1, None, 3, None],
            "b": ["x", None, "z", None],
        }
    )

    plan = TransformationPlan(operations=(RemoveEmptyRows(),))

    result = apply_transformation_plan(
        data,
        plan,
    )

    assert result.data.shape == (2, 2)

    assert result.log[0].affected_rows == 2
    assert result.log[0].operation == "remove_empty_rows"


def test_drop_selected_columns() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 6],
        }
    )

    plan = TransformationPlan(
        operations=(
            DropColumns(
                columns=("b", "c"),
            ),
        )
    )

    result = apply_transformation_plan(
        data,
        plan,
    )

    assert result.data.columns.tolist() == ["a"]
    assert result.final_shape == (2, 1)

    assert result.log[0].affected_columns == (
        "b",
        "c",
    )


def test_unknown_column_raises_error() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2],
        }
    )

    plan = TransformationPlan(
        operations=(
            DropColumns(
                columns=("missing_column",),
            ),
        )
    )

    with pytest.raises(UnknownColumnError):
        apply_transformation_plan(
            data,
            plan,
        )


def test_cannot_drop_every_column() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )

    plan = TransformationPlan(
        operations=(
            DropColumns(
                columns=("a", "b"),
            ),
        )
    )

    with pytest.raises(InvalidTransformationError):
        apply_transformation_plan(
            data,
            plan,
        )


def test_replace_sentinel_with_missing() -> None:
    data = pd.DataFrame(
        {
            "temperature": [
                10.0,
                -200.0,
                20.0,
            ],
            "humidity": [
                -200.0,
                50.0,
                -200.0,
            ],
        }
    )

    plan = TransformationPlan(
        operations=(
            ReplaceValueWithMissing(
                columns=(
                    "temperature",
                    "humidity",
                ),
                value=-200.0,
            ),
        )
    )

    result = apply_transformation_plan(
        data,
        plan,
    )

    assert result.data.isna().sum().sum() == 3

    assert result.log[0].affected_cells == 3
    assert result.log[0].affected_rows == 3


def test_replace_value_only_affects_selected_columns() -> None:
    data = pd.DataFrame(
        {
            "sensor_a": [-200, 10],
            "sensor_b": [-200, 20],
        }
    )

    plan = TransformationPlan(
        operations=(
            ReplaceValueWithMissing(
                columns=("sensor_a",),
                value=-200,
            ),
        )
    )

    result = apply_transformation_plan(
        data,
        plan,
    )

    assert pd.isna(
        result.data.loc[
            0,
            "sensor_a",
        ]
    )

    assert (
        result.data.loc[
            0,
            "sensor_b",
        ]
        == -200
    )


def test_transformation_order_is_preserved() -> None:
    data = pd.DataFrame(
        {
            "a": [-200.0, 1.0, None],
            "b": [-200.0, 2.0, None],
        }
    )

    replace_then_remove = TransformationPlan(
        operations=(
            ReplaceValueWithMissing(
                columns=("a", "b"),
                value=-200.0,
            ),
            RemoveEmptyRows(),
        )
    )

    remove_then_replace = TransformationPlan(
        operations=(
            RemoveEmptyRows(),
            ReplaceValueWithMissing(
                columns=("a", "b"),
                value=-200.0,
            ),
        )
    )

    first_result = apply_transformation_plan(
        data,
        replace_then_remove,
    )

    second_result = apply_transformation_plan(
        data,
        remove_then_replace,
    )

    assert len(first_result.data) == 1
    assert len(second_result.data) == 2

    assert [entry.operation for entry in first_result.log] == [
        "replace_value_with_missing",
        "remove_empty_rows",
    ]


def test_source_dataframe_is_never_modified() -> None:
    data = pd.DataFrame(
        {
            "a": [-200.0, 10.0, None],
            "b": [1.0, 2.0, None],
        }
    )

    original = data.copy(deep=True)

    plan = TransformationPlan(
        operations=(
            ReplaceValueWithMissing(
                columns=("a",),
                value=-200.0,
            ),
            RemoveEmptyRows(),
        )
    )

    apply_transformation_plan(
        data,
        plan,
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_empty_column_selection_is_invalid() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )

    plan = TransformationPlan(
        operations=(
            DropColumns(
                columns=(),
            ),
        )
    )

    with pytest.raises(InvalidTransformationError):
        apply_transformation_plan(
            data,
            plan,
        )


def test_replacing_missing_with_missing_is_invalid() -> None:
    data = pd.DataFrame(
        {
            "a": [1, None, 3],
        }
    )

    plan = TransformationPlan(
        operations=(
            ReplaceValueWithMissing(
                columns=("a",),
                value=None,
            ),
        )
    )

    with pytest.raises(InvalidTransformationError):
        apply_transformation_plan(
            data,
            plan,
        )


def test_duplicate_columns_in_operation_are_invalid() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )

    plan = TransformationPlan(
        operations=(
            DropColumns(
                columns=("a", "a"),
            ),
        )
    )

    with pytest.raises(InvalidTransformationError):
        apply_transformation_plan(
            data,
            plan,
        )


def test_non_scalar_replacement_value_is_invalid() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2, 3],
        }
    )

    plan = TransformationPlan(
        operations=(
            ReplaceValueWithMissing(
                columns=("a",),
                value=[1, 2],
            ),
        )
    )

    with pytest.raises(InvalidTransformationError):
        apply_transformation_plan(
            data,
            plan,
        )


def test_replace_value_unknown_column_raises_error() -> None:
    data = pd.DataFrame(
        {
            "a": [1, -200, 3],
        }
    )

    plan = TransformationPlan(
        operations=(
            ReplaceValueWithMissing(
                columns=("missing_column",),
                value=-200,
            ),
        )
    )

    with pytest.raises(UnknownColumnError):
        apply_transformation_plan(
            data,
            plan,
        )
