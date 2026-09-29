import pandas as pd

from data_diagnostics.ingestion import SemanticType, infer_schema


def _columns_by_name(schema):
    return {column.name: column for column in schema.columns}


def test_infer_mixed_dataset_schema() -> None:
    data = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004"],
            "is_active": [True, False, True, False],
            "quantity": [1, 2, 3, 2],
            "revenue": [10.50, 20.25, 15.75, 50.10],
            "city": ["Delhi", "Mumbai", "Delhi", "Bengaluru"],
            "customer_name": [
                "Alice Smith",
                "Bob Jones",
                "Charlie Brown",
                "Diana Williams",
            ],
        }
    )

    schema = infer_schema(data)
    columns = _columns_by_name(schema)

    assert schema.row_count == 4
    assert schema.column_count == 6

    assert columns["customer_id"].semantic_type == SemanticType.IDENTIFIER
    assert columns["is_active"].semantic_type == SemanticType.BINARY
    assert columns["quantity"].semantic_type == SemanticType.NUMERIC_DISCRETE
    assert columns["revenue"].semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert columns["city"].semantic_type == SemanticType.CATEGORICAL
    assert columns["customer_name"].semantic_type == SemanticType.TEXT


def test_detect_native_datetime_column() -> None:
    data = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-01 10:00",
                    "2026-01-02 11:00",
                    "2026-01-03 12:00",
                ]
            )
        }
    )

    schema = infer_schema(data)
    column = schema.columns[0]

    assert column.semantic_type == SemanticType.DATETIME
    assert column.confidence == 1.0


def test_detect_datetime_string_column() -> None:
    data = pd.DataFrame(
        {
            "order_date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ]
        }
    )

    schema = infer_schema(data)

    assert schema.columns[0].semantic_type == SemanticType.DATETIME
    assert schema.columns[0].confidence >= 0.80


def test_detect_binary_numeric_column() -> None:
    data = pd.DataFrame(
        {
            "converted": [0, 1, 0, 1, 1],
        }
    )

    schema = infer_schema(data)

    assert schema.columns[0].semantic_type == SemanticType.BINARY


def test_all_missing_column_is_unknown() -> None:
    data = pd.DataFrame(
        {
            "unknown_value": [None, None, None],
        }
    )

    schema = infer_schema(data)
    column = schema.columns[0]

    assert column.semantic_type == SemanticType.UNKNOWN
    assert column.missing_count == 3
    assert column.missing_ratio == 1.0


def test_schema_contains_cardinality_metadata() -> None:
    data = pd.DataFrame(
        {
            "city": ["Delhi", "Delhi", "Mumbai", None],
        }
    )

    schema = infer_schema(data)
    column = schema.columns[0]

    assert column.unique_count == 2
    assert column.missing_count == 1
    assert column.missing_ratio == 0.25


def test_schema_inference_does_not_modify_dataframe() -> None:
    data = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "value": [10.5, 20.5],
        }
    )

    original = data.copy(deep=True)

    infer_schema(data)

    pd.testing.assert_frame_equal(data, original)
