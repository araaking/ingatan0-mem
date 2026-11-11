from __future__ import annotations

import pytest

from ingatan import errors


@pytest.mark.parametrize(
    "cls",
    [
        errors.ConfigError,
        errors.NotFoundError,
        errors.ValidationError,
        errors.ScopeError,
        errors.SchemaError,
    ],
)
def test_subclasses_carry_a_message(cls: type[errors.IngatanError]) -> None:
    error = cls("something readable")
    assert isinstance(error, errors.IngatanError)
    assert str(error) == "something readable"
    assert error.message == "something readable"
