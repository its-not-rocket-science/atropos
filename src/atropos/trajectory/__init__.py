"""Canonical trajectory schema, serde helpers, and format converters."""

from .converters import (
    from_line_world_history,
    from_line_world_rollout,
    from_offline_rl_transitions,
    from_rlhf_pairs,
)
from .schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    RewardSignal,
    TrajectoryRecord,
    TrajectorySchemaSpec,
    TrajectoryStep,
    build_schema_spec,
    record_to_dict,
)
from .serde import (
    ArrowTableAdapter,
    from_arrow_rows,
    from_json,
    read_json,
    to_arrow_rows,
    to_json,
    write_json,
)

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "RewardSignal",
    "TrajectoryStep",
    "TrajectoryRecord",
    "TrajectorySchemaSpec",
    "build_schema_spec",
    "record_to_dict",
    "to_json",
    "from_json",
    "write_json",
    "read_json",
    "ArrowTableAdapter",
    "to_arrow_rows",
    "from_arrow_rows",
    "from_line_world_history",
    "from_line_world_rollout",
    "from_rlhf_pairs",
    "from_offline_rl_transitions",
]
