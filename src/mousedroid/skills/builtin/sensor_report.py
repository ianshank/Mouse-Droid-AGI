"""``mousedroid-sensor-report`` — read-only structured scene description."""

from __future__ import annotations

from pydantic import BaseModel, Field

from mousedroid.skills.protocol import SkillSpec


class SensorReportInput(BaseModel):
    """Input schema for a sensor-report request."""

    include_lidar: bool = Field(True, description="Include LiDAR sectors in the report.")
    include_imu: bool = Field(True, description="Include IMU pose in the report.")
    include_battery: bool = Field(True, description="Include battery voltage / health.")


class SensorReportOutput(BaseModel):
    """Output schema for a sensor-report request."""

    timestamp_s: float = Field(..., description="Monotonic capture time in seconds.")
    lidar: list[float] | None = Field(None, description="Normalised sector readings.")
    pose: dict[str, float] | None = Field(None, description="IMU-derived pose.")
    battery_voltage_v: float | None = Field(None, description="Battery voltage in volts.")
    health: dict[str, float | str] | None = Field(None, description="HealthMonitor snapshot.")


SYSTEM_PROMPT = (
    "You are the sensor-report skill of a Star Wars MSE-6 Mouse Droid. "
    "Produce a concise structured snapshot of the requested sensors. "
    "Never call actuation tools."
)


SPEC = SkillSpec(
    name="mousedroid-sensor-report",
    description=(
        "Return the latest LiDAR, IMU, battery, and health snapshot as a "
        "structured JSON document for OpenClaw consumption."
    ),
    tool_names=frozenset({"read_distance", "read_encoders", "read_battery", "query_health"}),
    system_prompt=SYSTEM_PROMPT,
    schema_in=SensorReportInput,
    schema_out=SensorReportOutput,
    source="builtin",
    metadata={"actuation": False, "channel": ("rest", "mcp"), "version": "1.0.0"},
)


__all__ = ["SPEC", "SensorReportInput", "SensorReportOutput"]
