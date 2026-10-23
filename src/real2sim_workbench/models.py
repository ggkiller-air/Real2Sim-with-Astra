"""Versioned scene records. SI units; right-handed Z-up world; image pixels."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
import math


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObjectRecord(Record):
    id: str = Field(pattern=r"^o_[a-f0-9]{12}$")
    name: str = Field(min_length=1, max_length=100)
    selected: bool = True
    mask: str | None = None
    bbox: list[float] = Field(min_length=4, max_length=4)
    score: float | None = None
    segmentation_source: str = "sam3"
    annotation: Literal["rigid", "hinge", "drawer", "cloth", "soft"] = "rigid"
    notes: str = Field(default="", max_length=2000)
    support: str = "table"
    geometry: Literal["none", "proxy", "imported", "sam3d"] = "none"
    mesh: str | None = None
    size: list[float] = Field(default_factory=lambda: [.08, .08, .1], min_length=3, max_length=3)
    position: list[float] = Field(default_factory=lambda: [0, 0, .05], min_length=3, max_length=3)
    yaw: float = 0
    mass: float = Field(default=.1, gt=0, le=1000)
    friction: float = Field(default=.5, ge=0, le=3)
    parameter_source: Literal["assumed", "user", "measured"] = "assumed"
    locked: bool = False  # Locks layout edits, never welds the simulated object.

    @model_validator(mode="after")
    def valid_geometry(self):
        if any(v <= 0 or v > 10 for v in self.size):
            raise ValueError("尺寸必须在 0–10 米之间")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("物体框不能为空")
        return self


class TableRecord(Record):
    width: float = Field(default=1.2, gt=.05, le=10)
    depth: float = Field(default=.8, gt=.05, le=10)
    # Clockwise: far-left, far-right, near-right, near-left.
    corners: list[list[float]] = Field(default_factory=list)
    calibration_source: Literal["assumed", "user"] = "assumed"

    @model_validator(mode="after")
    def valid_corners(self):
        if self.corners and (len(self.corners) != 4 or any(len(p) != 2 for p in self.corners)):
            raise ValueError("桌面需要四个像素坐标点")
        return self


class SceneRecord(Record):
    schema_version: str = "0.1"
    id: str = Field(pattern=r"^[a-f0-9]{16}$")
    name: str = Field(max_length=100)
    revision: int = 0
    image: str = "input/source.png"
    image_sha256: str
    image_size: list[int]
    table: TableRecord = Field(default_factory=TableRecord)
    objects: list[ObjectRecord] = Field(default_factory=list)
    preprocessing: dict = Field(default_factory=dict)
    reports: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_relations(self):
        items = {o.id: o for o in self.objects}
        if len(items) != len(self.objects):
            raise ValueError("物体 ID 重复")
        for obj in self.objects:
            seen = {obj.id}
            parent = obj.support
            while parent != "table":
                if parent not in items:
                    raise ValueError("支撑物不存在")
                if parent in seen:
                    raise ValueError("支撑关系存在循环")
                seen.add(parent)
                parent = items[parent].support
            if obj.selected and obj.support != "table" and not items[obj.support].selected:
                raise ValueError("选中物体的支撑物也需要保留")
        return self
