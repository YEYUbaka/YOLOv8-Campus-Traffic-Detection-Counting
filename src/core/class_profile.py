"""
模型类别配置。
将当前模型的 names 映射成交通场景下统一的中文展示名和业务类别集合。
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Mapping, Optional, Set


TRAFFIC_CLASS_ORDER = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "tricycle",
]

DISPLAY_NAME_MAP = {
    "person": "行人",
    "bicycle": "自行车",
    "car": "轿车",
    "motorcycle": "摩托车",
    "bus": "公交车",
    "truck": "卡车",
    "tricycle": "三轮车",
}

CLASS_COLOR_MAP = {
    "person": (255, 0, 0),
    "bicycle": (0, 255, 255),
    "car": (0, 255, 0),
    "motorcycle": (255, 0, 255),
    "bus": (0, 128, 255),
    "truck": (128, 0, 128),
    "tricycle": (255, 255, 0),
}

NAME_ALIASES = {
    "person": "person",
    "pedestrian": "person",
    "people": "person",
    "walker": "person",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cycle": "bicycle",
    "cyclist": "bicycle",
    "car": "car",
    "auto": "car",
    "automobile": "car",
    "sedan": "car",
    "vehicle": "car",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "motor cycle": "motorcycle",
    "moped": "motorcycle",
    "bus": "bus",
    "coach": "bus",
    "truck": "truck",
    "lorry": "truck",
    "pickup": "truck",
    "pickup truck": "truck",
    "tricycle": "tricycle",
    "trike": "tricycle",
    "three wheeler": "tricycle",
    "three wheel": "tricycle",
    "three wheeled vehicle": "tricycle",
}


@dataclass(frozen=True)
class TrafficClassProfile:
    raw_names: Dict[int, str]
    canonical_names: Dict[int, str]
    display_names: Dict[int, str]
    class_colors: Dict[int, tuple]
    traffic_class_ids: Set[int]
    vehicle_class_ids: Set[int]
    table_class_ids: List[int]
    table_display_names: List[str]
    classes_filter: List[int]

    def display_name_for(self, class_id: int) -> str:
        return self.display_names.get(class_id, f"未知({class_id})")

    def canonical_name_for(self, class_id: int) -> Optional[str]:
        return self.canonical_names.get(class_id)


def normalize_class_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None

    normalized = re.sub(r"[_-]+", " ", str(name).strip().lower())
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in NAME_ALIASES:
        return NAME_ALIASES[normalized]

    singular = normalized[:-1] if normalized.endswith("s") else normalized
    return NAME_ALIASES.get(singular)


def _coerce_model_names(model_names: object) -> Dict[int, str]:
    if isinstance(model_names, Mapping):
        return {int(key): str(value) for key, value in model_names.items()}

    if isinstance(model_names, Iterable) and not isinstance(model_names, (str, bytes)):
        return {index: str(name) for index, name in enumerate(model_names)}

    return {}


def build_traffic_class_profile(model_names: object) -> TrafficClassProfile:
    raw_names = _coerce_model_names(model_names)
    canonical_names: Dict[int, str] = {}
    display_names: Dict[int, str] = {}
    class_colors: Dict[int, tuple] = {}
    traffic_class_ids: Set[int] = set()
    vehicle_class_ids: Set[int] = set()

    for class_id, raw_name in raw_names.items():
        canonical_name = normalize_class_name(raw_name)
        if canonical_name is None:
            continue

        canonical_names[class_id] = canonical_name
        display_names[class_id] = DISPLAY_NAME_MAP.get(canonical_name, raw_name)
        class_colors[class_id] = CLASS_COLOR_MAP.get(canonical_name, (0, 255, 0))
        traffic_class_ids.add(class_id)
        if canonical_name != "person":
            vehicle_class_ids.add(class_id)

    table_class_ids: List[int] = []
    table_display_names: List[str] = []
    for canonical_name in TRAFFIC_CLASS_ORDER:
        for class_id, current_name in canonical_names.items():
            if current_name == canonical_name:
                table_class_ids.append(class_id)
                table_display_names.append(display_names[class_id])
                break

    return TrafficClassProfile(
        raw_names=raw_names,
        canonical_names=canonical_names,
        display_names=display_names,
        class_colors=class_colors,
        traffic_class_ids=traffic_class_ids,
        vehicle_class_ids=vehicle_class_ids,
        table_class_ids=table_class_ids,
        table_display_names=table_display_names,
        classes_filter=sorted(traffic_class_ids),
    )


DEFAULT_TRAFFIC_CLASS_PROFILE = build_traffic_class_profile({
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "truck",
    6: "tricycle",
})
