TRUE_COLOR = "#2e9e5b"
FALSE_COLOR = "#d64545"
MID_COLOR = "#e0a100"
BAR_COLOR = "#4f6df5"
TRUE_SCORE_COLOR = "#4f6df5"
PRED_SCORE_COLOR = "#f2994a"

PALETTE = [
    "#4f6df5",
    "#2e9e5b",
    "#f2994a",
    "#d64545",
    "#9b59b6",
    "#16a2a2",
    "#c99a06",
    "#8e44ad",
    "#3498db",
    "#c0392b",
]


def accuracy_color(value: float) -> str:
    return TRUE_COLOR if value >= 0.8 else MID_COLOR if value >= 0.5 else FALSE_COLOR
