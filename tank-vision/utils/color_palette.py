"""Sinif renk paleti tanimlari (HUD goruntusu icin)."""

# BGR renkleri (OpenCV formati)
CLASS_COLORS = {
    "drone": (0, 165, 255),       # Turuncu
    "tank": (0, 0, 255),          # Kirmizi
    "human": (0, 255, 0),         # Yesil
    "weapon": (0, 0, 200),        # Koyu kirmizi
    "vehicle": (255, 255, 0),     # Cyan
    "aircraft": (255, 0, 255),    # Magenta
    "bird": (200, 200, 200),      # Gri
}

THREAT_COLORS = {
    0: (200, 200, 200),   # Gri - none
    1: (0, 255, 0),       # Yesil - low
    2: (0, 255, 255),     # Sari - medium
    3: (0, 165, 255),     # Turuncu - high
    4: (0, 0, 255),       # Kirmizi - critical
}

FOE_COLORS = {
    "friend": (0, 255, 0),    # Yesil
    "foe": (0, 0, 255),       # Kirmizi
    "unknown": (0, 255, 255), # Sari
}
