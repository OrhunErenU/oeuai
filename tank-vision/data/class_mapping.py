"""Merkezi sinif tanimlari ve eslemeleri.

Tum moduller bu dosyadaki sinif ID'lerini ve isimlerini kullanir.
"""

# Ana detektor siniflari (7 sinif)
PRIMARY_CLASSES = {
    0: "drone",
    1: "tank",
    2: "human",
    3: "weapon",
    4: "vehicle",
    5: "aircraft",
    6: "bird",
}

PRIMARY_CLASS_NAMES = {v: k for k, v in PRIMARY_CLASSES.items()}

# Silah alt-siniflandirma
WEAPON_CLASSES = {
    0: "rpg",
    1: "rifle",
    2: "pistol",
    3: "sniper",
    4: "grenade",
    5: "machine_gun",
}

WEAPON_CLASS_NAMES = {v: k for k, v in WEAPON_CLASSES.items()}

# Tank marka/model siniflandirma
TANK_CLASSES = {
    0: "m1_abrams",
    1: "leopard_2",
    2: "t72",
    3: "t90",
    4: "challenger_2",
    5: "merkava_4",
    6: "altay",
    7: "type_99",
    8: "k2_black_panther",
    9: "unknown",
}

TANK_CLASS_NAMES = {v: k for k, v in TANK_CLASSES.items()}

# Insan alt-siniflandirma
HUMAN_CLASSES = {
    0: "soldier",
    1: "civilian",
}

HUMAN_CLASS_NAMES = {v: k for k, v in HUMAN_CLASSES.items()}

# Dost/Dusman siniflandirma
FOE_CLASSES = {
    0: "friend",
    1: "foe",
    2: "unknown",
}

FOE_CLASS_NAMES = {v: k for k, v in FOE_CLASSES.items()}

# Tehdit seviyeleri
THREAT_LEVELS = {
    0: "none",
    1: "low",
    2: "medium",
    3: "high",
    4: "critical",
}

# COCO -> Ana sinif esleme (download/convert islemleri icin)
COCO_TO_PRIMARY = {
    0: 2,   # person -> human
    2: 4,   # car -> vehicle
    5: 4,   # bus -> vehicle
    7: 4,   # truck -> vehicle
    14: 6,  # bird -> bird
    4: 5,   # airplane -> aircraft
}
