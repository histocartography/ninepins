# maximum index of each dataset
DATASET_IDX_LIMITS = {
    "CoNSeP": {
        "train": 27,
        "test": 14
    },
    "MoNuSeg": {
        "train": 16,
        "test": 14
    }
}

# a list of random colors for label2rgb
HEX_LABEL_COLOR_LIST = [
    "#F5F5DC",
    "#B8860B",
    "#4169E1",
    "#6B8E23",
    "#FF00FF",
    "#DAA520",
    "#800080",
    "#B22222",
    "#ADFF2F",
    "#9932CC",
    "#FF4500",
    "#D2B48C",
    "#008080",
    "#708090",
    "#000080",
    "#2F4F4F",
    "#000000",
    "#B0E0E6",
    "#00FFFF"
]

LABEL_COLOR_LIST = [tuple([int(ll[i: i+2], 16) / 255 for i in range(1, 7, 2)]) for ll in HEX_LABEL_COLOR_LIST]