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

# idx to uid mapping for MoNuSeg dataset
MONUSEG_IDX_2_UID = {
    "train": [
        "TCGA-HE-7130-01Z-00-DX1",
        "TCGA-A7-A13F-01Z-00-DX1",
        "TCGA-G9-6348-01Z-00-DX1",
        "TCGA-G9-6336-01Z-00-DX1",
        "TCGA-AR-A1AS-01Z-00-DX1",
        "TCGA-38-6178-01Z-00-DX1",
        "TCGA-HE-7129-01Z-00-DX1",
        "TCGA-A7-A13E-01Z-00-DX1",
        "TCGA-HE-7128-01Z-00-DX1",
        "TCGA-G9-6356-01Z-00-DX1",
        "TCGA-AR-A1AK-01Z-00-DX1",
        "TCGA-G9-6363-01Z-00-DX1",
        "TCGA-18-5592-01Z-00-DX1",
        "TCGA-49-4488-01Z-00-DX1",
        "TCGA-B0-5711-01Z-00-DX1",
        "TCGA-50-5931-01Z-00-DX1"
    ],
    "test": [
        "TCGA-B0-5698-01Z-00-DX1",
        "TCGA-21-5786-01Z-00-DX1",
        "TCGA-RD-A8N9-01A-01-TS1",
        "TCGA-G9-6362-01Z-00-DX1",
        "TCGA-KB-A93J-01A-01-TS1",
        "TCGA-E2-A14V-01Z-00-DX1",
        "TCGA-B0-5710-01Z-00-DX1",
        "TCGA-21-5784-01Z-00-DX1",
        "TCGA-CH-5767-01Z-00-DX1",
        "TCGA-E2-A1B5-01Z-00-DX1",
        "TCGA-DK-A2I6-01A-01-TS1",
        "TCGA-NH-A8F7-01A-01-TS1",
        "TCGA-AY-A8YK-01A-01-TS1",
        "TCGA-G2-A2EK-01A-02-TSB"
    ]
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