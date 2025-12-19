"""
Module contains target information for docking evaluation.
"""
from pathlib import Path

# Get the directory of the current file
CURRENT_DIR = Path(__file__).parent

# Ligands from PDB
## Docking information from TDC
## https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py
docking_target_info = {
    "1iep": {
        "center": (15.61389189189189, 53.38013513513513, 15.454837837837842),
        "size": (15, 15, 15),
        "ligand": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C", # STI, Imatinib
        "pH": 6.5,
        "pdbqt": CURRENT_DIR / 'pdbs/1iep.pdbqt'
    },
    "3eml": {
        "center": (-9.063639999999998, -7.1446, 55.86259999999999),
        "size": (15, 15, 15),
        "ligand": "c1cc(oc1)c2nc3nc(nc(n3n2)N)NCCc4ccc(cc4)O", # ZMA
        "pH": 6.5,
        "pdbqt": CURRENT_DIR / 'pdbs/3eml.pdbqt'
    },
    "3ny8": {
        "center": (2.2488, 4.68495, 51.39820000000001),
        "size": (15, 15, 15),
        "ligand": "Cc1ccc(c2c1CCC2)OC[C@H]([C@H](C)NC(C)C)O", # JRZ
        "pH": 7.5,
        "pdbqt": CURRENT_DIR / 'pdbs/3ny8.pdbqt'
    },
    "4rlu": {
        "center": (-0.7359999999999999, 22.75547368421052, -31.2368947368421),
        "size": (15, 15, 15),
        "ligand": "c1cc(ccc1C=CC(=O)c2ccc(cc2O)O)O", # HCC
        "pH": 7.5,
        "pdbqt": CURRENT_DIR / 'pdbs/4rlu.pdbqt'
    },
    "4unn": {
        "center": (5.684346153846153, 18.191769230769232, -7.37157692307692),
        "size": (15, 15, 15),
        "ligand": "COc1cccc(c1)C2=CC(=C(C(=O)N2)C#N)c3ccc(cc3)C(=O)O", # QZZ
        "pH": 7.4, # non reported
        "pdbqt": CURRENT_DIR / 'pdbs/4unn.pdbqt'
    },
    "5mo4": {
        "center": (-44.901709677419355, 20.490354838709674, 8.483354838709678),
        "size": (15, 15, 15),
        "ligand": "c1cc(ccc1NC(=O)c2cc(c(nc2)N3CC[C@H](C3)O)c4ccn[nH]4)OC(F)(F)Cl", # AY7, asciminib
        "pH": 7.5,
        "pdbqt": CURRENT_DIR / 'pdbs/5mo4.pdbqt'
    },
    "7l11": {
        "center": (-21.814812500000006, -4.216062499999999, -27.983781250000),
        "size": (15, 15, 15),
        "ligand": "CCCOc1cc(cc(c1)Cl)C2=CC(=CN(C2=O)c3cccnc3)c4ccccc4C#N", # XF1
        "pH": 6.0,
        "pdbqt": CURRENT_DIR / 'pdbs/7l11.pdbqt'
    },
}
