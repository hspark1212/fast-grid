from pathlib import Path

import numpy as np
import pandas as pd


PARENT_DIR = Path(__file__).parent
FF_TYPES = ["UFF"]  # , "GAFF", "OPLS", "MMFF", "MMFF94", "MMFF94s", "MMFF94smod"]


def get_mixing_epsilon_sigma(
    symbols: np.ndarray,
    ff: str,
    gas_epsilon: float,
    gas_sigma: float,
):
    if ff == "UFF":
        df_ff = pd.read_csv(PARENT_DIR / "assets/uff_ff.csv", index_col=0)
    else:
        raise NotImplementedError(f"{ff} should be one of {FF_TYPES}")

    # mixing rules for epsilon and sigma
    epsilons = np.sqrt(df_ff.loc[symbols, "epsilon"].values * gas_epsilon)
    sigmas = (df_ff.loc[symbols, "sigma"].values + gas_sigma) / 2

    return epsilons, sigmas
