import numpy as np
import pandas as pd

from src.indicators import fast_rolling_spearman


def run_sensitivity_grid_search(
    df: pd.DataFrame,
    test_windows: tuple[int, ...] | list[int],
    hold_horizons: tuple[int, ...] | list[int],
    hit_ratio_threshold: float = 0.70,
    spearman_threshold: float = 0.60,
    session_start: str = "09:30",
    session_end: str = "16:00",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate signal robustness across metric windows and holding periods."""
    results_win_rate: dict[int, dict[str, float]] = {}
    results_bps: dict[int, dict[str, float]] = {}
    event_counts: dict[int, int] = {}

    for window in test_windows:
        temp_df = df[["btc_ret", "nq_ret", "nq_close"]].copy()
        temp_df["corr_spearman"] = fast_rolling_spearman(
            temp_df["btc_ret"].to_numpy(),
            temp_df["nq_ret"].to_numpy(),
            window,
        )
        direction_match = (np.sign(temp_df["btc_ret"]) == np.sign(temp_df["nq_ret"])).astype(float)
        temp_df["hit_ratio"] = direction_match.rolling(window=window).mean()

        time_filter = (
            (temp_df.index.time >= pd.to_datetime(session_start).time())
            & (temp_df.index.time <= pd.to_datetime(session_end).time())
        )
        temp_df["valid_signal"] = (
            time_filter
            & (temp_df["hit_ratio"] >= hit_ratio_threshold)
            & (temp_df["corr_spearman"] >= spearman_threshold)
        )
        previous_signal = temp_df["valid_signal"].shift(1, fill_value=False)
        temp_df["event_start"] = temp_df["valid_signal"] & (~previous_signal)
        temp_df["mom_window"] = temp_df["nq_close"] / temp_df["nq_close"].shift(window) - 1
        temp_df["regime_dir"] = np.sign(temp_df["mom_window"]).replace(0, 1)

        for horizon in hold_horizons:
            temp_df[f"nq_fwd_{horizon}"] = temp_df["nq_close"].shift(-horizon) / temp_df["nq_close"] - 1

        events = temp_df[temp_df["event_start"]].copy()
        event_counts[window] = int(len(events))

        win_rate_row: dict[str, float] = {}
        bps_row: dict[str, float] = {}
        for horizon in hold_horizons:
            directional_return = events[f"nq_fwd_{horizon}"] * events["regime_dir"]
            win_rate_row[f"Hold_{horizon}m"] = (directional_return > 0).mean() * 100
            bps_row[f"Hold_{horizon}m"] = directional_return.mean() * 10000

        results_win_rate[window] = win_rate_row
        results_bps[window] = bps_row

    win_rate_matrix = pd.DataFrame.from_dict(results_win_rate, orient="index")
    win_rate_matrix.index.name = "Window_Bars"
    win_rate_matrix.insert(0, "Total_Events", pd.Series(event_counts))

    bps_matrix = pd.DataFrame.from_dict(results_bps, orient="index")
    bps_matrix.index.name = "Window_Bars"
    bps_matrix.insert(0, "Total_Events", pd.Series(event_counts))

    return win_rate_matrix, bps_matrix
