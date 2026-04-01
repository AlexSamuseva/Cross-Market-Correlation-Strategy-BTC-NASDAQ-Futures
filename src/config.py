# ---------------------------------------------------------------------------
# FILE PATHS 
# ---------------------------------------------------------------------------
FILE_NQ  = "/Users/sasha/Documents/MASTERS/Semester 1/ML in Finance 1/Cross-Market Correlation Strategy - BTC & NASDAQ Futures/datasets/NQ_stitched_1min_2024-03-07_to_2026-03-07.csv"
FILE_BTC = "/Users/sasha/Documents/MASTERS/Semester 1/ML in Finance 1/Cross-Market Correlation Strategy - BTC & NASDAQ Futures/datasets/BTCUSDT_1m_2024-03-07_to_2026-03-07.csv"
# ---------------------------------------------------------------------------
# SIMULATION CONSTANTS (The Physics of the Backtest)
# ---------------------------------------------------------------------------
TZ_OFFSET_HOURS      = 1        # UTC+1 (Warsaw winter = CET) - Needed for pandas pipeline
PENALTY_PER_CONTRACT = 1.5      # NQ pts per contract round-trip (spread + fees)
INITIAL_EQUITY       = 100_000  # Account starting balance

# ---------------------------------------------------------------------------
# DEFAULT PARAMETER SET (Fully Optimizable)
# These are your starting baselines. Every single one of these can now 
# be overwritten dynamically during your grid search stages.
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # --- Time & Session Rules ---
    SESSION_START_H    = 15,     # 15:30 CET = 09:30 ET
    SESSION_START_M    = 30,
    EOD_H              = 21,     # Hard EOD exit (21:55 CET = 15:55 ET)
    EOD_M              = 55,
    
    # --- Core Signal Rules ---
    CORR_WINDOW        = 15,
    CORR_THRESHOLD     = 0.95,
    MOM_LOOKBACK       = 30,
    STD_THRESHOLD      = 2.0,
    
    # --- Trade Management ---
    MIN_HOLD_BARS      = 5,
    MAX_HOLD_BARS      = 60,
    TRAIL_ACTIVATE_PTS = 9999,   # 9999 = disabled sentinel
    TRAIL_DISTANCE_PTS = 15,
    
    # --- Filters & Cooldowns ---
    MAX_DAILY_TRADES   = 999,    # 999 = disabled sentinel
    LOSS_COOLDOWN_PTS  = 20,
    LOSS_COOLDOWN_BARS = 0,      # 0 = disabled sentinel
    TREND_SMA_BARS     = 0,      # 0 = disabled sentinel
    DOWNTREND_SMA_BARS = 0,      # 0 = disabled sentinel
    GAP_FILTER_PCT     = 0.002,  
    COOLDOWN_BARS      = 10,     
    
    # --- Position Sizing ---
    VOL_TARGET         = 0.0,    # 0 = fixed contracts mode
    ATR_WINDOW         = 14,
    MIN_CONTRACTS      = 1,
    MAX_CONTRACTS      = 40,
    FIXED_CONTRACTS    = 1,
)

# ---------------------------------------------------------------------------
# PRE-COMPUTATION ARRAYS (Indicator Grids)
# Defines the boundaries for data pre-calculated in src/indicators.py.
# If you want to test a SMA of 300, you must add it to this list first!
# ---------------------------------------------------------------------------
CORR_WINDOWS_ALL    = [10, 15, 20, 30]
MOM_LOOKBACKS_ALL   = [10, 15, 20, 30, 45]
SMA_BARS_ALL        = [0, 30, 60, 100, 120, 200]     
ATR_WINDOWS_ALL     = [10, 14, 20]

print("src/config.py loaded successfully. Ready for dynamic optimization.")