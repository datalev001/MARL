"""
Meta-Control + DQN for Multi-Channel Ad Bidding
================================================
•  reads    C:\backupcgi\final_bak\MAML_adv.csv   (115 209 rows)
•  two–level agent
      Meta-Controller (PPO)        – every hour chooses an "option"
      Option 0  : cost-control DQN – bids conservatively
      Option 1  : volume-boost DQN – bids aggressively
•  bottom DQN acts every 15 min inside its option window
•  reward  = gross_profit − overspend_penalty
•  logs KPIs and plots learning curves
------------------------------------------------
pip install gymnasium stable-baselines3 pandas numpy matplotlib
"""

import os, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

CSV_PATH = Path("MAML_adv.csv")
assert CSV_PATH.exists(), "Generate the CSV first!"
df_raw = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
print("✅ data loaded:", df_raw.shape)

# -------------------- hyper-params ---------------------------------
CLICK_VALUE   = 1.4        # $ revenue per click
DAILY_BUDGET  = 800        # $ per channel per day
PENALTY       = 0.5        # $ per $ overspend
OPTION_HOURS  = 1          # meta decision granularity
WIN_RATE_LOW  = 0.30       # cost-control avg win prob
WIN_RATE_HIGH = 0.55       # volume-boost avg win prob
# -------------------------------------------------------------------

# ---------- Env that supports two levels --------------------
class AdEnv(gym.Env):
    """
    One step = one 15-min slot for a single channel.
    Option k defines the bid multiplier range.
        k = 0 →  [0.6 , 1.0] * base_cpc
        k = 1 →  [1.0 , 1.6] * base_cpc
    DQN action = 3 discrete multipliers inside the range.
    """
    def __init__(self, df: pd.DataFrame, channel: str):
        super().__init__()
        self.df = df[df.channel == channel].sort_values("timestamp").reset_index(drop=True)
        self.ptr = 0
        self.day_spend = 0.0
        self.channel = channel
        feat_cols = ["ctr", "cpc", "impression", "budget_left_pct", "hour"]
        self.mu = self.df[feat_cols].mean()
        self.sig = self.df[feat_cols].std().replace(0, 1)
        self.observation_space = spaces.Box(low=-4, high=4, shape=(len(feat_cols)+1,), dtype=np.float32)
        self.action_space      = spaces.Discrete(3)   # low/med/high multiplier
        self.curr_option = 0  # set externally by Meta agent

    # ---- helper
    def _norm_state(self, row):
        num = ((row[["ctr","cpc","impression","budget_left_pct","hour"]] - self.mu) / self.sig).to_numpy()
        return np.concatenate([num, [self.curr_option]]).astype(np.float32)
    
    def current_obs(self):
        """Return the normalised observation for the *current* pointer."""
        row = self.df.iloc[self.ptr]
        return self._norm_state(row)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = 0
        self.day_spend = 0.0
        row = self.df.loc[self.ptr]
        return self._norm_state(row), {}

    def step(self, action:int):
        row = self.df.loc[self.ptr]
        # option-specific bid multiplier
        rng_low, rng_high = (0.6, 1.0) if self.curr_option==0 else (1.0, 1.6)
        mult = np.linspace(rng_low, rng_high, 3)[action]
        effective_cpc = row.cpc * mult
        clicks = row.ctr * row.impression * (WIN_RATE_LOW if self.curr_option==0 else WIN_RATE_HIGH)
        cost   = clicks * effective_cpc
        self.day_spend += cost
        profit = clicks * CLICK_VALUE - cost
        overspend_pen = PENALTY * max(0, self.day_spend - DAILY_BUDGET)
        reward = profit - overspend_pen

        done = False
        self.ptr += 1
        if self.ptr >= len(self.df):
            done = True
        state = self._norm_state(self.df.loc[self.ptr]) if not done else np.zeros(self.observation_space.shape, np.float32)
        info  = dict(profit=profit, cost=cost, day_spend=self.day_spend, option=self.curr_option)
        return state, reward, done, False, info

# ---------- Meta-Env wraps three channels & Options -------------
class MetaEnv(gym.Env):
    """
    Observation  (9-dim float32):
        [ctr_s , cpc_s , budget_s ,
         ctr_v , cpc_v , budget_v ,
         ctr_f , cpc_f , budget_f ]
        (s=search, v=video, f=feed)

    Meta-action (int):
        0  →  pick Option-0  "cost control"  for all channels this hour
        1  →  pick Option-1  "volume boost"

    One meta-step  =  four 15-minute inner steps × 3 channels
    Reward         =  sum of channel profits – overspend penalty
    """
    def __init__(self, df):
        super().__init__()
        self.channels = ["search", "video", "feed"]
        # base env for each channel (defined earlier as AdEnv)
        self.envs = {ch: AdEnv(df, ch) for ch in self.channels}

        # option-specific DQN agents ──  {channel: {option: dqn}}
        self.dqns = {ch: {0: None, 1: None} for ch in self.channels}
        self._build_dqns()

        # meta spaces
        self.observation_space = spaces.Box(low=-5, high=5, shape=(9,), dtype=np.float32)
        self.action_space      = spaces.Discrete(2)   # choose option id
        self.ticks_per_meta    = 4                    # four 15-min slots ⇒ 1 h

    # ---------------------------------------------------------
    #  build per-channel DQN agents for both options
    # ---------------------------------------------------------
    def _build_dqns(self):
        for ch in self.channels:
            for opt in (0, 1):
                env_fn = lambda ch=ch: Monitor(self._make_subenv(ch))
                self.dqns[ch][opt] = DQN(
                    "MlpPolicy",
                    DummyVecEnv([env_fn]),
                    learning_rate=1e-3,
                    buffer_size=20_000,
                    gamma=0.99,
                    exploration_fraction=0.1,
                    verbose=0
                )

    # wrap AdEnv so that the meta layer can modify env.curr_option
    def _make_subenv(self, channel):
        base_env = self.envs[channel]

        class OptionWrapper(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
        return OptionWrapper(base_env)

    # ---------------------------------------------------------
    #  construct 9-dim meta state from current pointers
    # ---------------------------------------------------------
    def _gather_meta_state(self):
        vec = []
        for ch in self.channels:
            e   = self.envs[ch]
            row = e.df.iloc[e.ptr]
            vec.extend([
                (row.ctr - e.mu.ctr)   / e.sig.ctr,
                (row.cpc - e.mu.cpc)   / e.sig.cpc,
                row.budget_left_pct
            ])
        return np.array(vec, dtype=np.float32)

    # ---------------------------------------------------------
    #  standard Gym API
    # ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for env in self.envs.values():
            env.reset()
        return self._gather_meta_state(), {}

    def step(self, meta_action: int):
        # 1) broadcast chosen option to all channels
        for env in self.envs.values():
            env.curr_option = meta_action

        cum_reward = 0.0
        infos      = []

        # 2) run 4 inner 15-min steps
        for _ in range(self.ticks_per_meta):
            for ch in self.channels:
                env  = self.envs[ch]
                obs  = env.current_obs()                 # guaranteed (6,)
                act, _ = self.dqns[ch][meta_action].predict(obs, deterministic=False)
                _, r, done, _, info = env.step(int(act))
                self.dqns[ch][meta_action].learn(total_timesteps=1)  # single online update
                cum_reward += r
                infos.append(info)

        next_state = self._gather_meta_state()
        # meta-episode never terminates in this toy demo
        return next_state, cum_reward, False, False, {"details": infos}

# ---------- Training -------------------------------------------
meta_env = DummyVecEnv([lambda: Monitor(MetaEnv(df_raw))])
meta_agent = PPO("MlpPolicy", meta_env, learning_rate=2e-4,
                 n_steps=16, batch_size=16, clip_range=0.2,
                 gamma=0.99, verbose=1)

class MetaLogger(BaseCallback):
    def __init__(self): super().__init__(); self.rew=[]
    def _on_step(self): 
        if len(self.locals["infos"])>0:
            self.rew.append(self.locals["rewards"][0])
        return True

logger = MetaLogger()
print("⏳ Training Meta-Controller (PPO)…")
meta_agent.learn(total_timesteps=5_000, callback=logger)

# ---------- 4. Quick evaluation plot ------------------------------
plt.plot(pd.Series(logger.rew).rolling(20).mean())
plt.title("Meta-Controller reward (smoothed)")
plt.xlabel("meta-step (≈1h)"); plt.ylabel("ROI reward")
plt.tight_layout(); plt.show()

#################adding result: business############################
"""
7-Day Offline Replay for Business KPI Report
===========================================

• Replays the trained Meta-Controller for 7×24 h×4 = 672 meta-steps
• Aggregates every 15-min inner-slot info into a single DataFrame
• Plots rolling ROI (12 inner slots ≈ 1 h) for visual trend
• Prints a plain-English KPI table + option-usage share that
  business stakeholders can read at a glance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ------------------------------------------------------------------
# Simulate policy for 7 days
# ------------------------------------------------------------------
def simulate_policy(meta_agent, env, n_steps=7 * 24 * 4):
    """
    Roll the vectorised Meta-Controller for `n_steps` meta-iterations
    (each ≈ 1 hour). Return a DataFrame with every inner-slot KPI row.
    """
    obs = env.reset()        # DummyVecEnv → ndarray (1, obs_dim)
    kpi_rows = []

    for _ in range(n_steps):
        act, _ = meta_agent.predict(obs, deterministic=True)   # ndarray (1,)
        obs, rewards, dones, infos = env.step(act)             # 4-tuple
        # infos[0]["details"] is the list of dicts returned by inner envs
        kpi_rows.extend(infos[0]["details"])

    return pd.DataFrame(kpi_rows)

# ---------- run replay (uses meta_agent & meta_env from training) ---
kpi_df = simulate_policy(meta_agent, meta_env)

# ------------------------------------------------------------------
# Rolling ROI curve (business-friendly)
# ------------------------------------------------------------------
kpi_df["hour_idx"] = np.arange(len(kpi_df)) / 3          # 3 channels → 1 hourly index
roi_by_hour = ((kpi_df.profit + kpi_df.cost)
               .rolling(12).sum() /
               kpi_df.cost.rolling(12).sum())

roi_by_hour.plot(title="Rolling ROI  (12 inner slots ≈ 1 h)")
plt.ylabel("ROI")
plt.xlabel("hour index")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# KPI summary table & option usage share
# ------------------------------------------------------------------
kpi_df["revenue"] = kpi_df["profit"] + kpi_df["cost"]

summary = pd.DataFrame({
    "spend"        : [kpi_df["cost"].sum()],
    "revenue"      : [kpi_df["revenue"].sum()],
    "gross_profit" : [kpi_df["profit"].sum()]
})
summary["ROI"] = summary["revenue"] / summary["spend"]
summary.index = ["value"]

print("\n=== 7-Day Business KPI ===")
print(summary.round(2))

print("\nOption usage (%):")
option_share = kpi_df["option"].value_counts(normalize=True).mul(100).round(1)
for opt, pct in option_share.items():
    tag = "cost-control" if opt == 0 else "volume-boost"
    print(f"{opt:<1}    {pct:4.1f}%   ← {tag}")