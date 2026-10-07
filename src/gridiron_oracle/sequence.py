"""A GRU over REAL sequences: each team's previous N games (extra ``sequence``; torch is lazy).

For a game, the home sequence and the away sequence are the stat vectors of each team's last N games
that ended before the game date (zero padding and a mask flag for shorter histories). One shared GRU
encodes both sequences; a linear layer on [home code, away code, Elo difference] gives the home-win
logit. Early stopping uses the validation season; the seed is fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Model

SEQ_STATS = ("points_for", "points_against", "yards", "yards_allowed", "giveaways", "takeaways", "win", "is_home")


def build_sequences(long: pd.DataFrame, n_last: int = 8) -> dict[tuple[str, str], np.ndarray]:
    """(game_id, team) -> (n_last, len(SEQ_STATS) + 1) array of the team's EARLIER games, oldest first."""
    out = {}
    for team, g in long.sort_values(["game_date", "game_id"]).groupby("team", sort=False):
        vals = g[list(SEQ_STATS)].to_numpy(dtype=np.float32)
        for k, gid in enumerate(g.game_id.to_numpy()):
            seq = np.zeros((n_last, len(SEQ_STATS) + 1), dtype=np.float32)
            hist = vals[max(0, k - n_last) : k]  # strictly before game k
            if len(hist):
                seq[-len(hist):, :-1] = hist
                seq[-len(hist):, -1] = 1.0
            out[(gid, team)] = seq
    return out


class GRUModel(Model):
    name = "gru"

    def __init__(self, long: pd.DataFrame, n_last: int = 8, hidden: int = 16, epochs: int = 40, lr: float = 3e-3,
                 patience: int = 5, seed: int = 0):
        self.seqs = build_sequences(long, n_last)
        self.hidden, self.epochs, self.lr, self.patience, self.seed = hidden, epochs, lr, patience, seed

    def _tensors(self, df: pd.DataFrame):
        import torch

        h = np.stack([self.seqs[(g, t)] for g, t in zip(df.game_id, df.home_team)])
        a = np.stack([self.seqs[(g, t)] for g, t in zip(df.game_id, df.away_team)])
        h[..., :-1] = (h[..., :-1] - self.mu) / self.sd * h[..., -1:]
        a[..., :-1] = (a[..., :-1] - self.mu) / self.sd * a[..., -1:]
        e = ((df.elo_home_pre - df.elo_away_pre).to_numpy(dtype=np.float32) / 100.0)[:, None]
        return torch.tensor(h), torch.tensor(a), torch.tensor(e)

    def fit(self, train, val=None):
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError('the GRU needs: pip install -e ".[sequence]"') from exc
        torch.manual_seed(self.seed)
        stack = np.concatenate([self.seqs[(g, t)] for g, t in zip(train.game_id, train.home_team)])
        real = stack[stack[:, -1] == 1, :-1]
        self.mu, self.sd = real.mean(axis=0), real.std(axis=0) + 1e-6
        hidden = self.hidden

        class Net(nn.Module):
            def __init__(self, n_in):
                super().__init__()
                self.gru = nn.GRU(n_in, hidden, batch_first=True)
                self.out = nn.Linear(2 * hidden + 1, 1)

            def forward(self, h, a, e):
                _, hh = self.gru(h)
                _, ha = self.gru(a)
                return self.out(torch.cat([hh[-1], ha[-1], e], dim=1)).squeeze(1)

        H, A, E = self._tensors(train)
        y = torch.tensor(train.home_win.to_numpy(dtype=np.float32))
        net = Net(H.shape[2])
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        has_val = val is not None and len(val) > 0
        if has_val:
            Hv, Av, Ev = self._tensors(val)
            yv = torch.tensor(val.home_win.to_numpy(dtype=np.float32))
        best, state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            net.train()
            for idx in torch.randperm(len(y), generator=g).split(128):
                opt.zero_grad()
                loss_fn(net(H[idx], A[idx], E[idx]), y[idx]).backward()
                opt.step()
            if has_val:
                net.eval()
                with torch.no_grad():
                    v = float(loss_fn(net(Hv, Av, Ev), yv))
                if v < best - 1e-4:
                    best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if state is not None:
            net.load_state_dict(state)
        self.net = net.eval()
        return self

    def predict_proba(self, df):
        import torch

        with torch.no_grad():
            return torch.sigmoid(self.net(*self._tensors(df))).numpy().astype(float)
