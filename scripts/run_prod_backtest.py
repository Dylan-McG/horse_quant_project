from pathlib import Path
import json, time
from hqp.eval.backtest import run as bt_run

CFG = {
  "edge_threshold": 0.14,
  "max_odds": 15.0,
  "per_race_max_bets": 1,
  "kelly_fraction": 0.10,
  "bankroll_init": 1000.0,
  "stake_max": 20.0,
  "stake_min": 0.0,
  "try_join_market": True,
}

edges = Path("data/market/edges_cal.parquet")
out = Path(bt_run(edges, CFG))
s = json.loads((out/"summary.json").read_text(encoding="utf-8"))
print(f"[PROD] bets={s['bets']} roi={s['roi']:.3f} pnl={s['pnl']:.2f} -> {out}")

# append a single-line log for quick glance/history
log = Path("reports/backtest/prod_log.csv")
log.parent.mkdir(parents=True, exist_ok=True)
header = "ts,dir,bets,roi,pnl,edge,min_odds,max_odds,per_race,kelly,bankroll,stake_max,stake_min\n"
line = f"{time.strftime('%Y-%m-%d %H:%M:%S')},{out.name},{s['bets']},{s['roi']},{s['pnl']}," \
       f"{CFG['edge_threshold']},1.0,{CFG['max_odds']},{CFG['per_race_max_bets']}," \
       f"{CFG['kelly_fraction']},{CFG['bankroll_init']},{CFG['stake_max']},{CFG['stake_min']}\n"
if not log.exists():
    log.write_text(header, encoding="utf-8")
with log.open("a", encoding="utf-8") as f:
    f.write(line)
