# web.py
import os
import time
import glob

import streamlit as st
import pandas as pd
import chess
import chess.pgn
import chess.svg

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None

st.set_page_config(page_title="RL Chess Training Dashboard", layout="wide")

st.title("RL Chess Engine – Training Dashboard")

# ==============
# Settings
# ==============
colA, colB, colC = st.columns(3)
with colA:
    logdir = st.text_input("TensorBoard logdir", value="runs")
with colB:
    pgn_path = st.text_input("PGN-fil (senaste spelet)", value="last_game.pgn")
with colC:
    auto_refresh = st.checkbox("Auto-refresh", value=True)

refresh_seconds = st.slider("Refresh-intervall (sek)", 1, 30, 3)

# ==============
# Helpers
# ==============
def find_latest_event_file(logdir: str) -> str | None:
    # TensorBoard event files look like: events.out.tfevents....
    files = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def load_scalars_from_event(event_path: str):
    if EventAccumulator is None:
        return {}

    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = pd.DataFrame({
            "step": [e.step for e in events],
            "value": [e.value for e in events],
            "wall_time": [e.wall_time for e in events],
        })
    return out

def load_last_game(pgn_path: str):
    if not os.path.exists(pgn_path):
        return None
    with open(pgn_path, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    return game

def game_to_board_positions(game: chess.pgn.Game):
    board = game.board()
    positions = [board.copy()]
    moves = []
    for move in game.mainline_moves():
        moves.append(move)
        board.push(move)
        positions.append(board.copy())
    return positions, moves

# ==============
# Training metrics (TensorBoard logs)
# ==============
st.subheader("Träning (från TensorBoard-loggar)")
event_file = find_latest_event_file(logdir)

if event_file is None:
    st.warning(f"Hittar ingen TensorBoard event-fil i '{logdir}'. Starta träningen med --logdir {logdir}.")
else:
    st.caption(f"Senaste event-fil: {event_file}")
    scalars = load_scalars_from_event(event_file)

    # Show the most important graphs if present
    metric_tags = [
        "train/loss_total",
        "train/loss_policy",
        "train/loss_value",
        "data/replay_size",
        "data/samples_added",
    ]

    cols = st.columns(2)
    for i, tag in enumerate(metric_tags):
        if tag in scalars and len(scalars[tag]) > 0:
            with cols[i % 2]:
                st.line_chart(scalars[tag].set_index("step")["value"], height=200)
                st.caption(tag)

    # Raw table (optional)
    with st.expander("Visa senaste värden (tabell)"):
        rows = []
        for tag, df in scalars.items():
            if len(df) > 0:
                rows.append({"tag": tag, "last_step": int(df["step"].iloc[-1]), "last_value": float(df["value"].iloc[-1])})
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values(["tag"]))
        else:
            st.write("Inga scalar-metrics hittades i event-filen.")

# ==============
# Latest game viewer (PGN)
# ==============
st.subheader("Senaste spelet (PGN)")

game = load_last_game(pgn_path)
if game is None:
    st.info(
        f"Hittar ingen '{pgn_path}'. Kör träningen med flaggan `--save-last-pgn` "
        f"så uppdateras filen automatiskt."
    )
else:
    headers = dict(game.headers)
    col1, col2 = st.columns([2, 3])

    with col1:
        st.write("**Headers**")
        st.json(headers)

        positions, moves = game_to_board_positions(game)
        max_idx = len(positions) - 1
        idx = st.slider("Drag", 0, max_idx, max_idx)

        b = positions[idx]
        st.write(f"**Position efter {idx} ply** | Tur: {'Vit' if b.turn else 'Svart'}")
        svg = chess.svg.board(b, size=420)
        st.image(svg)

    with col2:
        st.write("**Moves**")
        # Build a simple move list
        san_list = []
        btmp = game.board()
        for m in moves:
            san_list.append(btmp.san(m))
            btmp.push(m)

        # Pretty-ish formatting
        lines = []
        for i in range(0, len(san_list), 2):
            move_no = i // 2 + 1
            w = san_list[i]
            bl = san_list[i + 1] if i + 1 < len(san_list) else ""
            lines.append(f"{move_no}. {w} {bl}".strip())

        st.code("\n".join(lines), language="text")

# ==============
# Auto refresh
# ==============
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
