# train_rl.py
from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import chess
import chess.engine
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from core import (
    PVNet, ReplayBuffer, Sample,
    MOVE_SPACE_SIZE, board_to_tensor, result_white,
    mcts_search, visits_to_policy, index_to_move
)

@dataclass
class Config:
    device: str = "cpu"          # "cuda" om du har NVIDIA
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    train_steps: int = 2000

    # data
    selfplay_games_per_iter: int = 10
    replay_max_size: int = 200_000
    max_game_moves: int = 300

    # temp
    temperature_moves: int = 20
    temperature: float = 1.0

    # mcts
    mcts_sims: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # engine sparring
    sparring_ratio: float = 0.2     # 0.0 = bara self-play
    uci_engine_path: str = ""       # eller env UCI_ENGINE_PATH
    uci_engine_depth: int = 6

    # checkpoint
    ckpt_dir: str = "checkpoints"
    save_every: int = 200

    # visual/logging
    log_dir: str = "runs"
    log_every: int = 20
    save_last_pgn: bool = False
    pgn_path: str = "last_game.pgn"

class UCIEngine:
    def __init__(self, path: str, depth: int):
        if not path:
            raise ValueError("Tom UCI-engine-path. Sätt cfg.uci_engine_path eller env UCI_ENGINE_PATH.")
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth = depth

    def move(self, board: chess.Board) -> chess.Move:
        r = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        return r.move

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

def pick_move_from_pi(board: chess.Board, pi: np.ndarray) -> chess.Move:
    if np.sum(pi) <= 0:
        return random.choice(list(board.legal_moves))
    a = int(np.random.choice(np.arange(len(pi)), p=pi))
    m = index_to_move(a)
    if m not in board.legal_moves:
        return random.choice(list(board.legal_moves))
    return m

def _maybe_save_last_pgn(cfg: Config, moves_uci: List[str]):
    if not cfg.save_last_pgn:
        return
    try:
        import chess.pgn
        b = chess.Board()
        game = chess.pgn.Game()
        node = game
        for u in moves_uci:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                break
            node = node.add_variation(m)
            b.push(m)
        game.headers["Result"] = b.result(claim_draw=True)
        with open(cfg.pgn_path, "w", encoding="utf-8") as f:
            print(game, file=f, end="\n\n")
    except Exception:
        pass

def self_play_game(model: PVNet, cfg: Config) -> List[Sample]:
    board = chess.Board()
    states, pis, turns = [], [], []
    moves_uci = []

    for ply in range(cfg.max_game_moves):
        if board.is_game_over(claim_draw=True):
            break

        turns.append(board.turn)
        states.append(board_to_tensor(board))

        root = mcts_search(
            model=model,
            root_board=board,
            device=cfg.device,
            sims=cfg.mcts_sims,
            c_puct=cfg.c_puct,
            dir_alpha=cfg.dirichlet_alpha,
            dir_eps=cfg.dirichlet_epsilon,
            add_noise=True
        )

        temp = cfg.temperature if ply < cfg.temperature_moves else 1e-9
        pi = visits_to_policy(root, temperature=temp)
        pis.append(torch.tensor(pi, dtype=torch.float32))

        move = pick_move_from_pi(board, pi)
        moves_uci.append(move.uci())
        board.push(move)

    _maybe_save_last_pgn(cfg, moves_uci)

    zw = result_white(board)
    samples: List[Sample] = []
    for s, pi_t, turn in zip(states, pis, turns):
        z = zw if turn == chess.WHITE else -zw
        samples.append((s, pi_t, float(z)))
    return samples

def game_vs_engine(model: PVNet, cfg: Config, engine: UCIEngine) -> List[Sample]:
    board = chess.Board()
    our_color = random.choice([chess.WHITE, chess.BLACK])
    saved: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for ply in range(cfg.max_game_moves):
        if board.is_game_over(claim_draw=True):
            break

        if board.turn == our_color:
            root = mcts_search(
                model=model,
                root_board=board,
                device=cfg.device,
                sims=cfg.mcts_sims,
                c_puct=cfg.c_puct,
                dir_alpha=cfg.dirichlet_alpha,
                dir_eps=cfg.dirichlet_epsilon,
                add_noise=False
            )
            temp = cfg.temperature if ply < cfg.temperature_moves else 1e-9
            pi = visits_to_policy(root, temperature=temp)
            pi_t = torch.tensor(pi, dtype=torch.float32)
            saved.append((board_to_tensor(board), pi_t))

            move = pick_move_from_pi(board, pi)
            board.push(move)
        else:
            move = engine.move(board)
            if move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))
            board.push(move)

    zw = result_white(board)
    z = zw if our_color == chess.WHITE else -zw
    return [(s, pi, float(z)) for (s, pi) in saved]

def generate_samples(model: PVNet, cfg: Config) -> List[Sample]:
    model.eval()
    out: List[Sample] = []

    engine = None
    engine_path = cfg.uci_engine_path or os.getenv("UCI_ENGINE_PATH", "")
    if cfg.sparring_ratio > 0 and engine_path:
        engine = UCIEngine(engine_path, cfg.uci_engine_depth)

    try:
        for _ in range(cfg.selfplay_games_per_iter):
            if engine and random.random() < cfg.sparring_ratio:
                out.extend(game_vs_engine(model, cfg, engine))
            else:
                out.extend(self_play_game(model, cfg))
    finally:
        if engine:
            engine.close()

    return out

def train_step(model: PVNet, opt, batch: List[Sample], device: str):
    model.train()
    states, pis, zs = zip(*batch)
    x = torch.stack(states).to(device)
    pi = torch.stack(pis).to(device)
    z = torch.tensor(zs, dtype=torch.float32).to(device)

    logits, v = model(x)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(pi * logp).sum(dim=1).mean()
    value_loss = torch.mean((v - z) ** 2)
    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())

def save_ckpt(model: PVNet, opt, step: int, cfg: Config):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"ckpt_step_{step}.pt")
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()}, path)
    print(f"[ckpt] sparad: {path}")

def run_training(cfg: Config):
    model = PVNet().to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    replay = ReplayBuffer(cfg.replay_max_size)

    writer = SummaryWriter(log_dir=cfg.log_dir)
    step = 0
    try:
        while step < cfg.train_steps:
            samples = generate_samples(model, cfg)
            replay.add_many(samples)

            writer.add_scalar("data/samples_added", len(samples), step)
            writer.add_scalar("data/replay_size", len(replay), step)

            if len(replay) < cfg.batch_size:
                continue

            batch = replay.sample(cfg.batch_size)
            loss, pl, vl = train_step(model, opt, batch, cfg.device)
            step += 1

            writer.add_scalar("train/loss_total", loss, step)
            writer.add_scalar("train/loss_policy", pl, step)
            writer.add_scalar("train/loss_value", vl, step)

            if step % cfg.log_every == 0:
                print(f"[train] step={step} loss={loss:.4f} policy={pl:.4f} value={vl:.4f}")

            if step % cfg.save_every == 0:
                save_ckpt(model, opt, step, cfg)

        save_ckpt(model, opt, step, cfg)
        return model
    finally:
        writer.close()
