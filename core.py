# core.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Encoding (board + moves)
# =========================

PIECE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}
PROMO_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
BASE = 64 * 64
MOVE_SPACE_SIZE = BASE + BASE * 4  # 20480

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    x = torch.zeros(13, 8, 8, dtype=torch.float32)
    for square, piece in board.piece_map().items():
        r = 7 - chess.square_rank(square)
        c = chess.square_file(square)
        offset = 0 if piece.color == chess.WHITE else 6
        plane = offset + PIECE_TO_PLANE[piece.piece_type]
        x[plane, r, c] = 1.0
    x[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return x

def move_to_index(move: chess.Move) -> int:
    frm, to = move.from_square, move.to_square
    idx = frm * 64 + to
    if move.promotion:
        p = PROMO_PIECES.index(move.promotion)
        return BASE + idx * 4 + p
    return idx

def index_to_move(idx: int) -> chess.Move:
    if idx < BASE:
        frm, to = idx // 64, idx % 64
        return chess.Move(frm, to)
    idx2 = idx - BASE
    base_idx, p = idx2 // 4, idx2 % 4
    frm, to = base_idx // 64, base_idx % 64
    return chess.Move(frm, to, promotion=PROMO_PIECES[p])

def legal_moves_mask(board: chess.Board, device: str) -> torch.Tensor:
    mask = torch.zeros(MOVE_SPACE_SIZE, dtype=torch.bool, device=device)
    for m in board.legal_moves:
        mask[move_to_index(m)] = True
    return mask

def result_white(board: chess.Board) -> float:
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0
    if res == "0-1":
        return -1.0
    return 0.0

# =========================
# Model (Policy + Value)
# =========================

class PVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, MOVE_SPACE_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

@torch.no_grad()
def net_eval(model: PVNet, board: chess.Board, device: str) -> Tuple[np.ndarray, float]:
    """
    Returnerar (policy_probs över MOVE_SPACE_SIZE, value ur side-to-move-perspektiv).
    """
    x = board_to_tensor(board).unsqueeze(0).to(device)
    logits, value = model(x)
    logits = logits.squeeze(0)

    mask = legal_moves_mask(board, device=device)
    masked = torch.full((MOVE_SPACE_SIZE,), -1e9, device=device)
    masked[mask] = logits[mask]

    probs = torch.softmax(masked, dim=0).detach().cpu().numpy()
    v = float(value.item())
    return probs, v

# =========================
# MCTS (PUCT)
# =========================

@dataclass
class Node:
    fen: str
    prior: float = 0.0
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    children: Dict[int, "Node"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}

def expand(node: Node, board: chess.Board, priors: np.ndarray):
    for m in board.legal_moves:
        a = move_to_index(m)
        if a in node.children:
            continue
        b2 = board.copy()
        b2.push(m)
        node.children[a] = Node(fen=b2.fen(), prior=float(priors[a]))

def add_dirichlet_noise(node: Node, alpha: float, epsilon: float):
    if not node.children:
        return
    actions = list(node.children.keys())
    noise = np.random.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
        ch = node.children[a]
        ch.prior = (1 - epsilon) * ch.prior + epsilon * float(n)

def select_child(node: Node, c_puct: float) -> Tuple[int, Node]:
    best_a, best_score = None, -1e9
    sqrtN = math.sqrt(node.N + 1e-8)
    for a, ch in node.children.items():
        U = c_puct * ch.prior * (sqrtN / (1 + ch.N))
        score = ch.Q + U
        if score > best_score:
            best_score, best_a = score, a
    return best_a, node.children[best_a]

def backup(path: List[Node], value: float):
    v = value
    for n in reversed(path):
        n.N += 1
        n.W += v
        n.Q = n.W / n.N
        v = -v

def mcts_search(
    model: PVNet,
    root_board: chess.Board,
    device: str,
    sims: int,
    c_puct: float,
    dir_alpha: float,
    dir_eps: float,
    add_noise: bool
) -> Node:
    root = Node(fen=root_board.fen(), prior=1.0)
    priors, _v = net_eval(model, root_board, device)
    expand(root, root_board, priors)
    root.N = 1
    if add_noise:
        add_dirichlet_noise(root, dir_alpha, dir_eps)

    for _ in range(sims):
        board = root_board.copy()
        node = root
        path = [root]

        while node.children and not board.is_game_over(claim_draw=True):
            a, node = select_child(path[-1], c_puct)
            move = index_to_move(a)
            if move not in board.legal_moves:
                break
            board.push(move)
            path.append(node)

        if board.is_game_over(claim_draw=True):
            zw = result_white(board)
            v_side = zw if board.turn == chess.WHITE else -zw
            backup(path, v_side)
            continue

        priors, v = net_eval(model, board, device)
        expand(node, board, priors)
        backup(path, v)

    return root

def visits_to_policy(root: Node, temperature: float) -> np.ndarray:
    counts = np.zeros((MOVE_SPACE_SIZE,), dtype=np.float32)
    for a, ch in root.children.items():
        counts[a] = ch.N

    s = float(np.sum(counts))
    if s <= 0:
        return counts

    if temperature <= 1e-6:
        out = np.zeros_like(counts)
        out[int(np.argmax(counts))] = 1.0
        return out

    counts_t = np.where(counts > 0, np.power(counts, 1.0 / temperature), 0.0)
    counts_t = counts_t / (np.sum(counts_t) + 1e-12)
    return counts_t

# =========================
# Replay buffer
# =========================

Sample = Tuple[torch.Tensor, torch.Tensor, float]  # (state, pi, z)

class ReplayBuffer:
    def __init__(self, max_size: int):
        from collections import deque
        self.buf = deque(maxlen=max_size)

    def add_many(self, samples: List[Sample]):
        self.buf.extend(samples)

    def __len__(self):
        return len(self.buf)

    def sample(self, n: int) -> List[Sample]:
        return random.sample(self.buf, n)
