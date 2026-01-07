# run.py
import os
import argparse
import torch
from train_rl import Config, run_training

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--games-per-iter", type=int, default=10)
    p.add_argument("--mcts-sims", type=int, default=200)

    p.add_argument("--sparring-ratio", type=float, default=0.2)
    p.add_argument("--engine-path", default="")
    p.add_argument("--engine-depth", type=int, default=6)

    # visual/logging
    p.add_argument("--logdir", default="runs")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-last-pgn", action="store_true")
    p.add_argument("--pgn-path", default="last_game.pgn")

    # multiprocessing
    p.add_argument("--workers", type=int, default=0)  # 0 = av
    p.add_argument("--mp-start", default="spawn", choices=["spawn", "fork", "forkserver"])
    p.add_argument("--mp-chunk-games", type=int, default=2)

    args = p.parse_args()

    cfg = Config(
        device=args.device,
        train_steps=args.steps,
        selfplay_games_per_iter=args.games_per_iter,
        mcts_sims=args.mcts_sims,
        sparring_ratio=args.sparring_ratio,
        uci_engine_path=args.engine_path or os.getenv("UCI_ENGINE_PATH", ""),
        uci_engine_depth=args.engine_depth,
        log_dir=args.logdir,
        log_every=args.log_every,
        save_last_pgn=args.save_last_pgn,
        pgn_path=args.pgn_path,
        workers=args.workers,
        mp_start_method=args.mp_start,
        mp_chunk_games=args.mp_chunk_games,
    )

    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA valdes men finns inte tillgängligt. Byter till CPU.")
            cfg.device = "cpu"
        # rekommenderat
        if cfg.workers > 0:
            print("Tips: GPU + multiprocessing kan strula. Sätter workers=0.")
            cfg.workers = 0

    run_training(cfg)

if __name__ == "__main__":
    main()
