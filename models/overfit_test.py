"""Step 3 sanity check: overfit 32 fully visible videos to verify learning."""

import argparse

import torch

from common.loss import balanced_mse
from data.generator import make_batch
from models.snn import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    batch = make_batch(32, seed=0, durations=(0,))
    frames = torch.from_numpy(batch["frames"]).to(device)
    xy = torch.from_numpy(batch["xy"]).to(device)
    vis = torch.from_numpy(batch["visible_frac"]).to(device)
    assert torch.all(vis == 1.0), "Expected all 32 videos to be fully visible"

    model = build_model(seed=args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Overfit test: {len(frames)} fully visible videos | device={device} | lr={args.lr}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(frames), device=device)
        total_loss = 0.0
        total_px_err = 0.0
        total_spikes = 0.0

        for i in range(0, len(frames), args.batch_size):
            idx = perm[i:i + args.batch_size]
            f_b, xy_b, vis_b = frames[idx], xy[idx], vis[idx]

            pred, spikes = model(f_b)
            loss = balanced_mse(pred, xy_b, vis_b)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                px_err = ((pred - xy_b) ** 2).sum(dim=-1).sqrt().mean()
                total_loss += loss.item() * len(f_b)
                total_px_err += px_err.item() * 64.0 * len(f_b)
                total_spikes += spikes.mean().item() * len(f_b)

        n = len(frames)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:3d}  loss={total_loss / n:.6f}  "
                f"mean_err={total_px_err / n:5.2f} px  spikes/frame={total_spikes / n:6.1f}"
            )


if __name__ == "__main__":
    main()
