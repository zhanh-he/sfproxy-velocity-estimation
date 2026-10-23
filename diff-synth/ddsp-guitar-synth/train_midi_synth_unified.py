#!/usr/bin/env python3
"""CLI trainer for DDSP-Guitar-Synth with the repo's unified contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List

import torch
import wandb
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def unwrap_optimizer_compile_wrappers() -> None:
    for name in ('add_param_group', 'zero_grad', 'state_dict', 'load_state_dict'):
        method = getattr(torch.optim.Optimizer, name)
        wrapped = getattr(method, '__wrapped__', None)
        if wrapped is not None:
            setattr(torch.optim.Optimizer, name, wrapped)
    step = getattr(torch.optim.Adam.step, '__wrapped__', None)
    if step is not None:
        def eager_adam_step(self, closure=None):
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(self.defaults.get('differentiable', False))
                return step(self, closure)
            finally:
                torch.set_grad_enabled(prev_grad)
        torch.optim.Adam.step = eager_adam_step


unwrap_optimizer_compile_wrappers()



def round_half_up(x: float) -> int:
    return int(math.floor(float(x) + 0.5))


def derive_hop_length(sample_rate: int, target_frame_rate: float) -> int:
    return max(1, round_half_up(float(sample_rate) / float(target_frame_rate)))


def parse_fft_sizes(text: str) -> List[int]:
    values = [int(part.strip()) for part in str(text).split(',') if part.strip()]
    if not values:
        raise ValueError('loss_fft_sizes must contain at least one FFT size')
    return values


def safe_log(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return torch.log(x + eps)


def multi_scale_spectral_loss_custom(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: Iterable[int],
    alpha: float = 1.0,
) -> torch.Tensor:
    loss = pred.new_tensor(0.0)
    for fft_size in fft_sizes:
        hop = max(1, int(fft_size // 4))
        window = torch.hann_window(fft_size, device=pred.device)
        pred_stft = torch.stft(
            pred,
            n_fft=fft_size,
            hop_length=hop,
            win_length=fft_size,
            window=window,
            normalized=True,
            return_complex=True,
        ).abs()
        target_stft = torch.stft(
            target,
            n_fft=fft_size,
            hop_length=hop,
            win_length=fft_size,
            window=window,
            normalized=True,
            return_complex=True,
        ).abs()
        linear = (pred_stft - target_stft).abs().mean()
        logmag = (safe_log(pred_stft) - safe_log(target_stft)).abs().mean()
        loss = loss + linear + alpha * logmag
    return loss


def resolve_target_audio(batch: dict, target_audio_key: str) -> torch.Tensor:
    if target_audio_key in batch:
        return batch[target_audio_key]
    if target_audio_key == 'mic_audio' and 'mix_audio' in batch:
        return batch['mix_audio']
    if target_audio_key == 'mix_audio' and 'mic_audio' in batch:
        return batch['mic_audio']
    raise KeyError(f"Target audio key '{target_audio_key}' not found in batch. Available: {list(batch.keys())}")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    train_loss_history: list[float],
    val_loss_history: list[float],
    renderer_config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': int(epoch),
            'epochs_trained': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss_history': list(train_loss_history),
            'val_loss_history': list(val_loss_history),
            'renderer_config': renderer_config,
        },
        path,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fft_sizes: list[int],
    target_audio_key: str,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch in loader:
        conditioning = batch['conditioning'].to(device)
        target_audio = resolve_target_audio(batch, target_audio_key).to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(conditioning)
        pred_audio = outputs['audio']
        min_len = min(pred_audio.size(-1), target_audio.size(-1))
        pred_audio = pred_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]
        loss = multi_scale_spectral_loss_custom(pred_audio, target_audio, fft_sizes=fft_sizes)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        batches += 1
    return total / max(1, batches)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fft_sizes: list[int],
    target_audio_key: str,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for batch in loader:
        conditioning = batch['conditioning'].to(device)
        target_audio = resolve_target_audio(batch, target_audio_key).to(device)
        outputs = model(conditioning)
        pred_audio = outputs['audio']
        min_len = min(pred_audio.size(-1), target_audio.size(-1))
        pred_audio = pred_audio[..., :min_len]
        target_audio = target_audio[..., :min_len]
        loss = multi_scale_spectral_loss_custom(pred_audio, target_audio, fft_sizes=fft_sizes)
        total += float(loss.item())
        batches += 1
    return total / max(1, batches)


def load_dataset(path: Path):
    from data.gset_midi_dataset import GsetMidiDataset
    return GsetMidiDataset(name=path.name, datasets_path=str(path.parent))


def sanitize_wandb_config_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize_wandb_config_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_wandb_config_value(v) for v in value]
    return value


def build_wandb_config(*, args: argparse.Namespace, run_dir: Path, effective_frame_rate: float, hop_length: int, fft_sizes: list[int]):
    config = {key: sanitize_wandb_config_value(value) for key, value in vars(args).items()}
    config.update({
        'run_dir': str(run_dir),
        'effective_frame_rate': float(effective_frame_rate),
        'hop_length': int(hop_length),
        'loss_fft_sizes': [int(v) for v in fft_sizes],
    })
    return config


def init_wandb_run(*, run_dir: Path, run_name: str, args: argparse.Namespace, config: dict):
    wandb_run_id_path = run_dir / 'wandb_run_id.txt'
    should_resume = args.resume_checkpoint is not None
    existing_wandb_id = None
    if should_resume and wandb_run_id_path.exists():
        existing_wandb_id = wandb_run_id_path.read_text().strip() or None
    resume_strategy = 'must' if existing_wandb_id else None
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=(args.wandb_run_name.strip() or run_name),
        config=config,
        id=existing_wandb_id,
        resume=resume_strategy,
    )
    wandb_run.define_metric('epoch')
    wandb_run.define_metric('train/*', step_metric='epoch')
    wandb_run.define_metric('val/*', step_metric='epoch')
    wandb_run_id_path.write_text(wandb_run.id)
    return wandb_run


def main() -> None:
    parser = argparse.ArgumentParser(description='Train DDSP-Guitar-Synth from prepared conditioning/audio .npz files.')
    parser.add_argument('--train_dataset_path', type=Path, required=True)
    parser.add_argument('--val_dataset_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--wandb_project', type=str, default='ddsp-guitar-synth-unified')
    parser.add_argument('--wandb_run_name', type=str, default='')
    parser.add_argument('--sample_rate', type=int, default=22050)
    parser.add_argument('--frame_rate', type=float, default=100.0, help='Target frame rate before integer hop rounding.')
    parser.add_argument('--hop_length', type=int, default=0, help='0 means derive from sample_rate/frame_rate using round-half-up.')
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--segment_seconds', type=float, default=10.0)
    parser.add_argument('--target_audio_key', type=str, default='mic_audio', choices=['mic_audio', 'mix_audio'])
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler_step_size', type=int, default=300)
    parser.add_argument('--scheduler_gamma', type=float, default=1.0)
    parser.add_argument('--loss_fft_sizes', type=str, default='128,256,512,1024,2048')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume_checkpoint', type=Path, default=None)
    args = parser.parse_args()

    fft_sizes = parse_fft_sizes(args.loss_fft_sizes)
    hop_length = int(args.hop_length) if int(args.hop_length) > 0 else derive_hop_length(args.sample_rate, args.frame_rate)
    effective_frame_rate = float(args.sample_rate) / float(hop_length)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    train_dataset = load_dataset(args.train_dataset_path)
    val_dataset = load_dataset(args.val_dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    from midi_synth.midi_synth import MidiSynth

    model = MidiSynth(
        sr=args.sample_rate,
        hop_length=hop_length,
        reverb_length=args.sample_rate,
        target_audio=args.target_audio_key,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    start_epoch = 0
    train_history: list[float] = []
    val_history: list[float] = []
    if args.resume_checkpoint is not None:
        state = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(state['model_state_dict'], strict=True)
        if 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = int(state.get('epoch', state.get('epochs_trained', 0)))
        train_history = list(state.get('train_loss_history', []))
        val_history = list(state.get('val_loss_history', []))

    run_name = args.run_name.strip() or f"ddsp_guitar_synth_sr{args.sample_rate}_fps{int(args.frame_rate)}_seg{args.segment_seconds:g}s"
    run_dir = args.output_dir / run_name
    checkpoint_root = run_dir / 'checkpoints'
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    renderer_config = {
        'checkpoint_family': 'ddsp_guitar_synth',
        'sample_rate': int(args.sample_rate),
        'target_frame_rate': float(args.frame_rate),
        'effective_frame_rate': float(effective_frame_rate),
        'hop_length': int(hop_length),
        'n_fft': int(args.n_fft),
        'segment_seconds': float(args.segment_seconds),
        'target_audio_key': str(args.target_audio_key),
        'loss_fft_sizes': list(fft_sizes),
    }
    (run_dir / 'renderer_config.json').write_text(json.dumps(renderer_config, indent=2))
    wandb_config = build_wandb_config(
        args=args,
        run_dir=run_dir,
        effective_frame_rate=effective_frame_rate,
        hop_length=hop_length,
        fft_sizes=fft_sizes,
    )
    wandb_run = init_wandb_run(run_dir=run_dir, run_name=run_name, args=args, config=wandb_config)

    print(json.dumps({
        'train_dataset_path': str(args.train_dataset_path),
        'val_dataset_path': str(args.val_dataset_path),
        'run_dir': str(run_dir),
        'sample_rate': args.sample_rate,
        'target_frame_rate': args.frame_rate,
        'effective_frame_rate': effective_frame_rate,
        'hop_length': hop_length,
        'n_fft': args.n_fft,
        'segment_seconds': args.segment_seconds,
        'loss_fft_sizes': fft_sizes,
        'device': str(device),
    }, indent=2))

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device, fft_sizes, args.target_audio_key)
            val_loss = validate_epoch(model, val_loader, device, fft_sizes, args.target_audio_key)
            scheduler.step()

            train_history.append(float(train_loss))
            val_history.append(float(val_loss))
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            wandb_run.log({
                'epoch': int(epoch),
                'train/loss': float(train_loss),
                'val/loss': float(val_loss),
                'train/lr': float(scheduler.get_last_lr()[0]),
            })

            if epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs:
                epoch_dir = checkpoint_root / (f"{epoch} epoch" if epoch == 1 else f"{epoch} epochs")
                ckpt_path = epoch_dir / 'model_checkpoint.pt'
                save_checkpoint(
                    ckpt_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss_history=train_history,
                    val_loss_history=val_history,
                    renderer_config=renderer_config,
                )
                latest_path = run_dir / 'latest_model_checkpoint.pt'
                save_checkpoint(
                    latest_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss_history=train_history,
                    val_loss_history=val_history,
                    renderer_config=renderer_config,
                )
    finally:
        wandb_run.finish()


if __name__ == '__main__':
    main()
