import os
import csv
import argparse
import torch
import wandb

import soundfile as sf
from tqdm import tqdm

from ddsp_piano.data_pipeline import (
    get_training_dataset,
    get_validation_dataset,
    get_test_dataset,
)
from ddsp_piano.default_model import get_model
from ddsp_piano.modules.loss import HybridLoss


def epoch_from_filename(path: str) -> int:
    base = os.path.basename(path)
    marker = 'epoch_'
    if marker not in base:
        return -1
    tail = base.split(marker, 1)[1]
    num = ''
    for ch in tail:
        if ch.isdigit():
            num += ch
        else:
            break
    return int(num) if num else -1


def step_from_filename(path: str) -> int:
    """Extract global step from filenames like 'ddsp-piano_60_params.pt'."""
    name = os.path.basename(path).split('.')[0]
    parts = name.split('_')
    nums = [int(p) for p in parts if p.isdigit()]
    if not nums:
        raise ValueError(f'No step number found in filename: {path}')
    return max(nums)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DDSP-Piano checkpoints on MAESTRO splits.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation (default: %(default)s).')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers (default: %(default)s).')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sampling rate used for synthesis and cache contract.')
    parser.add_argument('--frame_rate', type=int, default=100, help='Conditioning frame rate used for synthesis and cache contract.')
    parser.add_argument('--duration', type=float, default=10.0, help='Segment duration in seconds used by the checkpoint.')
    parser.add_argument('--reverb_duration', type=float, default=1.5, help='Reverb IR duration in seconds.')
    parser.add_argument('--phase', type=int, default=1, help='Training phase the checkpoint belongs to (default: %(default)s).')
    parser.add_argument('--cuda', type=int, default=1, help='Use CUDA if available (default: %(default)s).')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a checkpoint (.pt/_params.pt) or a directory containing checkpoints.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where evaluation metrics and audio (optional) will be stored.')
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test',
                        help='Dataset split to evaluate (default: %(default)s).')
    parser.add_argument('--ckpt_order', choices=['epoch', 'step'], default='epoch',
                        help='When scanning a checkpoint directory, evaluate in epoch order or by global step (default: %(default)s).')
    parser.add_argument('--save_audio', action='store_true', help='Save synthesized audio for each evaluated chunk.')
    parser.add_argument('--debug_mode', action='store_true', help='Debug mode: limit evaluation to 20 batches.')
    # Weights & Biases logging (always on)
    parser.add_argument('--wandb_project', type=str, default='ddsp-piano', help='W&B project for evaluation logs.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Run name for W&B evaluation run.')
    parser.add_argument('maestro_cache_path', type=str, help='Path to the cached numpy segments directory.')
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)
    return model


def discover_state_info(checkpoint_path, ordering):
    """Extract (epoch, global_step) from sibling _state.pt when present.
    If absent, derive only the relevant field per ordering.
    """
    candidates = []
    if checkpoint_path.endswith('_params.pt'):
        candidates.append(checkpoint_path.replace('_params.pt', '_state.pt'))
    if checkpoint_path.endswith('.pt'):
        candidates.append(checkpoint_path.replace('.pt', '_state.pt'))
    for cand in candidates:
        if os.path.isfile(cand):
            state = torch.load(cand, map_location='cpu')
            ep = int(state.get('completed_epochs', state.get('epoch', -1)))
            gs = int(state.get('global_step', -1))
            ep = ep if ep >= 0 else None
            gs = gs if gs >= 0 else None
            if ordering == 'step':
                ep = None
            return ep, gs
    if ordering == 'epoch':
        ep = epoch_from_filename(checkpoint_path)
        ep = ep if ep >= 0 else None
        gs = None
    else:
        ep = None
        gs = step_from_filename(checkpoint_path)
    return ep, gs


def save_wav(signal, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    signal_np = signal.squeeze().detach().cpu().numpy()
    sf.write(path, signal_np, sr)


def discover_checkpoints(path, ordering):
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith('.pt')
        ]
        param_ckpts = [p for p in candidates if p.endswith('_params.pt')]
        checkpoints = param_ckpts or candidates
        # Filter by intent: epoch-only or step-only
        if ordering == 'epoch':
            checkpoints = [p for p in checkpoints if 'epoch_' in os.path.basename(p)]
            checkpoints.sort(key=lambda p: epoch_from_filename(p))
        else:
            checkpoints = [p for p in checkpoints if 'epoch_' not in os.path.basename(p)]
            checkpoints.sort(key=lambda p: step_from_filename(p))
        return checkpoints
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f'Checkpoint path {path} does not exist.')


def get_dataset_for_split(split, metadata_source, cache_path, max_polyphony):
    loaders = {
        'train': get_training_dataset,
        'validation': get_validation_dataset,
        'test': get_test_dataset,
    }
    return loaders[split](metadata_source, cache_path, max_polyphony=max_polyphony)


def evaluate_checkpoint(model, loss_func, dataloader, args, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    wav_dir = os.path.join(output_dir, 'wav') if args.save_audio else None
    debug_max_batches = 20 if getattr(args, 'debug_mode', False) else None

    total_loss = 0.0
    total_mss = 0.0
    total_reverb = 0.0
    total_items = 0
    rows = []

    total_batches = debug_max_batches if debug_max_batches is not None else len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating', total=total_batches)):
            if debug_max_batches is not None and batch_idx >= debug_max_batches:
                break
            audio, conditioning, pedal, piano_model = batch
            audio = audio.to(device)
            conditioning = conditioning.to(device)
            pedal = pedal.to(device)
            piano_model = piano_model.to(device)

            signal, reverb_ir, _ = model(conditioning, pedal, piano_model)
            loss, loss_mss, loss_reverb = loss_func(signal, audio, reverb_ir)

            batch_size = audio.shape[0]
            total_loss += loss.item() * batch_size
            total_mss += loss_mss.item() * batch_size
            total_reverb += loss_reverb.item() * batch_size
            total_items += batch_size

            if args.save_audio:
                rows.append({
                    'batch_index': batch_idx,
                    'batch_size': batch_size,
                    'loss': loss.item(),
                    'loss_mss': loss_mss.item(),
                    'loss_reverb_l1': loss_reverb.item()
                })

            if wav_dir is not None:
                os.makedirs(wav_dir, exist_ok=True)
                for i in range(batch_size):
                    sample_idx = batch_idx * args.batch_size + i
                    wav_path = os.path.join(wav_dir, f'sample_{sample_idx:05d}.wav')
                    save_wav(signal[i], args.sampling_rate, wav_path)

    if total_items == 0:
        raise RuntimeError('No samples were evaluated. Ensure the cache for the requested split exists.')

    # Per-batch CSV is removed by default; keep only averages. If saving audio,
    # optionally store per-batch losses next to audio for inspection.
    if args.save_audio and rows:
        metrics_path = os.path.join(output_dir, 'batches.csv')
        with open(metrics_path, 'w', newline='') as csvfile:
            fieldnames = ['batch_index', 'batch_size', 'loss', 'loss_mss', 'loss_reverb_l1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    avg_loss = total_loss / total_items
    avg_mss = total_mss / total_items
    avg_reverb = total_reverb / total_items

    # Do not print per-ckpt metrics here; caller aggregates to a single CSV.
    return {
        'avg_loss': avg_loss,
        'avg_mss': avg_mss,
        'avg_reverb_l1': avg_reverb,
        'metrics_csv': os.path.join(output_dir, 'batches.csv') if args.save_audio and rows else None,
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    first_phase = (args.phase % 2) == 1

    if args.debug_mode:
        print('======= DEBUG MODE: evaluating only 20 batches =======')

    wandb_run = wandb.init(
        project=args.wandb_project,
        name=(args.wandb_run_name or f"eval_phase{args.phase}_{args.split}"),
        config={
            'phase': args.phase,
            'split': args.split,
            'ckpt_order': args.ckpt_order,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'sampling_rate': args.sampling_rate,
        }
    )
    wandb_run.define_metric('global_step')
    wandb_run.define_metric('epoch')
    wandb_run.define_metric('eval/*', step_metric='global_step')

    checkpoint_paths = discover_checkpoints(args.checkpoint, args.ckpt_order)
    if not checkpoint_paths:
        raise RuntimeError(f'No checkpoints found under {args.checkpoint}')

    # Build dataset/dataloader using the first checkpoint architecture.
    probe_model = get_model(duration=args.duration, frame_rate=args.frame_rate, sample_rate=args.sample_rate, reverb_duration=args.reverb_duration)
    probe_model.alternate_training(first_phase=first_phase)
    probe_model = load_checkpoint(probe_model, checkpoint_paths[0], device)
    max_polyphony = probe_model.n_synths

    metadata_source = args.maestro_cache_path
    dataset = get_dataset_for_split(args.split, metadata_source, args.maestro_cache_path, max_polyphony=max_polyphony)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Prepare consolidated CSV for all checkpoints
    os.makedirs(args.output_dir, exist_ok=True)
    consolidated_csv = os.path.join(args.output_dir, f"metrics_summary_by_{args.ckpt_order}.csv")
    with open(consolidated_csv, 'w', newline='') as csvfile:
        fieldnames = ['ckpt_name', 'epoch', 'global_step', 'split', 'avg_loss', 'avg_mss', 'avg_reverb_l1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ckpt_path in checkpoint_paths:
            ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            ckpt_output = os.path.join(args.output_dir, ckpt_name) if args.save_audio else args.output_dir

            model = get_model(duration=args.duration, frame_rate=args.frame_rate, sample_rate=args.sample_rate, reverb_duration=args.reverb_duration)
            model.alternate_training(first_phase=first_phase)
            model = load_checkpoint(model, ckpt_path, device)
            model.to(device)
            model.eval()

            n_ffts = [2048, 1024, 512, 256, 128, 64]
            loss_func = HybridLoss(n_ffts, model.inharm_model, first_phase)
            loss_func.to(device)
            loss_func.eval()

            # Discover epoch/global_step for logging
            ep, gs = discover_state_info(ckpt_path, args.ckpt_order)
            # If no global step and epoch order, align gs=epoch for plotting
            if gs is None and args.ckpt_order == 'epoch' and ep is not None:
                gs = ep

            print(f'=== Evaluating checkpoint: {ckpt_path} ({args.split} split) ===')
            summary = evaluate_checkpoint(model, loss_func, dataloader, args, ckpt_output, device)

            # Append consolidated row
            writer.writerow({
                'ckpt_name': ckpt_name,
                'epoch': ep if ep is not None else -1,
                'global_step': gs,
                'split': args.split,
                'avg_loss': summary['avg_loss'],
                'avg_mss': summary['avg_mss'],
                'avg_reverb_l1': summary['avg_reverb_l1'],
            })

            wandb_run.log({
                'eval/loss': summary['avg_loss'],
                'eval/loss_mss': summary['avg_mss'],
                'eval/loss_reverb_l1': summary['avg_reverb_l1'],
                'global_step': gs,
                'epoch': ep if ep is not None else -1,
                'ckpt_name': ckpt_name,
                'split': args.split,
            })
            del model, loss_func
            torch.cuda.empty_cache()

    wandb_run.finish()
    print(f'Metrics saved to {consolidated_csv}')


if __name__ == '__main__':
    main()
