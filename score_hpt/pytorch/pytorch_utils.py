import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb

def move_data_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if np.issubdtype(x.dtype, np.floating):
        return torch.tensor(x, dtype=torch.float32, device=device)
    if np.issubdtype(x.dtype, np.integer):
        return torch.tensor(x, dtype=torch.long, device=device)
    return x


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

def forward(model, x, batch_size):
    """Forward data to model in mini-batch. 
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int
    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
    """
    
    output_dict = {}
    device = next(model.parameters()).device
    
    pointer = 0
    while True:
        if pointer >= len(x):
            break
        batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)
            
        for key in batch_output_dict.keys():
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict


def forward_velo(model, x1, x2, x3, batch_size):
    """Forward data to model in mini-batch.
    Args:
      model: object
      x1: audio_segments (N, segment_samples)
      x2: onset_roll(N, frames_num, classes_num)
      x3: frame_roll or exframe_roll (onset-excluded frame) (N, frames_num, classes_num)
      batch_size: int
    Returns:
      output_dict: dict, {'velocity_output': (segments_num, frames_num, classes_num)}
    """
    output_dict = {}
    device = next(model.parameters()).device
    pointer = 0

    while pointer < len(x1):
        # Process batches
        batch_audio = move_data_to_device(x1[pointer: pointer + batch_size], device)
        batch_input2 = move_data_to_device(x2[pointer: pointer + batch_size], device) if x2 is not None else None
        batch_input3 = move_data_to_device(x3[pointer: pointer + batch_size], device) if x3 is not None else None
        pointer += batch_size

        with torch.no_grad():
            model.eval()
            inputs = [batch_audio]
            if x2 is not None:
                inputs.append(batch_input2)
            if x3 is not None:
                inputs.append(batch_input3)
            batch_output_dict = model(*inputs)

        append_to_dict(output_dict, 'velocity_output', batch_output_dict['velocity_output'].data.cpu().numpy())

    output_dict['velocity_output'] = np.concatenate(output_dict['velocity_output'], axis=0)
    return output_dict


def _note_segments(active_mask):
    padded = np.pad(active_mask.astype(np.int8), (1, 1), mode="constant")
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _post_process_rolls(pred_vis, target_raw, onset_roll):
    _, keys = target_raw.shape
    frame_pick_roll = np.zeros_like(pred_vis, dtype=np.float32)
    onset_pick_roll = np.zeros_like(pred_vis, dtype=np.float32)

    for key in range(keys):
        note_active = target_raw[:, key] > 0
        for start, end in _note_segments(note_active):
            if end <= start:
                continue
            pred_note = pred_vis[start:end, key].copy()
            pred_note[pred_note <= 1e-4] = 0.0

            frame_val = float(np.max(pred_note)) if pred_note.size else 0.0
            frame_pick_roll[start:end, key] = frame_val

            onset_mask = onset_roll[start:end, key] > 0
            if np.any(onset_mask):
                onset_val = float(np.max(pred_note[onset_mask]))
            else:
                onset_val = 0.0
            onset_pick_roll[start:end, key] = onset_val

    return frame_pick_roll, onset_pick_roll


def log_velocity_rolls(cfg, iteration, batch_output_dict, batch_data_dict):
    """Log prediction vs target velocity rolls to WandB at configured intervals."""
    interval = cfg.exp.eval_iteration
    if not interval or interval <= 0:
        return
    if iteration % interval != 0:
        return

    pred = batch_output_dict.get("velocity_output")
    if pred is None:
        pred = batch_output_dict.get("vel_corr")
    target = batch_data_dict.get("velocity_roll")
    onset = batch_data_dict.get("onset_roll")
    if pred is None or target is None:
        return

    velocity_scale = getattr(cfg.feature, "velocity_scale", 128)
    pred_img = pred[0].detach().cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred[0])
    pred_max = float(np.max(pred_img))
    pred_min = float(np.min(pred_img))
    if pred_max > 1.0 + 1e-3 or pred_min < -1e-3:
        pred_vis = np.clip(pred_img / velocity_scale, 0.0, 1.0)
    else:
        pred_vis = np.clip(pred_img, 0.0, 1.0)
    target_raw = target[0].detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target[0])
    if float(np.max(target_raw)) <= 1.0 + 1e-3:
        target_raw = target_raw * velocity_scale
    target_img = np.clip(target_raw / velocity_scale, 0.0, 1.0)
    if onset is None:
        onset_roll = np.zeros_like(target_raw)
    else:
        onset_roll = onset[0].detach().cpu().numpy() if torch.is_tensor(onset) else np.asarray(onset[0])

    frame_pick_img, onset_pick_img = _post_process_rolls(
        pred_vis=pred_vis,
        target_raw=target_raw,
        onset_roll=onset_roll,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    specs = [
        ("Ground Truth", target_img),
        ("Prediction", pred_vis),
        ("Post-Proc Frame-Pick", frame_pick_img),
        ("Post-Proc Onset-Pick", onset_pick_img),
    ]
    for ax, (title, data) in zip(np.ravel(axes), specs):
        im = ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Key")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Velocity roll @ iter {iteration}")
    fig.tight_layout()

    wandb.log(
        {
            "velocity_roll_comparison": wandb.Image(fig),
        },
        step=iteration,
    )
    plt.close(fig)
