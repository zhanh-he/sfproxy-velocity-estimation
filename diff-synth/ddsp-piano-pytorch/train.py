import os, time, datetime, argparse, json, wandb
import numpy as np 
from tqdm import tqdm
import torch
import torch.nn as nn 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from logger.saver import Saver 
from ddsp_piano.data_pipeline import get_training_dataset
from ddsp_piano.default_model import get_model
from ddsp_piano.modules.loss import HybridLoss


def unwrap_optimizer_compile_wrappers():
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


class SaverManager:
    """Saver proxy that becomes a no-op off the main rank."""

    def __init__(self, args, enable: bool):
        self.expdir = args.exp_dir
        self.global_step = -1
        self._saver = Saver(args) if enable else None
        self._init_time = time.time()
        self._last_time = self._init_time

    def _call(self, method, *args, **kwargs):
        if self._saver is not None:
            return getattr(self._saver, method)(*args, **kwargs)

    def log_info(self, *args, **kwargs):
        self._call('log_info', *args, **kwargs)

    def log_value(self, *args, **kwargs):
        self._call('log_value', *args, **kwargs)

    def save_models(self, *args, **kwargs):
        self._call('save_models', *args, **kwargs)

    def make_report(self, *args, **kwargs):
        self._call('make_report', *args, **kwargs)

    def get_interval_time(self, update=True):
        if self._saver is not None:
            return self._saver.get_interval_time(update)
        cur_time = time.time()
        interval = cur_time - self._last_time
        if update:
            self._last_time = cur_time
        return interval

    def get_total_time(self, to_str=True):
        if self._saver is not None:
            return self._saver.get_total_time(to_str)
        total = time.time() - self._init_time
        if to_str:
            total = str(datetime.timedelta(seconds=total))[:-5]
        return total

    def global_step_increment(self):
        self.global_step += 1
        if self._saver is not None:
            self._saver.global_step = self.global_step


class DistEnv:
    """Lightweight wrapper for torch.distributed setup/cleanup."""

    def __init__(self, args):
        requested = bool(getattr(args, 'use_ddp', False))
        env_world = max(1, int(os.environ.get('WORLD_SIZE', '1')))
        self.world_size = env_world
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.enabled = requested or env_world > 1
        if self.enabled and env_world <= 1:
            print('[DDP] --use_ddp specified but WORLD_SIZE=1; running single-process.')
            self.enabled = False
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0

    def setup(self, args):
        want_cuda = bool(args.cuda) and torch.cuda.is_available()
        if self.enabled and not want_cuda:
            raise RuntimeError('Distributed training requires CUDA-enabled GPUs.')
        if self.enabled:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://', world_size=self.world_size, rank=self.rank)
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
        else:
            self.device = torch.device('cuda' if want_cuda else 'cpu')
        self.is_main = (self.rank == 0)

    def wrap_model(self, model):
        model = model.to(self.device)
        if self.enabled:
            return DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
        return model

    def build_sampler(self, dataset):
        if not self.enabled:
            return None
        return DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def cleanup(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def default_wandb_run_name(args) -> str:
    return (
        f"ddsp_piano_phase{int(args.phase)}"
        f"_sr{int(args.sample_rate)}"
        f"_fps{int(args.frame_rate)}"
        f"_seg{float(args.duration):g}s"
    )


def process_args():
    # Get arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', '-b', type=int, default=6,
                        help="Number of elements per batch.\
                        (default: %(default)s)")

    parser.add_argument('--epochs', '-e', type=int, default=7,
                        help="Number of epochs. (default: %(default)s)")

    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate. (default: %(default)s)")

    parser.add_argument('--phase', '-p', type=int, default=1,
                        help="Training phase strategy to apply. \
                        Set to even for fine-tuning only the detuner and \
                        inharmonicity sub-modules.\
                        (default: %(default)s)")

    parser.add_argument('--cuda', type=int, default=1,
                        help="Using Cuda or not")

    parser.add_argument('--logs_interval', type=int, default=20)

    parser.add_argument('--save_interval', type=int, default=2000,
                        help="Global step interval for checkpointing. (default: %(default)s)")

    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Audio sample rate for model and cache contract.')

    parser.add_argument('--frame_rate', type=int, default=100,
                        help='Conditioning frame rate for model and cache contract.')

    parser.add_argument('--duration', type=float, default=10.0,
                        help='Segment duration in seconds for model and cache contract.')

    parser.add_argument('--reverb_duration', type=float, default=1.5,
                        help='Reverb IR duration in seconds.')

    parser.add_argument('--loss_n_ffts', type=str, default='2048,1024,512,256,128,64',
                        help='Comma-separated FFT sizes for HybridLoss.')

    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of worker processes for DataLoader. (default: %(default)s)")

    parser.add_argument('--continue_from', type=str, default=None,
                        help="Resume current phase training from a saved epoch checkpoint (.pt).")
    parser.add_argument('--init_from', type=str, default=None,
                        help="Initialize model weights from a previous phase checkpoint (.pt) without optimizer state.")

    parser.add_argument('--wandb_project', type=str, default='ddsp-piano-unified',
                        help="Weights & Biases project name.")
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help="Optional run name for Weights & Biases.")
    parser.add_argument('--debug_mode', action='store_true',
                        help="Debug mode: run only a handful of steps per epoch and save frequently.")
    parser.add_argument('--use_ddp', action='store_true',
                        help="Enable DistributedDataParallel (multi-GPU) training via torchrun.")

    parser.add_argument('maestro_cache_path', type=str,
                        help="Path to the MAESTRO cache dataset folder.")
    parser.add_argument('exp_dir', type=str,
                        help="Folder to store experiment results and logs.")

    return parser.parse_args()


def epoch_from_filename(path: str) -> int:
    for chunk in os.path.basename(path).split('_'):
        if chunk.isdigit():
            return int(chunk)
    return -1


def load_model_weights(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path)
    state = ckpt.state_dict() if isinstance(ckpt, nn.Module) else ckpt
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state)


def parse_loss_n_ffts(spec: str):
    return [int(tok.strip()) for tok in str(spec).split(',') if tok.strip()]


def load_cache_contract(cache_root: str):
    path = os.path.join(cache_root, 'cache_config.json')
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_cache_contract(cache_root: str, *, sample_rate: int, frame_rate: int, duration: float):
    contract = load_cache_contract(cache_root)
    if contract is None:
        print('[CacheContract] cache_config.json not found. Skipping cache contract validation.')
        return
    expected = {
        'sample_rate': int(sample_rate),
        'frame_rate': int(frame_rate),
        'segment_duration': float(duration),
    }
    mismatches = []
    for key, value in expected.items():
        cached = contract.get(key)
        if key == 'segment_duration':
            if cached is None or abs(float(cached) - value) > 1e-6:
                mismatches.append((key, cached, value))
        else:
            if cached is None or int(cached) != int(value):
                mismatches.append((key, cached, value))
    if mismatches:
        lines = [
            'Cache contract mismatch between training args and preprocessed cache.',
            f'cache_root={cache_root}',
        ]
        for key, cached, value in mismatches:
            lines.append(f'  {key}: cache={cached}, train={value}')
        raise ValueError('\n'.join(lines))


def main(args):
    """Training loop script.
    Args:
        - batch_size (int): nb of elements per batch.
        - epochs (int): nb of epochs.
        - restore (path): load model and optimizer states from this folder.
        - phase (int): current training phase.
        - exp_dir (path): folder to store experiment results and logs.
    """
    #print('args : ', args)
    # Format training phase strategy
    first_phase_strat = ((args.phase % 2) == 1)
    dist_env = DistEnv(args)
    dist_env.setup(args)
    is_main_process = dist_env.is_main
    use_ddp = dist_env.enabled
    use_cuda = (dist_env.device.type == 'cuda')

    # Prepare model
    validate_cache_contract(
        args.maestro_cache_path,
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        duration=args.duration,
    )

    raw_model = get_model(
        duration=args.duration,
        frame_rate=args.frame_rate,
        sample_rate=args.sample_rate,
        reverb_duration=args.reverb_duration,
    )
    raw_model.alternate_training(first_phase=first_phase_strat)
    continue_ckpt = getattr(args, 'continue_from', None)
    init_ckpt = getattr(args, 'init_from', None)
    continue_state = None
    if continue_ckpt and init_ckpt:
        raise ValueError('--continue_from and --init_from cannot be used together.')

    completed_epochs = 0
    resume_global_step = 0

    if continue_ckpt:
        base = continue_ckpt.replace('_params', '')
        state_path = base.replace('.pt', '_state.pt')
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"Expected state file not found: {state_path}")
        continue_state = torch.load(state_path)
        raw_model.load_state_dict(continue_state['model'])
        completed_epochs = int(continue_state.get('completed_epochs', continue_state.get('epoch', epoch_from_filename(continue_ckpt))))
        if completed_epochs < 0:
            raise ValueError(f'Unable to determine completed epochs from checkpoint: {continue_ckpt}')
        resume_global_step = int(continue_state.get('global_step', 0))
        print(f'[Resume] continue_from {continue_ckpt} (completed epoch {completed_epochs})')

    elif init_ckpt:
        load_model_weights(raw_model, init_ckpt)
        print(f'[Resume] init_from {init_ckpt}')

    model = dist_env.wrap_model(raw_model)
    # Prepare dataset 
    metadata_source = args.maestro_cache_path
    training_dataset = get_training_dataset(metadata_source, args.maestro_cache_path, max_polyphony=raw_model.n_synths)
    train_sampler = dist_env.build_sampler(training_dataset)

    training_dataset_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        num_workers=args.num_workers
    )
    # Prepare Loss function 
    n_ffts = parse_loss_n_ffts(args.loss_n_ffts)
    loss_func = HybridLoss(n_ffts, raw_model.inharm_model, first_phase_strat)

    # Optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # GPU or CPU (default: GPU)
    if use_cuda:
        loss_func = loss_func.cuda()
    
    # Inits before the training loop
    args.exp_dir = os.path.join(args.exp_dir, f'phase_{args.phase}')
    saver = SaverManager(args, enable=is_main_process)
    if continue_state:
        optimizer.load_state_dict(continue_state['optimizer'])
        if train_sampler is not None and hasattr(train_sampler, 'load_state_dict'):
            sampler_state = continue_state.get('sampler')
            if sampler_state is not None:
                train_sampler.load_state_dict(sampler_state)
        saver.global_step = resume_global_step
    step_save_interval = 20 if args.debug_mode else args.save_interval

    wandb_run_id_path = os.path.join(args.exp_dir, 'wandb_run_id.txt')
    existing_wandb_id = None
    if os.path.isfile(wandb_run_id_path):
        with open(wandb_run_id_path, 'r') as f:
            existing_wandb_id = f.read().strip() or None

    if is_main_process:
        resume_strategy = 'must' if existing_wandb_id else 'allow'
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=(args.wandb_run_name or default_wandb_run_name(args)),
            config=vars(args),
            id=existing_wandb_id,
            resume=resume_strategy,
        )
        # Define metric relationships for better charts
        wandb_run.define_metric('global_step')
        wandb_run.define_metric('train/*', step_metric='global_step')
        wandb_run.define_metric('epoch')
        if existing_wandb_id is None:
            with open(wandb_run_id_path, 'w') as f:
                f.write(wandb_run.id)
    else:
        wandb_run = None

    # Training loop
    base_steps = len(training_dataset_loader)
    if completed_epochs > 0 and saver.global_step <= 0:
        saver.global_step = completed_epochs * base_steps

    if completed_epochs >= args.epochs:
        if is_main_process:
            saver.log_info('[Resume] Checkpoint already finished requested epochs. Nothing to train.')
        if is_main_process and wandb_run is not None:
            wandb_run.finish()
        dist_env.cleanup()
        return

    if is_main_process:
        if args.debug_mode:
            saver.log_info('======= DEBUG MODE: running limited 100 steps per epoch =======')
        else:
            saver.log_info('======= start training with full data =======')
            
    for epoch_idx in range(completed_epochs, args.epochs):
        epoch_number = epoch_idx + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)
        epoch_total_steps = base_steps
        debug_cap = 100
        if args.debug_mode:
            epoch_total_steps = min(epoch_total_steps, debug_cap)
        progress = tqdm(total=epoch_total_steps,
                        desc=f'Epoch {epoch_number}/{args.epochs}',
                        ncols=100) if is_main_process else None
        steps_in_epoch = 0
        for idx, batch in enumerate(training_dataset_loader):
            saver.global_step_increment()
            audio, conditioning, pedal, piano_model = batch
            '''
                - audio: (b, sample_rate*duration)
                - conditioning: (b, n_frames, max_polyphony, 2)
                - pedal: (b, n_frames, 4)
                - piano_model: (b)
            '''
            if use_cuda:
                audio = audio.cuda()
                conditioning = conditioning.cuda()
                pedal = pedal.cuda()
                piano_model = piano_model.cuda()

            # forward 
            signal, reverb_ir, non_ir_signal = model(conditioning, pedal, piano_model)

            # loss 
            loss, loss_mss, loss_reverb_l1 = loss_func(signal, audio, reverb_ir)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_main_process and saver.global_step % args.logs_interval == 0:
                saver.log_info(
                    'epoch: {}/{} {:3d}/{:3d} | {} | t: {:.2f} | loss: {:.6f} | time: {} | counter: {}'.format(
                        epoch_number,
                        args.epochs,
                        idx,
                        len(training_dataset_loader),
                        saver.expdir,
                        saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    ),
                    console=False
                )
                saver.log_info(
                    ' > loss: {:.6f}, mss loss: {:.6f}, reverb_loss: {:.6f}'.format(
                       loss.item(),
                       loss_mss.item(),
                       loss_reverb_l1.item()
                    ),
                    console=False
                )
                saver.log_value({
                    'train loss': loss.item(),
                    'train loss mss': loss_mss.item(),
                    'train loss reverb_l1': loss_reverb_l1.item()
                })
                wandb_run.log({
                    'train/loss': loss.item(),
                    'train/loss_mss': loss_mss.item(),
                    'train/loss_reverb_l1': loss_reverb_l1.item(),
                    'global_step': saver.global_step,
                    'epoch': epoch_number
                })
            if is_main_process and step_save_interval > 0 and saver.global_step % step_save_interval == 0:
                model_to_save = unwrap_model(model)
                saver.save_models({'ddsp-piano': model_to_save}, postfix=f'{saver.global_step}')
                saver.make_report()

            if progress is not None:
                progress.set_postfix({'loss': f'{loss.item():.3f}', 'mss': f'{loss_mss.item():.3f}'})
                progress.update(1)
            steps_in_epoch += 1
            if args.debug_mode and steps_in_epoch >= debug_cap:
                break
        
        if progress is not None:
            progress.close()

        if is_main_process:
            model_to_save = unwrap_model(model)
            saver.save_models({'ddsp-piano': model_to_save}, postfix=f'epoch_{epoch_number}')
            saver.make_report()
            ckpt_dir = os.path.join(args.exp_dir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            state_path = os.path.join(ckpt_dir, f'ddsp-piano_epoch_{epoch_number}_state.pt')
            sampler_snapshot = train_sampler.state_dict() if train_sampler and hasattr(train_sampler, 'state_dict') else None
            torch.save({
                'epoch': epoch_number,
                'completed_epochs': epoch_number,
                'global_step': saver.global_step,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'sampler': sampler_snapshot,
            }, state_path)
            # Mark epoch boundary in W&B
            wandb_run.log({'epoch': epoch_number, 'epoch/end_global_step': saver.global_step})

    if is_main_process:
        wandb_run.finish()

    dist_env.cleanup()

if __name__ == '__main__':
    main(process_args())
