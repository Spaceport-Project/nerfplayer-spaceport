
# Nerfplayer Troubleshouting

This guide is created to step by step troubleshouting while training Nerfplayer model.

Let the conda environment name be 'nerfspaceport':

```bash
  conda activate nerfspaceport
  pip uninstall nerfplayer
  pip install git+https://github.com/lsongx/nerfplayer-nerfstudio
```
After successfully installing the nerfplayer python package into the conda environment, we need to make 2 small modifications in nerfplayer_config.py file of nerfplayer package.

This file can be found in following path:
```console
(nerfspaceport) ubuntu@ip-172-31-12-218:/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer$ ls
__init__.py  cuda                  nerfplayer_nerfacto.py        nerfplayer_ngp.py        temporal_grid.py
__pycache__  nerfplayer_config.py  nerfplayer_nerfacto_field.py  nerfplayer_ngp_field.py
```

```python
"""
NeRFPlayer config.
"""


from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfplayer.nerfplayer_nerfacto import NerfplayerNerfactoModelConfig
from nerfplayer.nerfplayer_ngp import NerfplayerNGPModelConfig


nerfplayer_nerfacto = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfplayer-nerfacto",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=DycheckDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NerfplayerNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="NeRFPlayer with nerfacto backbone.",
)

nerfplayer_ngp = MethodSpecification(
    config=TrainerConfig(
        method_name="nerfplayer-ngp",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=DynamicBatchPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=DycheckDataParserConfig(),
                train_num_rays_per_batch=8192,
            ),
            model=NerfplayerNGPModelConfig(
                eval_num_rays_per_chunk=8192,
                grid_levels=1,
                alpha_thre=0.0,
                render_step_size=0.001,
                disable_scene_contraction=True,
                near_plane=0.01,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=64000),
        vis="viewer",
    ),
    description="NeRFPlayer with InstantNGP backbone.",
)
```

In this file add following import line somewhere in the begining of file:

```python
from nerfstudio.data.dataparsers.nerfplayer_multicam_dataparser import NerfplayerMulticamDataParserConfig
```

Also change the following line for both nerfplayer_nerfacto and nerfplayer_ngp:

```python
dataparser=DycheckDataParserConfig(),
```
to
```python
dataparser=NerfplayerMulticamDataParserConfig(),
```

After making this changes when we try to start training as follows:

```bash
  ns-train nerfplayer-nerfacto --data /home/ubuntu/Softwares/data/wrapped_data_new_600
```

We get and link error for jit compilation:
```console
...
FAILED: nerfacc_cuda.so 
c++ scan.cuda.o grid.cuda.o pdf.cuda.o camera.cuda.o nerfacc.o -shared -L/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/opt/conda/envs/nerfspaceport/lib64 -lcudart -o nerfacc_cuda.so
/usr/bin/ld: cannot find -lcudart
collect2: error: ld returned 1 exit status
ninja: build stopped: subcommand failed.
```

To resolve this error open '.bashrc' file add following lines, 

```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda"
```

then source '.bashrc' file and activate conda environment again, navigate to the nerfspaceport and start training again:

```console
(nerfspaceport) ubuntu@ip-172-31-12-218:~$ source .bashrc 
(base) ubuntu@ip-172-31-12-218:~$ cd Softwares/nerf-spaceport/
(base) ubuntu@ip-172-31-12-218:~/Softwares/nerf-spaceport$ conda activate nerfspaceport
(nerfspaceport) ubuntu@ip-172-31-12-218:~/Softwares/nerf-spaceport$ ns-train nerfplayer-nerfacto --data /home/ubuntu/Softwares/data/wrapped_data_new_600
```

It is possible to get another error message as follows when you tried to start training:

```console
...
File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1611, in _write_ninja_file_and_build_library
    _write_ninja_file_to_build_library(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 2048, in _write_ninja_file_to_build_library
    _write_ninja_file(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 2102, in _write_ninja_file
    assert len(sources) > 0
AssertionError
```
It is caused because of some missing source files in nerfplayer installation.
To handle this error from the link:
https://github.com/Spaceport-Project/nerfplayer-spaceport/tree/main/nerfplayer/cuda 
copy csrc file to the cuda folder under nerfplayer installation path. To do so:

```bash
cd ~/nerfplayer-spaceport/nerfplayer/cuda
sudo cp -r csrc /opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer/cuda
```

When we give another try to training, it returns no errors but also does not start training (normally it should start trainin in 5 mins, maybe longer if it is first successful training):

```console
...
No depth data found! Generating pseudodepth...
 Using psueodepth: forcing depth loss to be ranking loss.
 Loading pseudodata depth from cache!
split val is empty, using the 1st training image
 No depth data found! Generating pseudodepth...
 Using psueodepth: forcing depth loss to be ranking loss.
 Loading pseudodata depth from cache!
Setting up training dataset...
Caching all 500 images.
Setting up evaluation dataset...
Caching all 1 images.
/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `PeakSignalNoiseRatio` from `torchmetrics` was deprecated and will be removed in 2.0. Import `PeakSignalNoiseRatio` from `torchmetrics.image` instead.
  _future_warning(
╭─────────────────────────────────────────── Viewer ───────────────────────────────────────────╮
│        ╷                                                                                     │
│   HTTP │ https://viewer.nerf.studio/versions/23-05-15-1/?websocket_url=ws://localhost:7007   │
│        ╵                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
[NOTE] Not running eval iterations since only viewer is enabled.
Use --vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard} to run with eval.
No Nerfstudio checkpoint to load, so training from scratch.
Disabled comet/tensorboard/wandb event writers
nerfstudio field components: Setting up CUDA (This may take a few minutes the first time)
```

When we killed the process with Ctrl+C if we get the following on the console:

```console
File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1284, in load
    return _jit_compile(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1468, in _jit_compile
    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/_cpp_extension_versioner.py", line 46, in bump_version_if_changed
    hash_value = hash_build_arguments(hash_value, build_arguments)
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/_cpp_extension_versioner.py", line 24, in hash_build_arguments
    hash_value = update_hash(hash_value, argument)
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/_cpp_extension_versioner.py", line 10, in update_hash
    return seed ^ (hash(value) + 0x9e
```

In _jit_compile function at /opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py, 'extra_ldflags' and 'extra_include_paths' passed empty within the 'build_arguments' at call 'version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(...'

Add following lines in _jit_compile function:
```python
extra_ldflags, extra_include_paths  = [], []
extra_include_paths.append('/opt/conda/envs/nerfspaceport/include')
extra_include_paths.append('/usr/local/cuda/lib64')
extra_include_paths.append('/opt/conda/envs/nerfspaceport/lib')
extra_ldflags.append('-L/opt/conda/envs/nerfspaceport/lib64')
extra_ldflags.append('-lcudart')
```
Just before the following lines:
```python
version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(
        name,
        sources,
        build_arguments=[extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths],
        build_directory=build_directory,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
    )
```
When we killed the process with Ctrl+C if we get the following on the console:

```console
╭─────────────────────────────────────────── Viewer ───────────────────────────────────────────╮
│        ╷                                                                                     │
│   HTTP │ https://viewer.nerf.studio/versions/23-05-15-1/?websocket_url=ws://localhost:7007   │
│        ╵                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
[NOTE] Not running eval iterations since only viewer is enabled.
Use --vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard} to run with eval.
No Nerfstudio checkpoint to load, so training from scratch.
Disabled comet/tensorboard/wandb event writers
nerfstudio field components: CUDA set up, loading (should be quick)
 ^CTraceback (most recent call last):
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/scripts/train.py", line 191, in launch
    main_func(local_rank=0, world_size=world_size, config=config)
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/scripts/train.py", line 102, in train_loop
    trainer.train()
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/engine/trainer.py", line 259, in train
    loss, loss_dict, metrics_dict = self.train_iteration(step)
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/utils/profiler.py", line 127, in inner
    out = func(*args, **kwargs)
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/engine/trainer.py", line 479, in train_iteration
    _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/utils/profiler.py", line 127, in inner
    out = func(*args, **kwargs)
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/pipelines/base_pipeline.py", line 299, in 
get_train_loss_dict
    model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
  File "/home/hamit/anaconda3/envs/nerfspaceport/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in 
_call_impl
    return forward_call(*args, **kwargs)
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/models/base_model.py", line 142, in forward
    return self.get_outputs(ray_bundle)
  File "/home/hamit/Softwares/repos/nerfplayer-nerfstudio/nerfplayer/nerfplayer_nerfacto.py", line 182, in get_outputs
    ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
  File "/home/hamit/anaconda3/envs/nerfspaceport/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in 
_call_impl
    return forward_call(*args, **kwargs)
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/model_components/ray_samplers.py", line 50, in forward
    return self.generate_ray_samples(*args, **kwargs)
  File "/home/hamit/Softwares/repos/nerf-spaceport/nerfstudio/model_components/ray_samplers.py", line 607, in 
generate_ray_samples
    density = density_fns(ray_samples.frustums.get_positions())
  File "/home/hamit/Softwares/repos/nerfplayer-nerfstudio/nerfplayer/nerfplayer_nerfacto_field.py", line 127, in 
density_fn
    density, _ = self.get_density(ray_samples)
  File "/home/hamit/Softwares/repos/nerfplayer-nerfstudio/nerfplayer/nerfplayer_nerfacto_field.py", line 139, in 
get_density
    x = self.encoding(positions_flat, time_flat).to(positions)
  File "/home/hamit/anaconda3/envs/nerfspaceport/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in 
_call_impl
    return forward_call(*args, **kwargs)
  File "/home/hamit/Softwares/repos/nerfplayer-nerfstudio/nerfplayer/temporal_grid.py", line 342, in forward
    outputs = TemporalGridEncodeFunc.apply(
  File "/home/hamit/anaconda3/envs/nerfspaceport/lib/python3.8/site-packages/torch/autograd/function.py", line 506, in 
apply
    return super().apply(*args, **kwargs)  # type: ignore
  File "/home/hamit/anaconda3/envs/nerfspaceport/lib/python3.8/site-packages/torch/cuda/amp/autocast_mode.py", line 98, 
in decorate_fwd
    return fwd(*args, **kwargs)
  File "/home/hamit/Softwares/repos/nerfplayer-nerfstudio/nerfplayer/temporal_grid.py", line 88, in forward
    _C.temporal_grid_encode_forward(
  File "/home/hamit/Softwares/repos/nerfplayer-nerfstudio/nerfplayer/cuda/__init__.py", line 23, in call_cuda
    from ._backend import _C
  File "/home/hamit/Softwares/repos/nerfplayer-nerfstudio/nerfplayer/cuda/_backend.py", line 34, in <module>
    _C = load(
  File "/home/hamit/anaconda3/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1284, 
in load
    return _jit_compile(
  File "/home/hamit/anaconda3/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1523, 
in _jit_compile
    baton.wait()
  File "/home/hamit/anaconda3/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/file_baton.py", line 42, in 
wait
    time.sleep(self.wait_seconds)
KeyboardInterrupt

Printing profiling stats, from longest to shortest duration in seconds
Trainer.train_iteration: 173.1363            
VanillaPipeline.get_train_loss_dict: 173.1357 
```
In _jit_compile function at /opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py, baton.try_acquire() returns False, which results with infinite wait loop. To handle this set if check to True as follows:

```python
if version != old_version:
        baton = FileBaton(os.path.join(build_directory, 'lock'))
        if baton.try_acquire():
```
change with:
```python
if version != old_version:
        baton = FileBaton(os.path.join(build_directory, 'lock'))
        if True:
```

At this stage training again does not start and when we killed the process with Ctrl+C, If we get followings on the terminal:
```console
[NOTE] Not running eval iterations since only viewer is enabled.
Use --vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard} to run with eval.
No Nerfstudio checkpoint to load, so training from scratch.
Disabled comet/tensorboard/wandb event writers
nerfstudio field components: CUDA set up, loading (should be quick)
C^CTraceback (most recent call last):
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/scripts/train.py", line 191, in launch
    main_func(local_rank=0, world_size=world_size, config=config)
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/scripts/train.py", line 102, in train_loop
    trainer.train()
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/engine/trainer.py", line 259, in train
    loss, loss_dict, metrics_dict = self.train_iteration(step)
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/utils/profiler.py", line 127, in inner
    out = func(*args, **kwargs)
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/engine/trainer.py", line 479, in train_iteration
    _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/utils/profiler.py", line 127, in inner
    out = func(*args, **kwargs)
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/pipelines/base_pipeline.py", line 299, in get_train_loss_dict
    model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/models/base_model.py", line 142, in forward
    return self.get_outputs(ray_bundle)
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer/nerfplayer_nerfacto.py", line 182, in 
get_outputs
    ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/model_components/ray_samplers.py", line 50, in forward
    return self.generate_ray_samples(*args, **kwargs)
  File "/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/model_components/ray_samplers.py", line 607, in 
generate_ray_samples
    density = density_fns(ray_samples.frustums.get_positions())
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer/nerfplayer_nerfacto_field.py", line 127, in
density_fn
    density, _ = self.get_density(ray_samples)
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer/nerfplayer_nerfacto_field.py", line 139, in
get_density
    x = self.encoding(positions_flat, time_flat).to(positions)
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer/temporal_grid.py", line 342, in forward
    outputs = TemporalGridEncodeFunc.apply(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/autograd/function.py", line 506, in apply
    return super().apply(*args, **kwargs)  # type: ignore
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/cuda/amp/autocast_mode.py", line 98, in 
decorate_fwd
    return fwd(*args, **kwargs)
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer/temporal_grid.py", line 88, in forward
    _C.temporal_grid_encode_forward(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer/cuda/__init__.py", line 23, in call_cuda
    from ._backend import _C
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/nerfplayer/cuda/_backend.py", line 37, in <module>
    _C = load(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1285, in load
    return _jit_compile(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1522, in 
_jit_compile
    _write_ninja_file_and_build_library(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1637, in 
_write_ninja_file_and_build_library
    _run_ninja_build(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 1908, in 
_run_ninja_build
    subprocess.run(
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/subprocess.py", line 495, in run
    stdout, stderr = process.communicate(input, timeout=timeout)
  File "/opt/conda/envs/nerfspaceport/lib/python3.8/subprocess.py", line 1015, in communicate
    stdout = self.stdout.read()
KeyboardInterrupt

Printing profiling stats, from longest to shortest duration in seconds
Trainer.train_iteration: 306.7058            
VanillaPipeline.get_train_loss_dict: 306.7046 
```

When we debug it, in _run_ninja_build fuction num_workers is set to None, to handle this change following lines again in the same file(cpp_extension.py):

```python
def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    ...
```

with
```python
def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    num_workers = 16 # set it manually depending on number of cpu core
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
```

After handling all those faced problems when we start training it should start properly :
```console
(nerfspaceport) ubuntu@ip-172-31-12-218:~/Softwares/nerf-spaceport$ ns-train nerfplayer-nerfacto --data /home/ubuntu/Softwares/data/wrapped_data_new_600

...
           Saving config to:                                                                    experiment_config.py:141
           outputs/wrapped_data_new_600/nerfplayer-nerfacto/2023-10-25_083122/config.yml                                
           Saving checkpoints to:                                                                         trainer.py:134
           outputs/wrapped_data_new_600/nerfplayer-nerfacto/2023-10-25_083122/nerfstudio_models                         
/home/ubuntu/Softwares/nerf-spaceport/nerfstudio/data/dataparsers/nerfplayer_multicam_dataparser.py:280: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  [torch.tensor(c[k], dtype=torch.float32) for c in cams])
 No depth data found! Generating pseudodepth...
 Using psueodepth: forcing depth loss to be ranking loss.
 Loading pseudodata depth from cache!
split val is empty, using the 1st training image
 No depth data found! Generating pseudodepth...
 Using psueodepth: forcing depth loss to be ranking loss.
 Loading pseudodata depth from cache!
Setting up training dataset...
Caching all 500 images.
Setting up evaluation dataset...
Caching all 1 images.
/opt/conda/envs/nerfspaceport/lib/python3.8/site-packages/torchmetrics/utilities/prints.py:62: FutureWarning: Importing `PeakSignalNoiseRatio` from `torchmetrics` was deprecated and will be removed in 2.0. Import `PeakSignalNoiseRatio` from `torchmetrics.image` instead.
  _future_warning(
╭─────────────────────────────────────────── Viewer ───────────────────────────────────────────╮
│        ╷                                                                                     │
│   HTTP │ https://viewer.nerf.studio/versions/23-05-15-1/?websocket_url=ws://localhost:7007   │
│        ╵                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
[NOTE] Not running eval iterations since only viewer is enabled.
Use --vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard} to run with eval.
No Nerfstudio checkpoint to load, so training from scratch.
Disabled comet/tensorboard/wandb event writers
nerfstudio field components: CUDA set up, loading (should be quick)
[08:32:27] Printing max of 10 lines. Set flag --logging.local-writer.max-log-size=0 to disable line        writer.py:438
           wrapping.                                                                                                    
Step (% Done)       Train Iter (time)    ETA (time)           Train Rays / Sec                       
-----------------------------------------------------------------------------------                  
7710 (25.70%)       107.563 ms           39 m, 57 s           38.30 K                                
7720 (25.73%)       107.565 ms           39 m, 56 s           38.30 K                                
7730 (25.77%)       108.736 ms           40 m, 21 s           37.95 K                                
7740 (25.80%)       107.473 ms           39 m, 52 s           38.34 K                                
7750 (25.83%)       107.504 ms           39 m, 51 s           38.33 K                                
7760 (25.87%)       108.854 ms           40 m, 20 s           37.91 K                                
7770 (25.90%)       107.583 ms           39 m, 51 s           38.30 K                                
7780 (25.93%)       107.517 ms           39 m, 49 s           38.32 K                                
7790 (25.97%)       108.793 ms           40 m, 16 s           37.93 K                                
7800 (26.00%)       107.539 ms           39 m, 47 s           38.31 K                                
---------------------------------------------------------------------------------------------------- 
Viewer at: https://viewer.nerf.studio/versions/23-05-15-1/?websocket_url=ws://localhost:7007         
```
