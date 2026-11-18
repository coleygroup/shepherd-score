"""Record and replay Triton launch configurations for reproducible measurements."""
from functools import lru_cache
import json
import os
from pathlib import Path
from types import MethodType

import triton

_KERNELS = {}


def _hardware():
    import torch
    return dict(gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),
                triton=triton.__version__)


@lru_cache(maxsize=None)
def _profile(path, hardware_json):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 1 or data['hardware'] != json.loads(hardware_json):
        raise ValueError('Triton profile requires its recorded GPU architecture and Triton version')
    return data


def _key(tuner, args, kwargs):
    values = dict(zip(tuner.arg_names, args))
    values.update(kwargs)
    values = {k:v for k,v in values.items() if k in tuner.arg_names}
    return tuple([values[k] for k in tuner.keys if k in values] +
                 [str(v.dtype) for v in values.values() if hasattr(v, 'dtype')])


def autotune(*args, **kwargs):
    """Triton's decorator, with optional strict replay via FSS_TRITON_CONFIGS.

    Unlisted shapes fail instead of silently tuning. The profile is read before
    capture, and worker processes inherit the same file through the environment.
    With no profile, Triton's normal autotuning and cache behavior are retained.
    """
    decorate = triton.autotune(*args, **kwargs)
    def apply(fn):
        tuner = decorate(fn)
        name = fn.__module__ + '.' + fn.__name__
        _KERNELS[name] = tuner
        original = tuner.run
        selected = os.environ.get('FSS_TRITON_CONFIGS')
        if selected:
            validated_devices = set()
            def replay(self, *call_args, **call_kwargs):
                import torch
                device = torch.cuda.current_device()
                if device not in validated_devices:
                    _profile(selected, json.dumps(_hardware(), sort_keys=True))
                    validated_devices.add(device)
                key = _key(self, call_args, call_kwargs)
                if key not in self.cache:
                    profile = _profile(selected, json.dumps(_hardware(), sort_keys=True))
                    entries = profile['kernels'].get(name, [])
                    entry = next((e for e in entries if tuple(e['key']) == key), None)
                    if entry is None:
                        raise ValueError(f'Missing frozen Triton configuration: {name}, {key}')
                    config = triton.Config(**entry['config'])
                    if not any(config.all_kwargs() == c.all_kwargs() for c in self.configs):
                        raise ValueError(f'Frozen configuration is not a supported candidate: {name}')
                    self.cache[key] = config
                return original(*call_args, **call_kwargs)
            tuner.run = MethodType(replay, tuner)
        return tuner
    return apply


def export_configurations(path):
    """Save configurations used so far; call after untimed workload preparation.

    Merge profiles from separate workloads only when overlapping entries agree.
    This records tuning choices; it does not claim they are optimal.
    """
    kernels = {}
    for name, tuner in sorted(_KERNELS.items()):
        entries = []
        for key, config in tuner.cache.items():
            if config.pre_hook is not None:
                raise ValueError('Cannot serialize a configuration with a pre-hook')
            entries.append(dict(key=list(key), config=dict(kwargs=config.kwargs,
                num_warps=config.num_warps, num_stages=config.num_stages,
                num_ctas=config.num_ctas, maxnreg=config.maxnreg)))
        if entries:
            kernels[name] = sorted(entries, key=lambda e:json.dumps(e['key']))
    if not kernels:
        raise ValueError('No Triton configurations have been used')
    data = dict(schema=1, hardware=_hardware(), kernels=kernels)
    Path(path).write_text(json.dumps(data, indent=2) + '\n')
    return data
