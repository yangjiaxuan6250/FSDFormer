# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
import os.path as osp
import pkgutil
import re
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory
from glob import glob
import torch
import torchvision
from torch.optim import Optimizer

import mmcv
from ..fileio import FileClient
from ..fileio import load as load_file
from ..parallel import is_module_wrapper
from ..utils import load_url, mkdir_or_exist, print_log
from .dist_utils import get_dist_info

ENV_MMCV_HOME = 'MMCV_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def _get_mmcv_home():
    mmcv_home = os.path.expanduser(
        os.getenv(
            ENV_MMCV_HOME,
            os.path.join(
                os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmcv')))

    mkdir_or_exist(mmcv_home)
    return mmcv_home


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    if hasattr(module, 'train'):
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
    else:
        for name in state_dict.keys():
            metadata = getattr(state_dict[name], '_metadata', None)
            state_dict[name] = state_dict[name].copy()
            if metadata is not None:
                state_dict[name]._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if not hasattr(module, '_load_from_state_dict'):
            for name, m in module.model.items():
                if is_module_wrapper(m):
                    m = m.module
                local_metadata = {} if metadata is None else metadata.get(
                    prefix[:-1], {})
                m._load_from_state_dict(state_dict[name], prefix, local_metadata, True,
                                        all_missing_keys, unexpected_keys,
                                        err_msg)
                for name, child in m._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')
        else:
            if is_module_wrapper(module):
                module = module.module
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                         all_missing_keys, unexpected_keys,
                                         err_msg)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def get_external_models():
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)

    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmcv.__path__[0], 'model_zoo/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)

    return mmcls_urls


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmcv.__path__[0],
                                   'model_zoo/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)

    return deprecate_urls


def _process_mmcls_checkpoint(checkpoint):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Some checkpoints converted from 3rd-party repo don't
        # have the "state_dict" key.
        state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)

    return new_checkpoint


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger=None):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Default: None
            logger (:mod:`logging.Logger`, optional): The logger for message.
                Default: None

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        print_log(
            f'load checkpoint from {class_name[10:]} path: {filename}', logger=logger)
        return checkpoint_loader(filename, map_location)


@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(filename, map_location):
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('http://', 'https://'))
def load_from_http(filename, map_location=None, model_dir=None):
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (string, optional): directory in which to save the object,
            Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(
            filename, model_dir=model_dir, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = load_url(
                filename, model_dir=model_dir, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes='pavi://')
def load_from_pavi(filename, map_location=None):
    """load checkpoint through the file path prefixed with pavi. In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Args:
        filename (str): checkpoint file path with pavi prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    assert filename.startswith('pavi://'), \
        f'Expected filename startswith `pavi://`, but get {filename}'
    model_path = filename[7:]

    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')

    model = modelcloud.get(model_path)
    with TemporaryDirectory() as tmp_dir:
        downloaded_file = osp.join(tmp_dir, model.name)
        model.download(downloaded_file)
        checkpoint = torch.load(downloaded_file, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=r'(\S+\:)?s3://')
def load_from_ceph(filename, map_location=None, backend='petrel'):
    """load checkpoint through the file path prefixed with s3.  In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Note:
        Since v1.4.1, the registered scheme prefixes have been enhanced to
        support bucket names in the path prefix, e.g. 's3://xx.xx/xx.path',
        'bucket1:s3://xx.xx/xx.path'.

    Args:
        filename (str): checkpoint file path with s3 prefix
        map_location (str, optional): Same as :func:`torch.load`.
        backend (str, optional): The storage backend type. Options are 'ceph',
            'petrel'. Default: 'petrel'.

    .. warning::
        :class:`mmcv.fileio.file_client.CephBackend` will be deprecated,
        please use :class:`mmcv.fileio.file_client.PetrelBackend` instead.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    allowed_backends = ['ceph', 'petrel']
    if backend not in allowed_backends:
        raise ValueError(f'Load from Backend {backend} is not supported.')

    if backend == 'ceph':
        warnings.warn(
            'CephBackend will be deprecated, please use PetrelBackend instead',
            DeprecationWarning)

    # CephClient and PetrelBackend have the same prefix 's3://' and the latter
    # will be chosen as default. If PetrelBackend can not be instantiated
    # successfully, the CephClient will be chosen.
    try:
        file_client = FileClient(backend=backend)
    except ImportError:
        allowed_backends.remove(backend)
        file_client = FileClient(backend=allowed_backends[0])

    with io.BytesIO(file_client.get(filename)) as buffer:
        checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('modelzoo://', 'torchvision://'))
def load_from_torchvision(filename, map_location=None):
    """load checkpoint through the file path prefixed with modelzoo or
    torchvision.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    model_urls = get_torchvision_models()
    if filename.startswith('modelzoo://'):
        warnings.warn(
            'The URL scheme of "modelzoo://" is deprecated, please '
            'use "torchvision://" instead', DeprecationWarning)
        model_name = filename[11:]
    else:
        model_name = filename[14:]
    return load_from_http(model_urls[model_name], map_location=map_location)


@CheckpointLoader.register_scheme(prefixes=('open-mmlab://', 'openmmlab://'))
def load_from_openmmlab(filename, map_location=None):
    """load checkpoint through the file path prefixed with open-mmlab or
    openmmlab.

    Args:
        filename (str): checkpoint file path with open-mmlab or
        openmmlab prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_external_models()
    prefix_str = 'open-mmlab://'
    if filename.startswith(prefix_str):
        model_name = filename[13:]
    else:
        model_name = filename[12:]
        prefix_str = 'openmmlab://'

    deprecated_urls = get_deprecated_model_names()
    if model_name in deprecated_urls:
        warnings.warn(
            f'{prefix_str}{model_name} is deprecated in favor '
            f'of {prefix_str}{deprecated_urls[model_name]}',
            DeprecationWarning)
        model_name = deprecated_urls[model_name]
    model_url = model_urls[model_name]
    # check if is url
    if model_url.startswith(('http://', 'https://')):
        checkpoint = load_from_http(model_url, map_location=map_location)
    else:
        filename = osp.join(_get_mmcv_home(), model_url)
        if not osp.isfile(filename):
            raise FileNotFoundError(f'{filename} can not be found.')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes='mmcls://')
def load_from_mmcls(filename, map_location=None):
    """load checkpoint through the file path prefixed with mmcls.

    Args:
        filename (str): checkpoint file path with mmcls prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_mmcls_models()
    model_name = filename[8:]
    checkpoint = load_from_http(
        model_urls[model_name], map_location=map_location)
    checkpoint = _process_mmcls_checkpoint(checkpoint)
    return checkpoint


def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str, optional): Same as :func:`torch.load`.
           Default: None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Default: None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
           OrderedDict storing model weights or a dict containing other
           information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


def _load_checkpoint_with_prefix(prefix, filename, map_location=None):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict


def get_checkpoint_dir(OUT_DIR):
    """Retrieves the location for storing checkpoints."""
    return os.path.join(OUT_DIR)


def get_last_checkpoint(OUT_DIR, NAME_PREFIX="amp_model_best", logger=None):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir(OUT_DIR)
    checkpoints = glob(checkpoint_dir + f"/{NAME_PREFIX}*")
    if len(checkpoints) > 0:
        last_checkpoint_name = sorted(checkpoints)[-1]
        last_checkpoint = os.path.join(checkpoint_dir, last_checkpoint_name)
        print_log(f"loading last_checkpoint file: {last_checkpoint}", logger=logger)
        return last_checkpoint
    else:
        return None


def get_best_k_model(OUT_DIR, indicator, _NAME_PREFIX="ckpt_ep_"):
    best_k_models = []
    best_fname = []
    if os.path.isfile(OUT_DIR):
        with open(OUT_DIR, 'r') as f:
            # TODO: 通常checkpoint不会很大，如果太大open方法不合适，因为我们需要的是最后几行，o1pen是从头遍历的
            stats2user = [line.strip('\n') for line in f.readlines()]

        for line in stats2user:
            metric = {}
            line = line.split(',')
            for v in line[1:]:
                v = re.sub(r"[{}'' ]", "", v)
                k, v = v.split(':')
                metric[str(k)] = v
            # fname, v = line.split('-')
            # epoch = line[0].replace('.pth.tar', '')
            epoch = line[0].replace('model_best_', '')
            # best_k_models[str(epoch)] = float(v)

            # 第一个line[0]用于save_top_k触发时, 删除多余的模型
            # 第二个用于加载best模型, 因为best_k_models是个[[]],索引不方便
            best_k_models.append([epoch, metric, line[0]])
            best_fname.append(line[0])
            # best_k_models['epoch'].append(epoch)
            # best_k_models[indicator].append(float(v))

    if len(best_k_models) == 0:
        msg = f"checkpoint in directory {OUT_DIR} don't exist or is empty"
        warnings.warn(msg)

    return best_k_models, best_fname


def load_checkpoint(resume_mode,
                    work_dir,
                    model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    ################
    # 只从work_dir里读ckpt，用于模型的继续训练
    ################
    if not os.path.isfile(filename):

        resume_mode = resume_mode.lower()
        if resume_mode == 'best':
            _, best_k_fname = get_best_k_model(os.path.join(work_dir, "checkpoint"), None)
            if len(best_k_fname) > 0:
                best_k_model = sorted(best_k_fname)[-1]
                filename = os.path.join(work_dir, best_k_model)
            else:
                print_log("loading best model failed, maybe it's from scratch currently.", logger=logger)

        elif resume_mode == 'auto':
            ckpt = get_last_checkpoint(work_dir)
            if ckpt is not None:
                filename = ckpt
        # if 'torchvision' not in filename and 'open-mmlab' not in filename:
        #     print_log(f"no checkpoint found at {filename}", logger=logger)
        #     return {'meta': {'epoch': 1,
        #                      'iter': 1,
        #                      'best_epoch': 1,
        #                      'best_metric': None}}
        if filename == '':
            print_log(f"no checkpoint found at {filename}", logger=logger)
            return {'meta': {'epoch': 1,
                            'iter': 1,
                            'best_epoch': 1,
                            'best_metric': None}}
    ################

    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    if 'meta' not in checkpoint.keys():
        checkpoint['meta'] = {}
    if hasattr(model, 'train'):
        mod = {'model': model}
        checkpoint = {'model': checkpoint}
    else:
        mod = model.model
    if isinstance(mod, dict):
        for name, m in mod.items():
            if 'state_dict' in checkpoint[name]:
                state_dict = checkpoint[name]['state_dict']
            else:
                state_dict = checkpoint[name]
            checkpoint = checkpoint.pop(name)
            # strip prefix of state_dict
            metadata = getattr(state_dict, '_metadata', OrderedDict())
            for p, r in revise_keys:
                state_dict = OrderedDict(
                    {re.sub(p, r, k): v
                     for k, v in state_dict.items()})
            # Keep metadata in state_dict
            state_dict._metadata = metadata
            load_state_dict(m, state_dict, strict, logger)
    else:
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # strip prefix of state_dict
        metadata = getattr(state_dict, '_metadata', OrderedDict())
        for p, r in revise_keys:
            state_dict = OrderedDict(
                {re.sub(p, r, k): v
                 for k, v in state_dict.items()})
        # Keep metadata in state_dict
        state_dict._metadata = metadata
        load_state_dict(mod, state_dict, strict, logger)
        # if optimizer is not None:
        #     if checkpoint.get('optimizer') is not None:
        #         optimizer.load_state_dict(checkpoint['optimizer'])
        #
        #     if lr > 0 and reset_lr:
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lr
        #     print_log("loaded checkpoint.optimizer")

        # load state_dict
        checkpoint['meta'].setdefault('epoch', 0) + 1
        checkpoint['meta'].setdefault('iter', 0) + 1
        checkpoint['meta'].setdefault('best_epoch', 0)
        checkpoint['meta'].setdefault('best_metric', None)

    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.

    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(#model,
                    filename,
                    #optimizer=None,
                    meta=None,
                    file_client_args=None):
    """Save checkpoint to file.
    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.
    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    # meta.update(mmcv_version=mmcv.__version__, time=time.asctime())
    if 'model' not in meta:
        checkpoint = {}
        for name,  sub_meta in meta.items(): # i.e. stats

            model = sub_meta.pop('model')
            optimizer = sub_meta.pop('optimizer')
            if is_module_wrapper(model):
                model = model.module

            if hasattr(model, 'CLASSES') and model.CLASSES is not None:
                # save class name to the meta
                sub_meta.update(CLASSES=model.CLASSES)
            checkpoint[name] = {
                'meta': sub_meta,
                'state_dict': weights_to_cpu(get_state_dict(model))
            }

            # save optimizer state dict in the checkpoint
            if isinstance(optimizer, Optimizer):
                checkpoint[name]['optimizer'] = optimizer.state_dict()

            file_client = FileClient.infer_client(file_client_args, filename)
            with io.BytesIO() as f:
                torch.save(checkpoint, f)
                file_client.put(f.getvalue(), filename)

    else:
        model = meta.pop('model')
        optimizer = meta.pop('optimizer')
        if is_module_wrapper(model):
            model = model.module

        if hasattr(model, 'CLASSES') and model.CLASSES is not None:
            # save class name to the meta
            meta.update(CLASSES=model.CLASSES)

        state_dict = {}
        if hasattr(model, 'model'):
            model = model.model

        if isinstance(model, dict):
            for name, m in model.items():
                state_dict.update({name: weights_to_cpu(get_state_dict(m))})

        checkpoint = {
            'meta': meta,
            'state_dict': state_dict if bool(state_dict) else weights_to_cpu(get_state_dict(model))
        }
        # save optimizer state dict in the checkpoint
        if isinstance(optimizer, Optimizer):
            checkpoint['optimizer'] = optimizer.state_dict()
        elif isinstance(optimizer, dict):
            checkpoint['optimizer'] = {}
            for name, optim in optimizer.items():
                checkpoint['optimizer'][name] = optim.state_dict()

        if filename.startswith('pavi://'):
            if file_client_args is not None:
                raise ValueError(
                    'file_client_args should be "None" if filename starts with'
                    f'"pavi://", but got {file_client_args}')
            try:
                from pavi import exception, modelcloud
            except ImportError:
                raise ImportError(
                    'Please install pavi to load checkpoint from modelcloud.')
            model_path = filename[7:]
            root = modelcloud.Folder()
            model_dir, model_name = osp.split(model_path)
            try:
                model = modelcloud.get(model_dir)
            except exception.NodeNotFoundError:
                model = root.create_training_model(model_dir)
            with TemporaryDirectory() as tmp_dir:
                checkpoint_file = osp.join(tmp_dir, model_name)
                with open(checkpoint_file, 'wb') as f:
                    torch.save(checkpoint, f)
                    f.flush()
                model.create_file(checkpoint_file, name=model_name)
        else:
            file_client = FileClient.infer_client(file_client_args, filename)
            with io.BytesIO() as f:
                torch.save(checkpoint, f)
                file_client.put(f.getvalue(), filename)