import torch
import weakref
import torch.nn as nn
import gc
from ..bench.time_bench import test_time
from ..common.global_values import GlobalValues
import math


def slice_run(
    func,
    input: torch.Tensor,
    output: torch.Tensor,
    in_dim=0,
    batch_num=2,
    out_dim=None,
):
    if out_dim is None:
        out_dim = in_dim

    in_shape = input.shape[in_dim]
    out_shape = output.shape[out_dim]

    batch_num = min(batch_num, in_shape, out_shape)
    in_stride = math.ceil(in_shape / batch_num)

    if in_shape >= out_shape:
        assert in_shape % out_shape == 0, "bad shape"
        radio = in_shape // out_shape

        out_stride = in_stride // radio
    else:
        assert out_shape % in_shape == 0, "bad shape"
        radio = out_shape // in_shape

        out_stride = in_stride * radio

    for i in range(batch_num):
        start_in = i * in_stride
        end_in = min((i + 1) * in_stride, in_shape)

        start_out = i * out_stride
        end_out = min((i + 1) * out_stride, out_shape)

        if end_in <= start_in or end_out <= start_out:
            break
        out_mem = output.narrow(
            dim=out_dim, start=start_out, length=(end_out - start_out)
        )
        input_mem = input.narrow(dim=in_dim, start=start_in, length=(end_in - start_in))

        out_mem[...] = func(input_mem)


def flush_memory():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@test_time(GlobalValues.ENABLE_PER)
def malloc_pin_memory(module: torch.nn.Module, ref_device=None):
    if ref_device is not None:
        module.to(ref_device)

    for module_name, sub_module in module.named_modules():
        if hasattr(sub_module, "pin_param"):
            pin_param = getattr(sub_module, "pin_param")
        else:
            pin_param = {}
        for name, param in sub_module.named_parameters(recurse=False):
            if name in pin_param:
                continue
            else:
                pin_param[name] = param.cpu().contiguous().pin_memory()
                print(f"register @{module_name}.{name} pin-memory.")
        setattr(sub_module, "pin_param", pin_param)


def register_runtime_device(module: nn.Module, device):
    for _, sub_module in module.named_modules():
        setattr(sub_module, "runtime_device", device)


def register_dynamic_offload(module: nn.Module, device):
    for _, sub_module in module.named_modules():
        warp_dynamic_offload(sub_module, device=device)


def warp_dynamic_offload(
    module: nn.Module, device, func_name="forward", container_sub=False
):
    origin_func = getattr(module, func_name)

    def new_func(*args, **kwargs):
        cuda_module(module, device=device, contain_sub=container_sub)
        result = origin_func(*args, **kwargs)
        offload_module(module)
        return result

    setattr(module, func_name, new_func)


@test_time(GlobalValues.ENABLE_PER)
def offload_module(module: torch.nn.Module):
    for _, sub_module in module.named_modules():
        if hasattr(sub_module, "pin_param"):
            pin_param = getattr(sub_module, "pin_param")
        else:
            pin_param = {}
        for name, param in sub_module.named_parameters(recurse=False):
            if name not in pin_param:
                continue
            else:
                param.data = pin_param[name]


@test_time(GlobalValues.ENABLE_PER)
def cuda_module(module: torch.nn.Module, device, contain_sub=False):
    cached_size = torch.cuda.memory_reserved(device)
    if cached_size > 4 * 1024**3:
        gc.collect()
        torch.cuda.empty_cache()

    for module_name, sub_module in module.named_modules():
        if not contain_sub:
            if len(module_name) > 0:
                continue

        assert hasattr(
            sub_module, "pin_param"
        ), "you need to register pin_param by malloc_pin_memory first"

        pin_param = getattr(sub_module, "pin_param")
        for name, param in sub_module.named_parameters(recurse=False):
            if name not in pin_param:
                print(f"warning: dismiss pin_param {name}")
                continue
            else:
                param.data = pin_param[name].to(device=device, non_blocking=True)
    torch.cuda.synchronize(device=device)


def wrap_forward(cls):
    def new_forward(self, *args, **kwargs):
        cls.memory_hook.load(offload_pre=False)
        result = cls.forward(self, *args, **kwargs)
        cls.memory_hook.offload(load_next=True)
        return result

    cls.forward = new_forward


class LowMemoryHook:
    def __init__(
        self,
        module,
        execute_device,
        func_names=["forward"],
        offload_device=torch.device("cpu"),
    ):
        self.weak_module = weakref.ref(module)
        # self.module, self.offload_hook = accelerate.cpu_offload_with_hook(module,execution_device=execute_device)
        self.offload_device = offload_device
        self.execute_device = execute_device
        for func_name in func_names:
            self.wrap_forword(func_name)
        setattr(self.weak_module(), "memory_hook", self)

    def load(self):
        if self.weak_module() is not None:
            self.weak_module().to(self.execute_device)

    def offload(self, empty_cache=True):
        if self.weak_module() is not None:
            # self.offload_hook.offload()
            self.weak_module().to(self.offload_device)

        if empty_cache:
            self.FlushMemory()

    def wrap_forword(self, func_name="forward"):
        origin_func = getattr(self.weak_module(), func_name)

        def new_forward(*args, **kwargs):
            self.load()
            result = origin_func(*args, **kwargs)
            self.offload(empty_cache=True)
            return result

        setattr(self.weak_module(), func_name, new_forward)

    def FlushMemory(self):
        flush_memory()
