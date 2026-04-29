import torch
import gc

def reset_cuda():
    """重置 CUDA 状态"""
    if torch.cuda.is_available():
        # 清空缓存
        torch.cuda.empty_cache()
        # 强制垃圾回收
        gc.collect()
        # 重置设备
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print("CUDA 状态已重置")

# 在代码开头调用
reset_cuda()