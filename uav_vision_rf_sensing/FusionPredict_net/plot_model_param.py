import torch, time, psutil, os
from thop import profile, clever_format
from tabulate import tabulate

def full_report(model,
                input_tuple,
                device='auto',
                n_warmup=10,
                n_test=30):
    """
    input_tuple : 任意数量的 dummy 输入张量，以 tuple 形式传入
                  例：(torch.randn(1,3,224,224), torch.randn(1,64))
    device      : 'auto' | 'cpu' | 'cuda'
    """
    # 1. 自动选设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval().to(device)
    dummy = tuple(t.to(device) for t in input_tuple)

    # 2. 参数量 & FLOPs
    macs, params = profile(model, inputs=dummy, verbose=False)
    flops, params_str = clever_format([macs * 2, params], "%.3f")

    # 3. 可训练 / 不可训练
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # 4. 模型大小 (fp32)
    model_size_mb = params * 4 / (1024 ** 2)

    # 5. 峰值显存 / 内存
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        _ = model(*dummy)                # 前向一次
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    # 6. 推理耗时
    for _ in range(n_warmup):
        _ = model(*dummy)
    torch.cuda.synchronize() if device == 'cuda' else None
    tic = time.time()
    for _ in range(n_test):
        _ = model(*dummy)
    torch.cuda.synchronize() if device == 'cuda' else None
    avg_latency = (time.time() - tic) / n_test * 1000  # ms

    # 7. 各层明细
    layer_table = [(i+1, name, p.numel(), "✓" if p.requires_grad else "✗")
                   for i, (name, p) in enumerate(model.named_parameters())]

    # 8. 打印总览
    print("\n================== 模型完整报告 ==================")
    overview = [
        ("总参数量",     params_str),
        ("可训练参数",   f"{trainable/1e6:.3f} M"),
        ("冻结参数",     f"{frozen/1e6:.3f} M"),
        ("FLOPs",        flops),
        ("模型大小(fp32)", f"{model_size_mb:.2f} MB"),
        ("峰值显存/内存", f"{peak_mem:.2f} MB"),
        ("平均推理耗时",  f"{avg_latency:.2f} ms"),
    ]
    print(tabulate(overview, tablefmt="rounded_outline"))

    # 9. 打印层明细
    print("\n================== 各层参数量明细 ==================")
    print(tabulate(layer_table,
                   headers=["#", "层名", "参数量", "可训练"],
                   tablefmt="rounded_grid"))

    return {
        "params_total": params_str,
        "params_train": trainable,
        "params_frozen": frozen,
        "flops": flops,
        "model_size_mb": model_size_mb,
        "peak_mem_mb": peak_mem,
        "latency_ms": avg_latency,
        "layer_table": layer_table
    }


# ================== DEMO ==================
if __name__ == '__main__':
    # 1. Vision_Net（三输入示例）
    from Calibration_nets.vision_net_est_ab_light import Vision_Net
    print("\n[Vision_Net]")
    net1 = Vision_Net()
    inputs1 = (torch.randn(1, 3, 64, 64),
               torch.randn(1, 2),
               torch.randn(1, 2))
    full_report(net1, inputs1)

    # 2. fuse_net_VisionFirstCome_pred_ab_nores（两输入）
    from fuse_net_VisionFirstCome_pred_a_1 import PredModel
    print("\n[VisionFirst Fuse]")
    net2 = PredModel()
    inputs2 = (torch.randn(1, 7, 2),
               torch.randn(1, 7, 3))
    full_report(net2, inputs2)

    # 3. Signal_net_EchoOnly_pred_ab（两输入）
    from Signal_net_EchoOnly_pred_a_1 import PredModel
    print("\n[EchoOnly]")
    net3 = PredModel()
    inputs3 = (#torch.randn(1, 7, 2),
               torch.randn(1, 1, 7, 3))
    full_report(net3, inputs3)


    from fuseNo_net_VisionFirstCome_pred_a_1 import PredModel
    print("\n[VisionFirst But No Fuse]")
    net2 = PredModel()
    inputs2 = (torch.randn(1, 7, 2),
               torch.randn(1, 7, 3))
    full_report(net2, inputs2)