import torch, time

def measure_bandwidth(size_gb=8.0, iters=100):
    n = int(size_gb*1024**3)//4
    src = torch.randn(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(10): dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): dst.copy_(src)
    torch.cuda.synchronize()
    bw = (size_gb*iters)/(time.perf_counter()-t0)
    print(f"Memory BW: {bw:.2f} GB/s")
    return bw
