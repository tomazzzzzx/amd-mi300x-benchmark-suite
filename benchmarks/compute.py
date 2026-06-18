import torch, time

def gemm_bench(M, N, K, dtype=torch.float16, iters=100):
    A, B = torch.randn(M,K,dtype=dtype,device="cuda"), torch.randn(K,N,dtype=dtype,device="cuda")
    for _ in range(10): torch.mm(A,B)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): torch.mm(A,B)
    torch.cuda.synchronize()
    tflops = (2*M*N*K*iters)/(time.perf_counter()-t0)/1e12
    print(f"GEMM ({M}x{N}x{K}): {tflops:.2f} TFLOPS")
    return tflops
