import sys, time; sys.path.insert(0,'.')
import torch, train_dev as T
N=int(sys.argv[1]) if len(sys.argv)>1 else 4096
env=T.GpuBallbot(N,seed=0,dr=True,tiles=True,surfaces=True,goal_p=0.4,pull=True,hist_k=4,asym=True)
a=torch.zeros(N,3,device="cuda")
def tm(f,n=40):
    for _ in range(3): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
phys=tm(lambda: env.be.step(5))
full=tm(lambda: env.step(a))
o=env.noisy(env.obs())
nz=tm(lambda: env.noisy(env.obs()))
rst=tm(lambda: env.reset(torch.rand(N,device="cuda")<0.01))
print(f"N={N}: physics {phys:.1f} ms | full env.step {full:.1f} ms | noisy+obs {nz:.1f} ms | reset(1%) {rst:.1f} ms | env-steps/s (env only) {N/full*1e3:.0f}")
