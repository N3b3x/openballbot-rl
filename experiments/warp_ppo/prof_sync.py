import sys, warnings, traceback; sys.path.insert(0,'.')
import torch, train_dev as T
N=4096
env=T.GpuBallbot(N,seed=0,dr=True,tiles=True,surfaces=True,goal_p=0.4,pull=True,hist_k=4,asym=True)
a=torch.zeros(N,3,device="cuda")
for _ in range(3): env.step(a)
env.reset(torch.rand(N,device="cuda")<0.01)
import collections
cnt=collections.Counter()
def hook(message, category, filename, lineno, file=None, line=None):
    cnt[(filename.split("\\")[-1], lineno)]+=1
warnings.showwarning=hook
torch.cuda.set_sync_debug_mode("warn")
warnings.simplefilter("always")
env.step(a); n1=sum(cnt.values()); 
env.reset(torch.rand(N,device="cuda")<0.01); n2=sum(cnt.values())-n1
torch.cuda.set_sync_debug_mode("default")
print("syncs in step:",n1," in reset:",n2)
for k,v in cnt.most_common(12): print(k,v)
