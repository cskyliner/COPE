import itertools, logging, os, sys, time
sys.path.insert(0,'/root/COPE')
import torch
from config import cfg
from cope.dataloader import make_dataloader
from cope.model import make_model
from cope.optimizer import make_optimizer
from cope.processor_train import train
from solver.make_optimizer import make_optimizer_1stage
from solver.lr_scheduler import WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from train_cope import set_seed
from utils.logger import setup_logger

OriginalScaler = torch.cuda.amp.GradScaler
class TrackedScaler(OriginalScaler):
 def step(self, optimizer, *args, **kwargs):
  self._old_scale = self.get_scale()
  return super().step(optimizer, *args, **kwargs)
 def update(self, *args, **kwargs):
  super().update(*args, **kwargs)
  print('SCALER_UPDATE',self._old_scale,'->',self.get_scale(),'peak_mib',round(torch.cuda.max_memory_allocated()/1048576,1),flush=True)
torch.cuda.amp.GradScaler = TrackedScaler

class LimitedLoader:
 def __init__(self,loader,limit): self.loader=loader; self.limit=limit
 def __iter__(self): return itertools.islice(iter(self.loader),self.limit)
 def __len__(self): return min(self.limit,len(self.loader))

c=cfg.clone(); c.merge_from_file('/root/COPE/configs/OCC_Duke/cope.yml'); c.DATASETS.ROOT_DIR='./data'; c.DATALOADER.NUM_WORKERS=8
c.OUTPUT_DIR='/root/autodl-tmp/cope/smoke-20261121/output20'; c.SOLVER.LOG_PERIOD=1
c.SOLVER.STAGE1.MAX_EPOCHS=1; c.SOLVER.STAGE1.CHECKPOINT_PERIOD=1; c.SOLVER.MAX_EPOCHS=1; c.SOLVER.EVAL_PERIOD=999; c.SOLVER.CHECKPOINT_PERIOD=999
os.makedirs(c.OUTPUT_DIR,exist_ok=True); setup_logger('COPE',c.OUTPUT_DIR,True); set_seed(c.SOLVER.SEED)
tr,val,cluster,nq,nc,cam,view=make_dataloader(c)
model=make_model(c,nc,camera_num=cam,view_num=view)
opt=make_optimizer(c,model)
sched=WarmupMultiStepLR(opt,c.SOLVER.STEPS,c.SOLVER.GAMMA,c.SOLVER.WARMUP_FACTOR,c.SOLVER.WARMUP_ITERS,c.SOLVER.WARMUP_METHOD,max_epoch=100)
pre=make_optimizer_1stage(c,model)
pre_sched=create_scheduler(pre,num_epochs=60,lr_min=c.SOLVER.STAGE1.LR_MIN,warmup_lr_init=c.SOLVER.STAGE1.WARMUP_LR_INIT,warmup_t=c.SOLVER.STAGE1.WARMUP_EPOCHS,noise_range=None)
def hook(name):
 last=[time.monotonic()]
 def f(*args,**kwargs):
  torch.cuda.synchronize(); now=time.monotonic(); print('SMOKE_STEP',name,'elapsed_s',round(now-last[0],3),'peak_allocated_mib',round(torch.cuda.max_memory_allocated()/1048576,1),'peak_reserved_mib',round(torch.cuda.max_memory_reserved()/1048576,1),flush=True); last[0]=now
 return f
pre.register_step_post_hook(hook('stage1')); opt.register_step_post_hook(hook('stage2'))
print('SMOKE_START: 20 batches per stage; full memory-bank pass; no evaluation; batch size 64',flush=True)
start=time.monotonic(); train(c,model,LimitedLoader(tr,20),val,cluster,opt,sched,nq,nc,pre,pre_sched)
print('SMOKE_SUCCESS total_seconds',round(time.monotonic()-start,2),flush=True)
