#!/usr/bin/env python3
from __future__ import annotations
import os,sys,json,hashlib,subprocess,platform,socket,itertools,random,re
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path("/workspace/project/avelang")
OUT=ROOT/"test/examples/kda/reports/D_20260927_73_AITER_PREFILL_THREE_WAY_TRUE_BEST"
OUT.mkdir(parents=True,exist_ok=True); (OUT/"objects").mkdir(exist_ok=True)
os.environ["CHUNK_DELTA_ATTN_TRITON_AUTOTUNE"]="0"
os.environ["CHUNK_DELTA_ATTN_USE_FLASH_KDA"]="1"
os.environ["TRITON_CACHE_DIR"]=str(OUT/"triton_cache")
sys.path[:0]=[str(ROOT/"kda_baseline"),str(ROOT/"third_party/aiter"),
 str(ROOT/"test/examples/kda/reports/C_20260925_68_AITER_SHORT_TRUE_BEST_ENVELOPE")]
import numpy as np,torch,triton
from bench_common import make_kda_prefill_inputs
import run_sweep as c68
import aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda as flash
import aiter.ops.triton._triton_kernels.chunk_delta_attn.chunk_fwd as chunk_fwd
from aiter.ops.triton.kimi_delta_attn.chunk_delta_attn import chunk_kimi_delta_attn

TS=(32,64,127,128,256,384,511,512,1024,2048,4096)
K1S=[{"W":w,"S":s} for w in (2,4) for s in (1,2)]
K2S=[{"BW":b,"W":w,"S":s,"num_ctas":1} for b in (16,32,64) for w in (2,4) for s in (1,2)]
SW=10; SR=10; SROUND=3; CW=50; CR=100; CROUND=3; PW=50; PR=100; PROUND=3; TOL=.03
COMMIT="7bb44998274fa679ece0a37bf8426ab780b1837d"
K1=flash._flash_kda_prepare_kernel.fn.fn
K2AT=flash._flash_kda_segment_kernel.fn
K2=K2AT.fn
ABI={
 "K1_outputs":[["ws_kd","bf16","[H*NT,C,K]"],["ws_qd","bf16","[H*NT,C,K]"],
 ["ws_kr","bf16","[H*NT,C,K]"],["ws_gt","fp32","[H*NT,K]"],
 ["ws_inv_mqk","fp16","[H*NT,2*C,C]"]],
 "K2_reads":["ws_kd","ws_qd","ws_kr","ws_gt","ws_inv_mqk"],
 "workload":{"B":1,"H":12,"K":128,"V":128,"C":32},"layout":"contiguous"}
ABI_ID=hashlib.sha256(json.dumps(ABI,sort_keys=True).encode()).hexdigest()

def save(n,x):
 p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+"\n")
def key1(c):return f"W{c['W']}_S{c['S']}"
def key2(c):return f"BW{c['BW']}_W{c['W']}_S{c['S']}"
def cfgstats(xs):
 a=np.asarray(xs,dtype=np.float64);m=float(np.median(a))
 return {"median_us":m,"mean_us":float(np.mean(a)),"p25_us":float(np.percentile(a,25)),
 "p75_us":float(np.percentile(a,75)),"p95_us":float(np.percentile(a,95)),
 "min_us":float(np.min(a)),"max_us":float(np.max(a)),
 "mad_us":float(np.median(np.abs(a-m))),"stddev_us":float(np.std(a)),"n":int(len(a))}
def mm(a,b):
 d=(a.float()-b.float()).abs();r=d/b.float().abs().clamp_min(1.)
 return {"max_abs":float(d.max().item()),"max_rel":float(r.max().item()),
 "mean_abs":float(d.mean().item()),"finite":bool(torch.isfinite(a).all().item())}
def cmp(got,ref):
 a=mm(got[0],ref[0]);b=mm(got[1],ref[1])
 return {"output":a,"final_state":b,"pass":a["finite"] and b["finite"] and a["max_abs"]<=TOL and b["max_abs"]<=TOL,
 "tolerance":{"max_abs":TOL,"source":"existing D71 controlled K2 gate"}}
def met(k):return getattr(k,"metadata",None)
def save_obj(kind,T,cfg,k,grid,extra=None):
 asm=getattr(k,"asm",{}) or {};blob=asm.get("hsaco")
 if not blob:raise RuntimeError(f"missing HSACO {kind} T{T} {cfg}")
 sha=hashlib.sha256(bytes(blob)).hexdigest();m=met(k)
 nw=int(getattr(m,"num_warps",cfg.get("W",0)));ns=int(getattr(m,"num_stages",cfg.get("S",0)))
 shared=int(getattr(m,"shared",0));th=str(getattr(k,"hash","") or getattr(m,"hash",""))
 p=OUT/"objects"/(sha+".hsaco")
 if not p.exists():p.write_bytes(bytes(blob))
 isa=asm.get("amdgcn")
 if isa and not p.with_suffix(".isa").exists():p.with_suffix(".isa").write_text(str(isa))
 row={"kernel_symbol":kind,"T":int(T),"config":dict(cfg),"grid":[int(x) for x in grid],
 "block":nw*64,"num_warps":nw,"num_stages":ns,"num_ctas":int(cfg.get("num_ctas",1)),
 "dynamic_shared_bytes":shared,"hsaco_path":str(p),"hsaco_sha256":sha,
 "triton_metadata_hash":th,"workspace_abi_id":ABI_ID}
 if extra:row.update(extra)
 meta=p.with_suffix(".json")
 if not meta.exists():meta.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
 return row
def run_k1(T,inp,ws,c):
 nt=triton.cdiv(T,32)
 return K1[(nt,12)](q=inp["q"],k=inp["k"],g_raw=inp["raw_g"],
 beta_raw=inp["raw_beta"].float().contiguous(),A_log=inp["A_log"],dt_bias=inp["dt_bias"].view(-1),
 ws_kd=ws["kd"],ws_qd=ws["qd"],ws_kr=ws["kr"],ws_gt=ws["gt"],ws_inv_mqk=ws["inv_mqk"],
 cu_seqlens=None,chunk_indices=None,scale=128**-.5,lower_bound=-5.,
 T=int(T),NT=int(nt),TOTAL_TILES=int(nt),H=12,K=128,C=32,BC=16,
 NUM_DOUBLING=3,NUM_MERGE=1,IS_VARLEN=False,HAS_BIAS=True,
 num_warps=int(c["W"]),num_stages=int(c["S"]),num_ctas=1)
def public(inp,state):
 return chunk_kimi_delta_attn(inp["q"],inp["k"],inp["v"],inp["raw_g"],inp["raw_beta"],
 A_log=inp["A_log"],dt_bias=inp["dt_bias"].view(-1),initial_state=state,output_final_state=True,
 use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,use_beta_sigmoid_in_kernel=True,
 safe_gate=True,lower_bound=-5.)
def capture_public(T,inp,state):
 rows=[];old1,old2=K1.run,K2.run
 scan=getattr(flash,"_flash_kda_seg_scan_kernel",None)
 if hasattr(scan,"fn"):scan=scan.fn
 if hasattr(scan,"fn"):scan=scan.fn
 oldscan=scan.run if scan is not None and hasattr(scan,"run") else None
 def hook(kind,orig):
  def wrapped(*args,**kw):
   k=orig(*args,**kw);m=met(k)
   conf={"W":int(getattr(m,"num_warps",0)),"S":int(getattr(m,"num_stages",0)),
         "num_ctas":int(kw.get("num_ctas",1))}
   if "BW" in kw:conf["BW"]=int(kw["BW"])
   if "W" in kw and kind=="_flash_kda_segment_kernel":conf["kernel_output_width"]=int(kw["W"])
   extra={}
   for q in ("HAS_V","COMPUTE_OUTPUT","STORE_H_OUT","STORE_FINAL","INIT_IDENTITY","NUM_SEGS_CLASS"):
    if q in kw:extra[q.lower()]=kw[q]
   g=kw.get("grid")
   if callable(g):
    try:g=g(m)
    except Exception:g=repr(g)
   if isinstance(g,int):g=[g,1,1]
   elif isinstance(g,tuple):g=list(g)+[1]*(3-len(g))
   elif isinstance(g,list):g=g+[1]*(3-len(g))
   else:g=repr(g)
   row=save_obj(kind,T,conf,k,g if isinstance(g,list) else [0,0,0],extra)
   row["captured_grid_raw"]=g
   row["launch_kwargs"]={q:(v if isinstance(v,(str,int,float,bool)) or v is None else repr(v))
    for q,v in kw.items() if q in ("BW","W","HAS_V","COMPUTE_OUTPUT","STORE_H_OUT",
    "STORE_FINAL","INIT_IDENTITY","NUM_SEGS_CLASS","num_warps","num_stages","num_ctas")}
   rows.append(row);return k
  return wrapped
 try:
  K1.run=hook("_flash_kda_prepare_kernel",old1);K2.run=hook("_flash_kda_segment_kernel",old2)
  if oldscan:scan.run=hook("_flash_kda_seg_scan_kernel",oldscan)
  got=public(inp,state);torch.cuda.synchronize()
 finally:
  K1.run=old1;K2.run=old2
  if oldscan:scan.run=oldscan
 return got,rows
def event_chain(T,inp,ws,case,a,b,start,stop):
 case["reset"]();start.record();run_k1(T,inp,ws,a);case["invoke"](b,warmup=False)
 stop.record();stop.synchronize();return float(start.elapsed_time(stop)*1000)
def main():
 actual=subprocess.check_output(["git","-C",str(ROOT/"third_party/aiter"),"rev-parse","HEAD"],text=True).strip()
 if actual!=COMMIT:raise RuntimeError("STOP: AITER commit mismatch "+actual)
 torch.cuda.set_device(0);gpu=torch.cuda.get_device_name(0);props=torch.cuda.get_device_properties(0)
 if "MI300X" not in gpu.upper():raise RuntimeError("STOP: GPU "+gpu)
 if (c68.BATCH,c68.HEADS,c68.K,c68.V,c68.C)!=(1,12,128,128,32):raise RuntimeError("STOP: shape mismatch")
 src2=[{"BW":int(c.kwargs["BW"]),"W":int(c.num_warps),"S":int(c.num_stages),"num_ctas":int(c.num_ctas)} for c in flash._K2_CONFIGS]
 if {key2(c) for c in src2}!={key2(c) for c in K2S}:raise RuntimeError("STOP: K2 candidate source mismatch")
 pub1=[{"W":int(c.num_warps),"S":int(c.num_stages)} for c in flash._flash_kda_prepare_kernel.fn.configs]
 pub2=[{"BW":int(c.kwargs.get("BW",-1)),"W":int(c.num_warps),"S":int(c.num_stages),"num_ctas":int(c.num_ctas)} for c in flash._flash_kda_segment_kernel.fn.configs]
 env={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"host":socket.gethostname(),"container":"ljd_sglang_kda",
 "repo_head":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
 "aiter_commit":actual,"gpu":gpu,"gfx":"gfx942","gpu_properties":str(props),"torch":torch.__version__,
 "hip":getattr(torch.version,"hip",None),"python":platform.python_version(),
 "default_stream_handle":int(torch.cuda.current_stream().cuda_stream),"T":list(TS),
 "shape":{"B":1,"H":12,"K":128,"V":128,"C":32},"K1_formal_candidates":K1S,"K2_legal_candidates":src2,
 "AITER_PUBLIC_default_K1_configs":pub1,"AITER_PUBLIC_default_K2_configs":pub2,
 "environment":{x:os.environ.get(x) for x in ("CHUNK_DELTA_ATTN_TRITON_AUTOTUNE","CHUNK_DELTA_ATTN_USE_FLASH_KDA","TRITON_CACHE_DIR")},
 "workspace_ABI":ABI,"workspace_ABI_SHA256":ABI_ID,
 "protocol":{"screen":{"warmups":SW,"repeats_per_round":SR,"rounds":SROUND},
 "winner_confirmation":{"warmups_per_round":CW,"repeats_per_round":CR,"rounds":CROUND},
 "three_paths":{"warmups_per_round":PW,"repeats_per_round":PR,"rounds":PROUND},
 "event":"state restore/reset outside HIP event; same default stream; K1 followed by K2 with no synchronization between"},
 "correctness":{"max_abs":TOL,"reference":"same-input AITER_PUBLIC_PREFILL; public additionally compared against AITER default pipeline"}}
 save("environment_manifest.json",env)
 print("D73 PHASE0 IDENTITY PASS",flush=True)
 rep=ROOT/"test/examples/kda/reports"
 c69=json.loads((rep/"C_20260925_69_k1_true_best_dispatch_audit/k1_true_best_results_final.json").read_text())
 k1map={int(x["T"]):x["true_best"] for x in c69["public_vs_true_best"]}
 d71=rep/"D_20260927_71_K2_TRUE_BEST_TIMING_RECALIBRATION"
 km=json.loads((d71/"candidate_manifest.json").read_text());best=json.loads((d71/"confirmed_true_best_map.json").read_text())
 d72=json.loads((rep/"D_20260927_72_K2_T32_TRUE_BEST_RESOURCE_GOLDEN/D_20260927_72_K2_T32_TRUE_BEST_RESOURCE_GOLDEN.json").read_text())
 stand={}
 for T in TS:
  x=k1map[T];m=re.search(r"w([24])_s([12])",str(x["config"]).lower())
  if not m:raise RuntimeError(f"STOP: K1 map ambiguous T={T}: {x}")
  if T==32:
   y=d72["identity"];k2cfg={"BW":16,"W":2,"S":2,"num_ctas":1};k2sha=y["hsaco_sha256"];k2hash=y.get("triton_hash") or y.get("triton_metadata_hash");g2=[8,12,1];b2=128
  else:
   y=best[str(T)];kk=y["canonical_winner_config"];y=km[str(T)][kk]
   k2cfg=y["config"];k2sha=y["sha256"];k2hash=y.get("cache_hash");g2=y["grid"];b2=y["block"]
  stand[T]={"K1":{"W":int(m.group(1)),"S":int(m.group(2)),"hsaco_sha256":x["hsaco_sha256"],
    "triton_metadata_hash":x["triton_hash"],"grid":x["grid"],"block":x["block"],"source":"C69/C71"},
    "K2":{"config":k2cfg,"hsaco_sha256":k2sha,"triton_metadata_hash":k2hash,"grid":g2,"block":b2,
    "source":"D72" if T==32 else "D71"}}
 save("standalone_winner_source_maps.json",stand)

 results={};cmanifest={};correct={};publicmap={};raw={};paths={}
 for T in TS:
  print(f"D73 START T={T}",flush=True)
  torch.manual_seed(91000+T)
  inp=make_kda_prefill_inputs(T=T,H=12,seed=6800+T)
  state=torch.randn((1,12,128,128),device="cuda",dtype=torch.float32).mul_(.1).contiguous()
  nt=triton.cdiv(T,32);ws=c68.allocate_workspace(nt,inp["q"].device);case=c68.make_case(inp,ws,state)
  flash.CHUNK_DELTA_ATTN_USE_FLASH_KDA=True;chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA=True
  pubfirst=public(inp,state);torch.cuda.synchronize()
  pg,trace=capture_public(T,inp,state)
  po=(pg[0].clone(),pg[1].clone() if pg[1] is not None else None)
  if po[1] is None:raise RuntimeError(f"STOP public missing final state T={T}")
  chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA=False
  default=public(inp,state);torch.cuda.synchronize()
  oracle=(default[0].clone(),default[1].clone())
  chunk_fwd.CHUNK_DELTA_ATTN_USE_FLASH_KDA=True
  pc=cmp(po,oracle)
  publicmap[str(T)]={"T":T,"entrypoint":"chunk_kimi_delta_attn","candidate_forcing":"none",
   "correctness_vs_default_AITER_pipeline":pc,"launches":trace,
   "note":"trace captured after uninstrumented warm/tune call"}
  save("aiter_public_provenance.json",publicmap)
  if not pc["pass"]:
   results[str(T)]={"status":"STOP_PUBLIC_CORRECTNESS","public_correctness":pc};save("joint_results.json",results);continue

  k1obj={};k2obj={}
  for a in K1S:
   case["reset"]();k=run_k1(T,inp,ws,a);torch.cuda.synchronize()
   k1obj[key1(a)]=save_obj("_flash_kda_prepare_kernel",T,a,k,(nt,12,1))
  for b in K2S:
   case["reset"]();st=stand[T]["K1"];run_k1(T,inp,ws,{"W":st["W"],"S":st["S"]})
   k=case["invoke"](b,warmup=False);torch.cuda.synchronize()
   k2obj[key2(b)]=save_obj("_flash_kda_segment_kernel",T,b,k,(triton.cdiv(128,b["BW"]),12,1),
     {"has_v":True,"compute_output":True,"num_segments":1})
  e1=stand[T]["K1"];e2=stand[T]["K2"];a1=k1obj[key1(e1)]
  a2=k2obj[key2(e2["config"])]
  check={"K1":{"expected_sha256":e1["hsaco_sha256"],"actual_sha256":a1["hsaco_sha256"],
   "match":e1["hsaco_sha256"]==a1["hsaco_sha256"]},
   "K2":{"expected_sha256":e2["hsaco_sha256"],"actual_sha256":a2["hsaco_sha256"],
   "match":e2["hsaco_sha256"]==a2["hsaco_sha256"],"grid_expected":e2["grid"],"grid_actual":a2["grid"],
   "block_expected":e2["block"],"block_actual":a2["block"],
   "grid_match":e2["grid"]==a2["grid"],"block_match":e2["block"]==a2["block"]}}
  if not(check["K1"]["match"] and check["K2"]["match"] and check["K2"]["grid_match"] and check["K2"]["block_match"]):
   results[str(T)]={"status":"STOP_STANDALONE_IDENTITY_MISMATCH","identity_checks":check}
   cmanifest[str(T)]={"K1":k1obj,"K2":k2obj};save("candidate_manifest.json",cmanifest);save("joint_results.json",results)
   print("STOP identity",T,check,flush=True);continue
  pairs=[];rows=[]
  for a in K1S:
   for b in K2S:
    label=key1(a)+"__"+key2(b)
    try:
     case["reset"]();ka=run_k1(T,inp,ws,a);kb=case["invoke"](b,warmup=False);torch.cuda.synchronize()
     gate=cmp((case["out"].clone(),case["final_state"].clone()),po)
     row={"pair":label,"K1":k1obj[key1(a)],"K2":k2obj[key2(b)],"correctness":gate,"valid":gate["pass"]}
     if gate["pass"]:pairs.append((a,b,label))
    except Exception as ex:
     row={"pair":label,"K1":k1obj.get(key1(a)),"K2":k2obj.get(key2(b)),"valid":False,"invalid_reason":repr(ex)}
    rows.append(row)
  correct[str(T)]={"T":T,"reference":"AITER_PUBLIC_PREFILL","workspace_abi_id":ABI_ID,"candidate_pairs":rows}
  cmanifest[str(T)]={"K1":k1obj,"K2":k2obj}
  save("candidate_manifest.json",cmanifest);save("correctness.json",correct)
  print(f"D73 CORRECTNESS T={T} {len(pairs)}/48 PASS",flush=True)
  if not pairs:
   results[str(T)]={"status":"STOP_NO_VALID_JOINT_PAIR","identity_checks":check};save("joint_results.json",results);continue
  start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
  for a,b,_ in pairs:
   for _ in range(SW):case["reset"]();run_k1(T,inp,ws,a);case["invoke"](b,warmup=False)
  torch.cuda.synchronize()
  scr={p[2]:[] for p in pairs}
  for rnd in range(SROUND):
   order=list(pairs);random.Random(7300+T+rnd).shuffle(order)
   for _ in range(SR):
    for a,b,l in order:scr[l].append(event_chain(T,inp,ws,case,a,b,start,stop))
  ss={l:cfgstats(v) for l,v in scr.items()};ordered=sorted(ss,key=lambda l:ss[l]["median_us"])
  close=[l for l in ordered if ss[l]["median_us"]<=ss[ordered[0]]["median_us"]*1.02]
  for l in ordered[:2]:
   if l not in close:close.append(l)
  confirmed={l:[] for l in close};rounds={l:[] for l in close}
  for rnd in range(CROUND):
   order=[next(p for p in pairs if p[2]==l) for l in close]
   if rnd%2:order.reverse()
   elif len(order)>1 and rnd:order=order[rnd:]+order[:rnd]
   for a,b,l in order:
    for _ in range(CW):case["reset"]();run_k1(T,inp,ws,a);case["invoke"](b,warmup=False)
    torch.cuda.synchronize();vals=[]
    for _ in range(CR):vals.append(event_chain(T,inp,ws,case,a,b,start,stop))
    confirmed[l].extend(vals);rounds[l].append(cfgstats(vals))
  cs={l:cfgstats(v) for l,v in confirmed.items()};rank=sorted(cs,key=lambda l:cs[l]["median_us"])
  win=rank[0];wp=next(p for p in pairs if p[2]==win);run=rank[1] if len(rank)>1 else None
  f1={"W":e1["W"],"S":e1["S"]};f2=e2["config"];fl=key1(f1)+"__"+key2(f2)
  if fl not in confirmed:
   a=f1;b=f2;vals=[]
   for rnd in range(CROUND):
    for _ in range(CW):case["reset"]();run_k1(T,inp,ws,a);case["invoke"](b,warmup=False)
    torch.cuda.synchronize();one=[]
    for _ in range(CR):one.append(event_chain(T,inp,ws,case,a,b,start,stop))
    vals.extend(one);rounds.setdefault(fl,[]).append(cfgstats(one))
   confirmed[fl]=vals;cs[fl]=cfgstats(vals)
  same=fl==win
  joint={"T":T,"status":"PASS","screen":ss,"screen_order":ordered,"confirmation_candidates":close,
   "confirmation":{l:{"pooled":cs[l],"per_round":rounds[l]} for l in confirmed},
   "winner":win,"winner_pair":{"K1":wp[0],"K2":wp[1]},
   "winner_latency_us":cs[win]["median_us"],"runner_up":run,
   "runner_up_latency_us":cs[run]["median_us"] if run else None,
   "runner_up_gap_pct":(cs[run]["median_us"]/cs[win]["median_us"]-1)*100 if run else None,
   "valid_pair_count":len(pairs),"identity_checks":check}
  fact={"label":fl,"K1":e1,"K2":e2,"screen":ss.get(fl),"confirmation":cs[fl],
   "vs_joint":{"same_pair":same,"factorized_e2e_us":cs[fl]["median_us"],
   "joint_e2e_us":cs[win]["median_us"],"delta_us":cs[fl]["median_us"]-cs[win]["median_us"],
   "delta_pct":(cs[fl]["median_us"]/cs[win]["median_us"]-1)*100,
   "K1_changed":wp[0]!=f1,"K2_changed":wp[1]!=f2}}
  results[str(T)]={"status":"SWEEP_PASS","identity_checks":check,"joint":joint,"factorized":fact}
  raw[str(T)]={"screen_samples_us":scr,"confirmation_samples_us":confirmed,"confirmation_rounds":rounds}
  save("joint_sweep_raw.json",raw);save("joint_results.json",results)
  print(f"D73 JOINT T={T} winner={win} median={cs[win]['median_us']:.4f}us fact={fl}",flush=True)

  # Three principal paths, each round gets 50 warmups and 100 interleaved timed repeats.
  fcfg=(f1,f2);jcfg=(wp[0],wp[1])
  path_names=["AITER_PUBLIC_PREFILL","AITER_FACTORIZED_BEST_PREFILL","AITER_TRUE_BEST_PREFILL"]
  ps={n:[] for n in path_names};pr={n:[] for n in path_names}
  state_work=case["alive"][3]
  for rnd in range(PROUND):
   order=path_names[:] if rnd%2==0 else path_names[::-1]
   for n in order:
    vals=[]
    if n=="AITER_PUBLIC_PREFILL":
     for _ in range(PW):state_work.copy_(state);public(inp,state_work)
    else:
     aa,bb=fcfg if n=="AITER_FACTORIZED_BEST_PREFILL" else jcfg
     for _ in range(PW):case["reset"]();run_k1(T,inp,ws,aa);case["invoke"](bb,warmup=False)
    torch.cuda.synchronize()
    for _ in range(PR):
     if n=="AITER_PUBLIC_PREFILL":
      state_work.copy_(state);start.record();public(inp,state_work);stop.record();stop.synchronize()
      us=float(start.elapsed_time(stop)*1000)
     else:
      aa,bb=fcfg if n=="AITER_FACTORIZED_BEST_PREFILL" else jcfg
      us=event_chain(T,inp,ws,case,aa,bb,start,stop)
     vals.append(us);ps[n].append(us)
    pr[n].append(cfgstats(vals))
  # Single-kernel contextual timings for path K1s and direct one-segment K2s.
  iso={}
  k1select={"PUBLIC":next((x["config"] for x in trace if x["kernel_symbol"]=="_flash_kda_prepare_kernel"),None),
   "FACTORIZED":f1,"TRUE_BEST":wp[0]}
  k1select={n:c for n,c in k1select.items() if c and c.get("W") in (2,4) and c.get("S") in (1,2)}
  for n,c in k1select.items():
   vals=[]
   for _ in range(50):run_k1(T,inp,ws,c)
   torch.cuda.synchronize()
   for rnd in range(3):
    for _ in range(100):
     case["reset"]();start.record();run_k1(T,inp,ws,c);stop.record();stop.synchronize()
     vals.append(float(start.elapsed_time(stop)*1000))
   iso["K1_"+n]={"config":c,"stats":cfgstats(vals)}
  for n,c,a in [("FACTORIZED",f2,f1),("TRUE_BEST",wp[1],wp[0])]:
   vals=[]
   for _ in range(50):case["reset"]();run_k1(T,inp,ws,a);case["invoke"](c,warmup=False)
   torch.cuda.synchronize()
   for rnd in range(3):
    for _ in range(100):
     case["reset"]();run_k1(T,inp,ws,a);start.record();case["invoke"](c,warmup=False);stop.record();stop.synchronize()
     vals.append(float(start.elapsed_time(stop)*1000))
   iso["K2_"+n]={"config":c,"stats":cfgstats(vals),"context":"one-segment controlled K2"}
  pt={n:cfgstats(v) for n,v in ps.items()}
  interactions={}
  for n,chain,k1n,k2n in [("FACTORIZED",pt["AITER_FACTORIZED_BEST_PREFILL"],"K1_FACTORIZED","K2_FACTORIZED"),
                           ("JOINT",pt["AITER_TRUE_BEST_PREFILL"],"K1_TRUE_BEST","K2_TRUE_BEST")]:
   if k1n in iso and k2n in iso:
    interactions[n]={"e2e_us":chain["median_us"],"k1_us":iso[k1n]["stats"]["median_us"],
     "k2_us":iso[k2n]["stats"]["median_us"],
     "interaction_delta_us":chain["median_us"]-iso[k1n]["stats"]["median_us"]-iso[k2n]["stats"]["median_us"]}
  paths[str(T)]={"round_medians":pr,"pooled":pt,"K1_K2_standalone":iso,
   "interaction_delta":interactions,"public_launches":trace,
   "factorized_pair":fl,"joint_pair":win,"same_pair":same,
   "public_correctness_vs_default":pc}
  results[str(T)]["three_way"]={"public_us":pt["AITER_PUBLIC_PREFILL"]["median_us"],
   "factorized_us":pt["AITER_FACTORIZED_BEST_PREFILL"]["median_us"],
   "joint_true_best_us":pt["AITER_TRUE_BEST_PREFILL"]["median_us"],
   "public_to_factorized_speedup":pt["AITER_PUBLIC_PREFILL"]["median_us"]/pt["AITER_FACTORIZED_BEST_PREFILL"]["median_us"],
   "public_to_joint_speedup":pt["AITER_PUBLIC_PREFILL"]["median_us"]/pt["AITER_TRUE_BEST_PREFILL"]["median_us"],
   "factorized_to_joint_speedup":pt["AITER_FACTORIZED_BEST_PREFILL"]["median_us"]/pt["AITER_TRUE_BEST_PREFILL"]["median_us"]}
  results[str(T)]["interaction_delta"]=interactions
  save("three_path_raw.json",paths);save("joint_results.json",results)
  print(f"D73 PATHS T={T} pub={pt[path_names[0]]['median_us']:.4f} fact={pt[path_names[1]]['median_us']:.4f} joint={pt[path_names[2]]['median_us']:.4f}us same={same}",flush=True)
 print("D73 PHASES 0-5 DONE",flush=True)

if __name__=="__main__":main()
