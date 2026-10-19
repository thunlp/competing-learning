"""Alpha-ranked streaming masks with independent cached greedy questions."""
import copy, json, time
import torch
import torch.nn.functional as F
import modeling.kda.attention as attention_module
SINK=128
RECENT=256
LAYERS=(0,4,8,12,16,20)
HEADS=12
def count_for_percent(p):return (p*HEADS+50)//100

def spans(labels):
 indices=(labels!=-100).nonzero().flatten().tolist();out=[]
 for i in indices:
  if not out or i!=out[-1][1]:out.append([i,i+1])
  else:out[-1][1]+=1
 return out

def streaming(q,k,v,sink=SINK,recent=RECENT):
 """Exact causal sink-union-recent mask, including cached query offsets."""
 b,t,h,d=q.shape;n=k.shape[1];offset=n-t;assert offset>=0
 parts=[]
 keys=torch.arange(n,device=q.device)
 for start in range(0,t,256):
  stop=min(t,start+256);qp=offset+torch.arange(start,stop,device=q.device)
  mask=(keys[None,:]<=qp[:,None])&((keys[None,:]<sink)|(keys[None,:]>qp[:,None]-recent))
  parts.append(F.scaled_dot_product_attention(q[:,start:stop].transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask,dropout_p=0.0,enable_gqa=True).transpose(1,2))
 return torch.cat(parts,dim=1)

class MaskController:
 def __init__(self,model):
  self.layer=None;self.selected=set();self.native=attention_module.flash_attn_func;self.handles=[]
  for layer in LAYERS:
   module=model.model.layers[layer].attn
   def before(mod,args,kwargs,layer=layer):self.layer=layer
   self.handles.append(module.register_forward_pre_hook(before,with_kwargs=True))
  attention_module.flash_attn_func=self.call
 def call(self,q,k,v,**kwargs):
  selected=[i for i in range(k.shape[2]) if (self.layer,i) in self.selected]
  if not selected:return self.native(q,k,v,**kwargs)
  groups=q.shape[2]//k.shape[2];qh=[h for kh in selected for h in range(kh*groups,(kh+1)*groups)]
  local=streaming(q[:,:,qh,:],k[:,:,selected,:],v[:,:,selected,:])
  if len(selected)==k.shape[2]:return local
  out=self.native(q,k,v,**kwargs).clone();out[:,:,qh,:]=local;return out
 def close(self):
  attention_module.flash_attn_func=self.native
  for h in self.handles:h.remove()

def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def evaluate(model,controller,dataset,tokenizer,ranking,n,out):
 controller.selected={(r['layer'],r['head']) for r in ranking[:n]}
 records=[];started=time.time()
 for i in range(len(dataset)):
  sample=dataset[i];ids=sample['input_ids'];ss=spans(sample['labels']);assert len(ss)==10
  # One common document containing all needles; each question starts from its cache.
  doc_end=ss[0][0]-len(dataset.question_prefixes[0]);assert doc_end>0
  with torch.inference_mode():
   doc=model(input_ids=ids[:doc_end].unsqueeze(0).cuda(),use_cache=True,logits_to_keep=1)
   base_cache=doc.past_key_values
   for j,(start,end) in enumerate(ss):
    target=ids[start:end].tolist();cache=copy.deepcopy(base_cache)
    question=torch.tensor(dataset.question_prefixes[j],device='cuda').unsqueeze(0)
    pred=model(input_ids=question,past_key_values=cache,use_cache=True,logits_to_keep=1)
    matched=0;generated=[]
    # Exact-match early rejection is equivalent to generating the full key:
    # a greedy mismatch cannot be repaired by subsequent generated tokens.
    for pos,expected in enumerate(target):
     token=int(pred.logits[0,-1].argmax());generated.append(token)
     if token!=expected:break
     matched+=1
     if pos+1<len(target):
      pred=model(input_ids=torch.tensor([[token]],device='cuda'),past_key_values=pred.past_key_values,use_cache=True,logits_to_keep=1)
    records.append(dict(example=i,passkey=j+1,exact=matched==len(target),matched_prefix_tokens=matched,target_tokens=len(target),target=tokenizer.decode(target),generated_prefix=tokenizer.decode(generated)))
    del cache,pred
   del doc,base_cache
  print(f'heads={n}/{HEADS} sample={i+1}/{len(dataset)} exact={sum(r["exact"] for r in records)}/{len(records)} elapsed={time.time()-started:.1f}s',flush=True)
  atomic(out,dict(n_streaming_heads=n,actual_percent=n/HEADS*100,n_examples=i+1,n_keys=len(records),correct=sum(r['exact'] for r in records),accuracy=sum(r['exact'] for r in records)/len(records),first_token_accuracy=sum(r['matched_prefix_tokens']>0 for r in records)/len(records),elapsed_seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,selected=ranking[:n],records=records))
 return json.loads(out.read_text())
