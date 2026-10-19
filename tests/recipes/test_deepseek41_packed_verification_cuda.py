"""Packed target rows preserve long per-request windows and accepted cache records."""
import os
from pathlib import Path
import pytest
import torch
from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
from ignition.components.parallel import TensorParallelShard
from ignition.components.paged_batch import Metadata
from ignition.components.ring_commit import PreparedRingCommit
from ignition.preparation.aot import aot_forward
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.attention import AttentionScratch, PreparedAttention, Rotary
from ignition.recipes.deepseek41.pool import ContinuationPool
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1'
    or not os.environ.get('FLARE_TEST_DS41_CHECKPOINT'), reason='requires an assigned GPU and a DS41 checkpoint')

@pytest.fixture(scope='module')
def weights():
    manifest = inspect_indexed_checkpoint(Path(os.environ['FLARE_TEST_DS41_CHECKPOINT']))
    device = torch.device('cuda:0')
    with DeepSeekWeightLoader(manifest, device, shard=TensorParallelShard(0,4), prefixes=('layers.0.attn.',)) as loader:
        prefix='layers.0.attn'
        loaded = dict(qkv_a=loader.load_linear_group((prefix+'.wq_a',prefix+'.wkv')),
            q_b=loader.load_linear(prefix+'.wq_b',rows=(0,8192)),
            output=loader.load_attention_output(prefix,(0,2048)),
            q_norm=loader.load_tensor(prefix+'.q_norm.weight'),
            kv_norm=loader.load_tensor(prefix+'.kv_norm.weight'),
            sink=loader.load_tensor(prefix+'.attn_sink',extent=slice(0,16)))
        config=loader.config
    return loaded,config

class Forward:
    def __init__(self, attention, commit, source, lengths):
        self.attention,self.commit,self.source,self.lengths=attention,commit,source,lengths
    def __call__(self):
        self.attention(self.source)
        self.commit(self.lengths)

@pytest.mark.parametrize('lengths',[(6,2),(6,5,4,3,2,2,1,1),(6,4,1,1,1,1,1,1)])
@torch.no_grad()
def test_packed_attention_and_ring_commit_cross_window_boundary(weights,lengths):
    torch._dynamo.reset()
    data,config=weights
    device=torch.device('cuda:0');batch=len(lengths);tokens=sum(lengths);width=max(lengths)
    pool=ContinuationPool.allocate(config,batch+1,batch*2+1,device)
    metadata=Metadata.allocate(tokens,batch,2,device,max_query_tokens=width)
    counts=torch.tensor(lengths,dtype=torch.int32,device=device)
    starts=torch.full((batch,),126,dtype=torch.int64,device=device)
    states=torch.arange(1,batch+1,dtype=torch.int64,device=device)
    pages=torch.arange(1,batch*2+1,dtype=torch.int32,device=device).view(batch,2)
    metadata.set(starts,counts,states,pages,torch.zeros((batch,width),dtype=torch.int64,device=device),
                 torch.tensor([batch],dtype=torch.int32,device=device))
    scratch=AttentionScratch.allocate(tokens,4,device)
    source=torch.randn((tokens,5120),dtype=torch.bfloat16,device=device)*.1
    commit=PreparedRingCommit(pool.reserved_backing,[pool.layout.region('layer0.window').offset],
        metadata.state_rows,metadata.starts,token_capacity=width,window_tokens=128,record_bytes=528)
    arena=ScratchArena(device,overlap=True)
    with RecordedPreparationSession(device=device,checkpoint='ds41-packed-window',
            workload_schema='packed-prefix-window',numerical_policy='checkpoint',compile_workers=16) as session:
        session.register_scratch(arena)
        attention=PreparedAttention(0,data,source,scratch,metadata,pool,Rotary(config,device),mode='verify',world=4,
            session=session,name=f'window.{batch}.{tokens}',workspace=arena,window_staging=commit.staging[0])
        session.tune(show_progress=False)
        forward=Forward(attention,commit,source,counts)
        compiled=aot_forward(forward)
        compiled(forward);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with session.capture(),torch.cuda.graph(graph):compiled(forward)
        try:
            for _ in range(3):
                pool.clear_for_preparation()
                before=torch.cuda.memory_stats()['allocation.all.allocated']
                graph.replay();torch.cuda.synchronize()
                assert torch.cuda.memory_stats()['allocation.all.allocated']==before
                assert torch.isfinite(scratch.attended).all()
                expected=pack_deepseek_v41_cache_reference(scratch.window_values,page_size=1,cache_kind='swa').reshape(tokens,528)
                offset=0
                for row,count in enumerate(lengths):
                    for column in range(count):
                        start=((126+column)%128)*528
                        assert torch.equal(pool.window(0)[row+1,start:start+528],expected[offset+column])
                    offset+=count
        finally:graph.reset()
