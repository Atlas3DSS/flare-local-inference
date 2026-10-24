#!/usr/bin/env python3
"""Create deterministic, newly authored public benchmark inputs (not historical traces)."""
import argparse
import json
from pathlib import Path

PROMPTS = (
    "Write a Python function that merges two sorted lists without modifying either input.\n```python\n",
    "Implement binary search in Rust and explain its invariant.\n```rust\n",
    "Write a TypeScript async retry helper with exponential backoff.\n```typescript\n",
    "Implement a stable topological sort in Python with cycle detection.\n```python\n",
    "Implement a bounded least-recently-used cache in Python and test eviction order.\n```python\n",
    "Write a Rust iterator that removes adjacent duplicate values and explain its ownership.\n```rust\n",
)


def content(index, blocks):
    context = ''.join(f'def example_{index}_{i}(values):\n    return sorted(set(values))\n\n' for i in range(blocks))
    return ('Review this generated example module as context. The final task follows.\n```python\n'
            + context + '```\n' + PROMPTS[index])


def make_rows(mode, blocks, *, tokenizer=None, target_tokens=None):
    rows=[]
    for index in range(6 if mode=='short' else 4):
        def row(n):
            return {'id':f'{"code" if mode=="short" else "long"}-{index:02d}',
                    'thinking':False,'messages':[{'role':'user','content':content(index,n)}],
                    'input_set':'public-generated-v1-new-baseline'}
        def length(n):
            ids=tokenizer.apply_chat_template(row(n)['messages'],tokenize=True,
                        add_generation_prompt=True,enable_thinking=False,return_dict=False)
            if not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids):
                raise ValueError('chat template must return a nonempty list of token IDs')
            return len(ids)
        count=blocks
        if target_tokens is not None:
            if tokenizer is None or length(0)>target_tokens:
                raise ValueError('target tokens require a tokenizer and room for the task')
            lo,hi=0,1
            while length(hi)<=target_tokens:
                if hi>target_tokens:
                    raise ValueError('tokenizer length did not grow with generated context')
                hi*=2
            while lo+1<hi:
                mid=(lo+hi)//2
                if length(mid)<=target_tokens:lo=mid
                else:hi=mid
            count=lo
        item=row(count);item['context_blocks']=count
        if tokenizer is not None:item['rendered_prompt_tokens']=length(count)
        rows.append(item)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('short','long'),default='short')
    parser.add_argument('--blocks',type=int,default=32)
    parser.add_argument('--target-tokens',type=int)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    if args.blocks<0 or (args.target_tokens is not None and (args.target_tokens<1 or not args.checkpoint)):
        parser.error('nonnegative blocks; target tokens require a positive count and --checkpoint')
    tokenizer=None
    if args.checkpoint:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.checkpoint,local_files_only=True)
    rows=make_rows(args.mode,args.blocks,tokenizer=tokenizer,target_tokens=args.target_tokens)
    with args.output.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({'input_set':'public-generated-v1-new-baseline','prompts':len(rows),
                      'rendered_prompt_tokens':[r.get('rendered_prompt_tokens') for r in rows]}))


if __name__=='__main__':main()
