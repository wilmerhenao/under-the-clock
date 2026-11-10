"""One model through vLLM on 4 x L4: how fast it runs and how well it predicts the agent's own tokens.

Runs in its own process (one per model), so every model starts from clean GPUs.
usage: python serve_bench.py MODEL_DIR OUT_JSON DATA_JSON [ADAPTER_DIR]
ADAPTER_DIR: a PEFT LoRA adapter for this model (already cut to its shapes): every request runs with it, so speed and
quality include vLLM's LoRA path.
DATA_JSON: {'heldout': [{'ids': [...], 'own': [positions written by the agent]}], 'speed': {'p2k': [ids], ...},
            'streams': [[ids], ...], 'sanity': [chat prompt ids], optional 'turns': [prompt ids ending where the agent's
            turn starts]: the model's own next turn, greedy (R['turns']: texts), to compare its actions with the full
            model's}  (greedy, 512 tokens)
"""
import json
import os
import sys
import time


def main():
    # everything runs inside main(): vLLM starts its GPU workers with "spawn", which imports this file again in each
    # worker; code outside the `if __name__ == '__main__'` guard below would run (and fail) there
    model_dir, out_path, data_path = sys.argv[1:4]
    adapter = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
    data = json.load(open(data_path))
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    R = {'model': model_dir, 'adapter': adapter}
    lora_kw = {}
    if adapter:
        from vllm.lora.request import LoRARequest
        rank = json.load(open(f'{adapter}/adapter_config.json'))['r']
        lora_kw = {'lora_request': LoRARequest('heal', 1, adapter)}
        R['lora_rank'] = rank

    def save():
        json.dump(R, open(out_path, 'w'), indent=1)

    t0 = time.time()
    # the same engine for every model; chunks of 1,024 prompt tokens and 20% of each GPU left outside vLLM's own budget
    # keep room for the prompt log-probabilities (1,024 tokens x 262k vocabulary at a time)
    # max_model_len 20,480 (serve-v2b): every prompt here fits with its output (<= 16,256 tokens); with CUDA graph memory
    # counted in vLLM's budget (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1, set by prune_serve.py) a LoRA engine, which
    # captures every graph twice, no longer has KV room for 32,768
    # PRUNE_MAX_BATCHED / PRUNE_GPU_UTIL (serve-v5's full_b): the same weights in an engine with other prefill chunks and
    # another KV-cache size, i.e. other batching and floating-point order: the floor of greedy agreement from numerics
    eng = {'max_num_batched_tokens': int(os.environ.get('PRUNE_MAX_BATCHED') or 1024),
           'gpu_memory_utilization': float(os.environ.get('PRUNE_GPU_UTIL') or 0.80)}
    if os.environ.get('PRUNE_SPEC'):   # serve-v7 (B202 Run 2'): speculative decoding, e.g. n-gram lookup in the prompt
        eng['speculative_config'] = json.loads(os.environ['PRUNE_SPEC'])
    # a JSON copy: vLLM writes objects into the dicts it is given (serve-v7 v1: speculative_config gained a ModelConfig
    # and the first save failed after the engine had started)
    R['engine'] = json.loads(json.dumps(eng))
    try:
        llm = LLM(model=model_dir, tensor_parallel_size=4, max_model_len=20480, dtype='auto',
                  enable_prefix_caching=False, seed=0, **eng,
                  **({'enable_lora': True, 'max_lora_rank': R['lora_rank'], 'max_loras': 1} if adapter else {}))
    except Exception as e:
        # an engine that fails to start: record why and leave at once (serve-v2: a failed LoRA engine kept its process
        # alive until the 20-minute kill); the parent then kills the workers left in this process group
        R['error'] = f'{type(e).__name__}: {str(e)[:500]}'
        save()
        print('ENGINE FAILED TO START', R['error'], flush=True)
        os._exit(3)
    R['load_s'] = round(time.time() - t0, 1)
    save()

    def run(prompts, max_tokens, **kw):
        sp = SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True, **kw)
        t = time.time()
        out = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False, **lora_kw)
        return out, time.time() - t

    run([data['speed']['p2k'][:256]], 8)          # warm-up
    # speed: prompt reading (prefill) and writing (decode), one request at a time and 8 at once
    sp_res = {}
    for name, ids in data['speed'].items():
        _, t_pre = run([ids], 1)
        _, t_all = run([ids], 256)
        sp_res[name] = {'prompt_tokens': len(ids), 'prefill_s': round(t_pre, 3),
                        'prefill_tok_s': round(len(ids) / t_pre, 1),
                        'decode_tok_s': round(255 / max(1e-6, t_all - t_pre), 2)}
    streams = data['streams']
    _, t_pre8 = run(streams, 1)
    _, t_all8 = run(streams, 256)
    sp_res['streams8'] = {'n': len(streams), 'prompt_tokens_each': len(streams[0]),
                          'decode_tok_s_total': round(len(streams) * 255 / max(1e-6, t_all8 - t_pre8), 2)}
    R['speed'] = sp_res
    save()

    # quality: average surprise (nats per token) on the tokens the agent wrote itself, held-out tasks
    # (PRUNE_QUALITY_ROWS, serve-v6: only the first rows, serve-v5's 40; the next-action states use them all;
    # PRUNE_SKIP_QUALITY, serve-v7: none, the weights are the full model's)
    hq = ([] if os.environ.get('PRUNE_SKIP_QUALITY') == '1' else
          data['heldout'][:int(os.environ.get('PRUNE_QUALITY_ROWS') or 0) or len(data['heldout'])])
    if hq:
        outs, t_q = run([r['ids'] for r in hq], 1, prompt_logprobs=0)
        tot, n, tot_all, n_all = 0.0, 0, 0.0, 0
        for r, o in zip(hq, outs):
            pl = o.prompt_logprobs
            for t in r['own']:
                tot -= pl[t][r['ids'][t]].logprob
                n += 1
            for t in range(1, len(r['ids'])):   # every token: also what the agent reads (tool output, prompt)
                tot_all -= pl[t][r['ids'][t]].logprob
                n_all += 1
        R['quality'] = {'agent_own_nats': round(tot / max(1, n), 4), 'agent_own_tokens': n, 'rows': len(outs),
                        'agent_all_nats': round(tot_all / max(1, n_all), 4), 'all_tokens': n_all, 's': round(t_q, 1)}
        try:   # serve-v6 (B202): the same numbers per role of the token (token_roles.py); never costs the measurements below
            if hq and hq[0].get('roles'):
                names, acc = data.get('role_names') or [], {}
                for r, o in zip(hq, outs):
                    for t in range(1, len(r['ids'])):
                        a = acc.setdefault(int(r['roles'][t]), [0.0, 0])
                        a[0] -= o.prompt_logprobs[t][r['ids'][t]].logprob
                        a[1] += 1
                R['quality']['by_role'] = {(names[g] if 0 <= g < len(names) else str(g)): {'nats': round(v[0] / v[1], 4),
                                                                                            'tokens': v[1]}
                                           for g, v in sorted(acc.items())}
        except Exception as e:
            R['quality']['by_role_error'] = f'{type(e).__name__}: {str(e)[:200]}'
        save()

    # its own next turn at held-out agent states (greedy, up to 512 tokens): which tool call would it make?
    if data.get('turns'):
        t = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in data['turns']],
                            SamplingParams(max_tokens=512, temperature=0, skip_special_tokens=False), use_tqdm=False,
                            **lora_kw)
        R['turns'] = [o.outputs[0].text for o in outs]
        R['turns_s'] = round(time.time() - t, 1)
        save()
        n_sp = int(os.environ.get('PRUNE_TURN_SPEED') or 0)
        if n_sp:   # serve-v7: decode speed on real agent turns (greedy, 512 tokens): one at a time, then 8 at once
            sp_t = {}
            for label, width in (('streams1', 1), ('streams8', 8)):
                t, toks = time.time(), 0
                for i in range(0, n_sp, width):
                    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in data['turns'][i:i + width]],
                                        SamplingParams(max_tokens=512, temperature=0, skip_special_tokens=False),
                                        use_tqdm=False, **lora_kw)
                    toks += sum(len(o.outputs[0].token_ids) for o in outs)
                dt = time.time() - t
                sp_t[label] = {'states': n_sp, 'tokens': toks, 's': round(dt, 1), 'tok_s': round(toks / max(1e-6, dt), 2)}
            R['turn_speed'] = sp_t
            save()
        temp = float(os.environ.get('PRUNE_SAMPLE_TURNS') or 0)
        if temp > 0:  # the same states sampled at the agent's temperature: how often a model agrees with itself
            t = time.time()
            outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in data['turns']],
                                SamplingParams(max_tokens=512, temperature=temp, seed=0, skip_special_tokens=False),
                                use_tqdm=False, **lora_kw)
            R['turns_sampled'] = [o.outputs[0].text for o in outs]
            R['turns_sampled_temperature'] = temp
            R['turns_sampled_s'] = round(time.time() - t, 1)
            save()

    # does it still write sensible code? (greedy, short)
    o = llm.generate([TokensPrompt(prompt_token_ids=data['sanity'])], SamplingParams(max_tokens=120, temperature=0),
                     use_tqdm=False, **lora_kw)[0]
    R['sanity'] = o.outputs[0].text
    save()
    print(json.dumps(R)[:3000], flush=True)


if __name__ == '__main__':
    main()
