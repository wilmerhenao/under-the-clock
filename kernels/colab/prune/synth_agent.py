"""Synthetic agent examples and tokenizer for the pruner's CPU tests (no download): rows in Gemma 4's chat and tool-call
markup with `spans` on the model's turns (as build_sft.trainable_spans), a word-level tokenizer whose chat and tool
markers are single special tokens (as in Gemma 4's), a minimal chat template, and a Gemma 4 config.json with text
and vision parts (the tests shrink it to a tiny model)."""
import json
import random
from pathlib import Path

Q = '<|"|>'
WORDS = 'the a file test error line value fix bug read def return self import class if else for in not none true'.split()
SPECIAL = ['<pad>', '<eos>', '<bos>', '<unk>', '<|turn>', '<turn|>', '<|tool_call>', '<tool_call|>', '<|tool_response>',
           '<tool_response|>', Q, '<|channel>', '<channel|>']
CHAT = ("{{ bos_token }}{% for m in messages %}<|turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
        "{{ m['content'] }}<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}")


def rows(n, seed=0):
    rng = random.Random(seed)

    def words(k):
        return ' '.join(rng.choice(WORDS) for _ in range(k))

    def call():
        k = rng.randrange(3)
        if k == 0:
            return (f'<|tool_call>call:read_file{{filepath:{Q}src/{rng.choice(WORDS)}.py{Q},'
                    f'start_line:{rng.randint(1, 99)}}}<tool_call|>')
        if k == 1:
            return f'<|tool_call>call:run_command{{command:{Q}pytest -x {words(2)}{Q}}}<tool_call|>'
        return (f'<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},old_string:{Q}{words(4)}{Q},new_string:{Q}{words(3)}{Q},'
                f'allow_multiple:false}}<tool_call|>')

    out = []
    for i in range(n):
        t = f'<bos><|turn>system\n{words(30)}<turn|>\n<|turn>user\n{words(40)}<turn|>\n'
        spans = []
        for _ in range(rng.randint(3, 6)):
            t += '<|turn>model\n'
            s = len(t)
            t += f'<|channel>thought\n{words(rng.randint(5, 25))}<channel|>' + call() + '<|tool_response>'
            spans.append([s, len(t)])
            t += f'response:x{{output:{Q}{words(rng.randint(20, 80))}{Q}}}<tool_response|>'
            s = len(t)
            t += words(rng.randint(2, 8)) + '<turn|>\n'
            spans.append([s, len(t)])
            t += f'<|turn>user\n{words(5)}<turn|>\n'
        out.append({'instance_id': f'task_{i}', 'text': t, 'spans': spans})
    return out


def tokenizer_dir(rs, d):
    """A directory with tokenizer.json, tokenizer_config.json (chat template inside) and config.json."""
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    vocab = {w: i for i, w in enumerate(SPECIAL)}
    ws = pre_tokenizers.WhitespaceSplit()
    for w in sorted({w for r in rs for w, _ in ws.pre_tokenize_str(r['text'])} | set(WORDS)):
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in SPECIAL])
    tk.save(str(d / 'tokenizer.json'))
    (d / 'tokenizer_config.json').write_text(json.dumps({
        'tokenizer_class': 'PreTrainedTokenizerFast', 'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
        'unk_token': '<unk>', 'chat_template': CHAT}))
    cfg = Gemma4Config(text_config=Gemma4TextConfig(vocab_size=max(1024, len(vocab) + 16), final_logit_softcapping=30.0,
                                                    hidden_size_per_layer_input=0),   # as the 31B: no per-layer inputs
                       vision_config=Gemma4VisionConfig()).to_dict()
    (d / 'config.json').write_text(json.dumps(cfg))
    return d
