import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


class MyLLM(LLM):
    def __init__(self, *args, **kwargs):
        # a hack to make the script work.
        # stop ray from manipulating CUDA_VISIBLE_DEVICES
        # at the top-level
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        print(args)
        print(kwargs)
        self.custom_tokenizer = AutoTokenizer.from_pretrained(kwargs['model'])
        super().__init__(*args, **kwargs)

    def custom_apply_chat_template_batch(self, questions):
        texts = []
        for question in questions:
            messages = [{'content': question, 'role':'user'}]
            text = self.custom_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        return texts

    def do_generate(self, questions, n, max_tokens=8000, temperature=0.7, top_p=0.95, eval_infer=False):
        texts = self.custom_apply_chat_template_batch(questions)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n)
        outputs =self.generate(texts, sampling_params)
        results = []
        for output in outputs:
            per_question_res = []
            for per in output.outputs:
                per_question_res.append(per.text)
            results.append(per_question_res)
        return results

