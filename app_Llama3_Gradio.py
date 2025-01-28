import gradio as gr
import os
import torch
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


DESCRIPTION = '''
<div>
<h1 style="text-align: center;">EmoLLM Llama3 V4.0</h1>

<p align="center">
  <a href="https://github.com/SmartFlowAI/EmoLLM/">
    <img src="https://st-app-center-006861-9746-jlroxvg.openxlab.space/media/cda6c1a05dc8ba5b19ad3e7a24920fdf3750c917751202385a6dbc51.png" alt="Logo" width="20%">
  </a>
</p>

<div align="center">

<!-- PROJECT SHIELDS -->
[![OpenXLab_Model][OpenXLab_Model-image]][OpenXLab_Model-url] 

<h2 style="text-align: center;"> EmoLLMIt is a series of mental health models that can support the mental health counseling link of understanding users, supporting users, and helping users. Welcome everyone star~⭐⭐</h2>
<p>https://github.com/SmartFlowAI/EmoLLM</p>
</div>

</div>

[OpenXLab_Model-image]: https://cdn-static.openxlab.org.cn/header/openxlab_models.svg
[OpenXLab_Model-url]: https://openxlab.org.cn/models/detail/chg0901/EmoLLM-Llama3-8B-Instruct3.0

'''

LICENSE = """
<p align="center"> Built with Meta Llama 3 </>
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">

</div>
"""


css = """
h1 {
  text-align: center;
  display: block;
}
<!--
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
-->
"""

# download internlm2 to the base_path directory using git tool
base_path = './EmoLLM-Llama3-8B-Instruct3.0'
os.system(f'git clone https://code.openxlab.org.cn/chg0901/EmoLLM-Llama3-8B-Instruct3.0.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, device_map="auto", torch_dtype=torch.float16).eval()  # to("cuda:0") 
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def chat_llama3_8b(message: str, 
              history: list, 
              temperature: float, 
              max_new_tokens: int,
              top_p: float
             ) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p = top_p,
        eos_token_id=terminators,
    )
    # This will enforce greedy generation (do_sample=False) when the temperature is passed 0, avoiding the crash.             
    if temperature == 0:
        generate_kwargs['do_sample'] = False
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

        

# Gradio block
chatbot=gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='EmoLLM Chat')

with gr.Blocks(fill_height=True, css=css) as demo:
    
    gr.Markdown(DESCRIPTION)
    # gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    gr.ChatInterface(
        fn=chat_llama3_8b,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0,
                      maximum=1, 
                      step=0.1,
                      value=0.95, 
                      label="Temperature", 
                      render=False),
            gr.Slider(minimum=128, 
                      maximum=4096,
                      step=1,
                      value=4096, 
                      label="Max new tokens", 
                      render=False ),
            gr.Slider(minimum=0.0, 
                      maximum=1,
                      step=0.01,
                      value=0.8, 
                      label="Top P", 
                      render=False ),
            # gr.Slider(minimum=128, 
            #           maximum=4096,
            #           step=1,
            #           value=512, 
            #           label="Max new tokens", 
            #           render=False ),
            ],
        examples=[
            ["Please introduce yourself."],
            ["I feel like I'm under a lot of academic pressure at school. Although I really like my major, I've been worrying lately that I won't be able to meet my own expectations, which makes me a little anxious."],
            ["I've been feeling stuck in a relationship lately. I've fallen in love with my friend but I'm afraid that expressing it will ruin our current relationship..."],
            ["I felt like I was stuck in an endless cycle. I woke up feeling heavy, and I had no interest in daily activities. I was bored with work, exercise, and even the things I once enjoyed."],
            ["I've been under a lot of work pressure lately, and there are also some family conflicts"]
            ],
        cache_examples=False,
                     )
    
    gr.Markdown(LICENSE)
    
if __name__ == "__main__":
    demo.launch(share=True)
    
    
