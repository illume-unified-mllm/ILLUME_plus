import argparse
import os
import traceback
import logging
from functools import partial
from threading import Thread

import re  # Added for parsing image tokens

import torch
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import TextIteratorStreamer

from transformers import AutoModel, AutoProcessor
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger("http").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

import gradio as gr

from illume.conversation import default_conversation, conv_templates, SeparatorStyle
# from conversation import default_conversation, conv_templates, SeparatorStyle

# --- Global Variables and Model Loading ---
model = None  # Global variable to hold the loaded ILLUME model
args = None  # Global variable to hold command line args
streamer = None  # Global variable to hold command line args

DEFAULT_IMAGE_TOKEN = '<image>'

# Define common resolutions
DEFAULT_RESOLUTIONS = [
    (256, 256), (512, 512), (384, 640), (640, 384), (512, 384),
    (384, 512), (256, 384), (384, 256), (256, 512), (512, 256)
]

DEFAULT_DIFFUSION_RESOLUTIONS = [
    (512, 512), (1024, 1024), (768, 1280), (1280, 768), (1024, 768),
    (768, 1024), (512, 768), (768, 512), (512, 1024), (1024, 512)
]

conv_templates_version = 'qwen2'


def check_image_token_num(image_embed_inds, token_nums=[81, 256], identifier=""):
    image_embed_inds_out = []
    if len(image_embed_inds) != len(token_nums):
        logging.error(
            f"{identifier} Mismatch between number of image token levels ({len(image_embed_inds)}) and expected token_nums ({len(token_nums)})")
        # Handle error appropriately - maybe return None or raise exception
        return None  # Indicate error

    for level, (embed_inds, token_num) in enumerate(zip(image_embed_inds, token_nums)):
        if not len(embed_inds) == token_num:
            logging.warning(
                f"{identifier} Level {level} embed_inds length {len(embed_inds)} not equal to expected {token_num}! Padding/truncating.")
            if len(embed_inds) > token_num:
                embed_inds = embed_inds[:token_num]
            elif len(embed_inds) == 0:
                # Handle empty case - perhaps fill with a default token?
                logging.warning(f"{identifier} Level {level} embed_inds is empty. Filling with zeros.")
                embed_inds = [0] * token_num  # Or a placeholder token ID
            else:
                # Pad with the last token ID
                embed_inds.extend([embed_inds[-1]] * (token_num - len(embed_inds)))
        image_embed_inds_out.append(embed_inds)
    return image_embed_inds_out


def pad_sequence(tokenizer, input_ids, batch_first, padding_value):
    # Assuming input_ids is a list of Tensors
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    # Manually pad if needed, or use torch utils if input_ids are tensors
    # This assumes input_ids are already tensors
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids_padded = torch.flip(input_ids_padded, [1])
    return input_ids_padded


# --- Gradio UI Functions ---
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)
server_error_msg = "**NETWORK ERROR OR SERVER ISSUE. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
server_oom_msg = "**OUT OF GPU MEMORY DETECTED. PLEASE DECREASE THE MAX OUTPUT TOKENS OR IMAGE RESOLUTION AND REGENERATE.**"


def load_demo_refresh_model_list():
    logging.info("load_demo.")
    # Use the conversation template from the loaded model/config
    # Ensure model is loaded before this runs
    if conv_templates_version in conv_templates:
        state = conv_templates[conv_templates_version].copy()
        logging.info(f"Using conversation template: {conv_templates_version}")
    else:
        logging.warning(f"Conversation template '{conv_templates_version}' not found. Using default.")
        # Find a default template name from conv_templates or define one
        default_template_name = next(iter(conv_templates))  # Get the first available template
        state = conv_templates[default_template_name].copy()
    return state


def regenerate(state):
    logging.info("regenerate.")
    if not state.messages or len(state.messages) < 2:
        return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2  # Use state's image

    # Clear the last assistant message
    state.messages[-1][-1] = None

    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2


def http_bot_conditional_then(state, temperature, top_k, top_p,
                              image_gen_temperature, image_gen_top_k, image_gen_top_p, max_output_tokens,
                              llm_cfg_scale, resolution_wh, use_diffusion, diffusion_cfg_scale,
                              diffusion_num_inference_steps):
    if state.mode == 'chat':
        result = yield from http_chat_bot(state, temperature, top_k, top_p, max_output_tokens)
    else:
        result = yield from http_gen_edit_bot(
            state, temperature, top_k, top_p, image_gen_temperature, image_gen_top_k, image_gen_top_p,
            max_output_tokens,
            llm_cfg_scale, resolution_wh, use_diffusion, diffusion_cfg_scale, diffusion_num_inference_steps)
    return result


def clear_history():
    logging.info("clear_history.")
    state = load_demo_refresh_model_list()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2


def add_text(state, text, image, mode):
    global model

    logging.info(f"add_text. Text len: {len(text)}, Image provided: {image is not None}")
    if len(text.strip()) == 0 and image is None:
        state.skip_next = True
        # Keep image in the imagebox if only image was present
        return (state, state.to_gradio_chatbot(), "", image) + (no_change_btn,) * 2

    if state.messages and state.messages[-1][1] and \
            isinstance(state.messages[-1][1], str) and state.messages[-1][1].startswith("**"):
        state = load_demo_refresh_model_list()  # Start fresh after error

    if mode == 'image-generation':
        state = load_demo_refresh_model_list()

    image_process_mode = "Default"

    if image is not None:
        if state.get_images():
            state = load_demo_refresh_model_list()

        if '<image>' not in text:
            text = f'<image>\n{text}'
        text = (text, image, image_process_mode)

    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)  # Placeholder for assistant
    state.skip_next = False
    state.mode = mode
    logging.info(f"Updated state messages: {len(state.messages)}")

    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 2


def stream_response(model, inputs, streamer, prompt, gen_kwargs):
    thread = Thread(target=model.generate, kwargs=dict(
        streamer=streamer,
        **inputs,
        **gen_kwargs
    ))
    thread.start()

    generated_text = prompt

    for new_text in streamer:
        generated_text += new_text
        yield generated_text


# @spaces.GPU
def http_chat_bot(state, temperature, top_k, top_p, max_new_tokens):
    global model, args, streamer  # Use global model and args
    logging.info("http_chat_bot.")

    if state.skip_next:
        logging.warning("Skipping bot generation. skip_next or model not ready.")
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
        return

    if len(state.messages) < 2:
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
        return

    # --- Prepare Inputs for ILLUME ---
    # Get the full prompt from the conversation state
    prompt = state.get_prompt()
    all_images = state.get_images(return_pil=True)

    logging.info(f"Raw Prompt: {prompt}")

    inputs = dict(
        text=prompt,
    )
    # Tokenize the prompt
    # run processors
    inputs = processor(**inputs, return_tensors="pt")
    inputs = inputs.to(model.device)

    # avoid mismatch resolution. process the images alone
    if len(all_images):
        images = []
        for image in all_images:
            images.append(processor.image_processor(image, return_tensors="pt")['pixel_values'].to(model.device))
        pixel_values = images
        inputs['pixel_values'] = pixel_values

    logging.info(f"Input IDs shape: {inputs.input_ids.shape}")

    # Set initial response placeholder
    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2

    # --- MLLM Generation ---’
    gen_kwargs = dict(
        pad_token_id=processor.tokenizer.pad_token_id,
        do_sample=True if temperature > 0 else False,  # Controlled by dynamic sampler now, but keep flag
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id  # Ensure EOS token is set
    )
    logging.info(f"==== request kwargs====\n{gen_kwargs}")

    if max_new_tokens < 1:
        state.messages[-1][-1] = "Exceeds max token length. Please start a new conversation, thanks."
        yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2
        return

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2

    # Stream output
    try:
        for generated_text in stream_response(model, inputs, streamer, prompt, gen_kwargs):
            output = generated_text[len(prompt):].strip()
            state.messages[-1][-1] = output
            yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
    except Exception as e:
        os.system("nvidia-smi")
        logging.info(traceback.print_exc())
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
    return (state, state.to_gradio_chatbot()) + (enable_btn,) * 2


def http_gen_edit_bot(state, temperature, top_k, top_p,
                      image_gen_temperature, image_gen_top_k, image_gen_top_p, max_output_tokens,
                      llm_cfg_scale, resolution_wh, use_diffusion, diffusion_cfg_scale, diffusion_num_inference_steps):
    global model, args  # Use global model and args
    logging.info("http_gen_edit_bot.")

    if state.skip_next:
        logging.warning("Skipping bot generation. skip_next or model not ready.")
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
        return

    if len(state.messages) < 2:
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
        return

    # --- Prepare Inputs for ILLUME ---
    # Get the full prompt from the conversation state
    all_images = state.get_images(return_pil=True)

    # read resolution from user defined.
    h_str, w_str = resolution_wh.split('x')
    h_out, w_out = int(h_str), int(w_str)

    if use_diffusion:
        h_out, w_out = (h_out // 2, w_out // 2)
    else:
        h_out, w_out = (h_out, w_out)
    ratio_tag = f"<height_{h_out}><width_{w_out}>"

    input_state = state.copy()

    # prepare the text.
    original_image_sizes = None
    if len(all_images):
        # image editing.
        original_image_sizes = [image.size for image in all_images]
        logging.info(f"original_image_sizes: {original_image_sizes}")

        all_images = [processor.transform_image_nearest_resolution_ratio(image) for image in all_images]

        inputs = dict(
            images=all_images
        )

        image_inputs = processor.image_processor(**inputs, return_tensors="pt")
        image_inputs = image_inputs.to(model.device)

        # overwrite the output resolution
        h, w = image_inputs['pixel_values'].shape[-2:]
        ratio_tag = f"<height_{h}><width_{w}>"
        h_out, w_out = h, w

        unconditional_text = f"{ratio_tag}{DEFAULT_IMAGE_TOKEN}\nReconstruct the image according to the given image\n"  # of {ratio_tag}

        instruction, img, image_process_type = input_state.messages[-2][-1]
        instruction = instruction.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        text = f"{ratio_tag}{DEFAULT_IMAGE_TOKEN}\nPlease edit the image according to the instruction: {instruction}\n"
        input_state.messages[-2][-1] = text, img, image_process_type
    else:
        # image generation
        unconditional_text = f"Generate a random image of {ratio_tag}\n"

        text = input_state.messages[-2][-1]
        logging.info(f"Current text is {text}")
        text = f"Generate an image of {ratio_tag}, the content of image is {text}\n"
        input_state.messages[-2][-1] = text
        logging.info(f"After formatting. current text is {text}")
        image_inputs = {}

    # Calculate ratio tag based on base resolution from config
    logging.info(f"Target Resolution: {h_out}x{w_out}, Ratio Tag: {ratio_tag}")
    target_image_resolution = (h_out, w_out)
    prompt = input_state.get_prompt()
    logging.info(f"Raw Prompt: {prompt}")

    # Tokenize the prompt
    inputs = dict(
        text=prompt + ratio_tag,
    )

    inputs = processor(**inputs, return_tensors="pt")
    inputs = inputs.to(model.device)
    inputs.update(image_inputs)

    conv_uncond = conv_templates[conv_templates_version].copy()
    conv_uncond.append_message(conv_uncond.roles[0], unconditional_text)
    conv_uncond.append_message(conv_uncond.roles[1], None)
    unconditional_prompt_str = conv_uncond.get_prompt()  # Add ratio tag

    print("unconditional_prompt_str", unconditional_prompt_str)

    uncond_inputs = dict(
        text=unconditional_prompt_str + ratio_tag,
        images=all_images
    )

    uncond_inputs = processor(**uncond_inputs, return_tensors="pt")
    uncond_inputs = uncond_inputs.to(model.device)

    logging.info(f"Input IDs shape: {inputs.input_ids.shape}")

    # Set initial response placeholder
    state.messages[-1][-1] = "image generating..."
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2

    gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    image_gen_kwargs = dict(
        negative_image_prompt_ids=uncond_inputs.input_ids,
        negative_image_prompt_attention_mask=uncond_inputs.attention_mask,
        target_image_resolution=target_image_resolution,
        guidance_scale=llm_cfg_scale,
        image_semantic_temperature=image_gen_temperature,
        image_semantic_top_k=image_gen_top_k,
        image_semantic_top_p=image_gen_top_p,
        image_pixel_temperature=image_gen_temperature,
        image_pixel_top_k=image_gen_top_k * 3,
        image_pixel_top_p=image_gen_top_p,
    )

    # --- MLLM Generation ---
    generated_image = None
    generated_text = ""
    try:
        with torch.inference_mode():  # Ensure no gradients are calculated
            output_ids = model.generate(
                **inputs,
                use_cache=True,
                **gen_kwargs,
                **image_gen_kwargs,
                pad_token_id = processor.tokenizer.pad_token_id,
                eos_token_id = processor.tokenizer.eos_token_id,
            )

            output_ids = output_ids[:, inputs['input_ids'].shape[1]:]

        logging.info(f"Generated output IDs shape: {output_ids.shape}")

        # Decode the generated IDs, skipping prompt and special tokens
        # We need to decode the full output first to parse image tokens
        # output_ids shape is likely (batch_size, seq_len), batch_size=1 here
        generated_ids = output_ids[0]  # Get only generated tokens
        full_output_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logging.info(f"Full decoded output: {full_output_text}")

        # --- Parse Output for Image Tokens and Text ---
        # Ensure levels are sorted and create the final list
        generated_text, image_embed_inds_list, list_image_token_parts = processor.parse_text_image(full_output_text,
                                                                                                   DEFAULT_IMAGE_TOKEN)

        assert len(image_embed_inds_list) == 1, 'The number of generated image should be 1.'
        image_embed_inds = image_embed_inds_list[0]
        # logging.info(f"The generated text: {full_output_text}")
        logging.info(f"Parsed generated text (image presents as {DEFAULT_IMAGE_TOKEN}): {generated_text}")

        # Update chat with generated text first
        state.messages[-1][-1] = "vision tokenizer decoding..."  # Remove cursor
        yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2  # Yield text update

        # --- Image Detokenization ---
        if any(image_embed_inds):
            logging.info("Image tokens found. Attempting detokenization...")
            yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 2

            samples = processor.decode_images(image_embed_inds_list, target_resolution=target_image_resolution,
                                              use_diffusion=use_diffusion, diffusion_cfg_scale=diffusion_cfg_scale,
                                              diffusion_num_inference_steps=diffusion_num_inference_steps)
            generated_image = samples[0]
            if use_diffusion:
                logging.info(
                    f"Using Diffusion Decoder (cfg: {diffusion_cfg_scale}, steps: {diffusion_num_inference_steps}) Image size: {generated_image.size}")
            else:
                logging.info(f"Using VQ Tokenizer Decoder. Image size: {generated_image.size}")

            if generated_image:
                if original_image_sizes is not None and len(
                        original_image_sizes) == 1:  # editing task, unpad and resize image to original size
                    original_size = original_image_sizes[0]
                    logging.info(f"original size: {original_size}. Output Image size: {generated_image.size}")
                    # generated_image = processor.unpad_and_resize_back(generated_image, original_size[0], original_size[1])
                    logging.info(f"final image size: {generated_image.size}")
                logging.info("Image successfully generated.")
                # <image> is placeholder.

            logging.info("Image successfully generated.")
            # <image> is placeholder.
            state.messages[-1][-1] = (DEFAULT_IMAGE_TOKEN, [generated_image], list_image_token_parts)
        else:
            # No image tokens generated
            state.messages[-1][-1] = generated_text  # Final text without image

        # Final yield with potentially updated message (text + image)
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"CUDA OutOfMemoryError during generation: {e}\n{traceback.format_exc()}")
        state.messages[-1][-1] = server_oom_msg
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2
    except Exception as e:
        logging.error(f"Error during model generation or detokenization: {e}\n{traceback.format_exc()}")
        state.messages[-1][-1] = f"{server_error_msg}\n```\n{traceback.format_exc()}\n```"  # Show traceback in error
        yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 2

    logging.info(f"Final Assistant Message Length: {len(state.messages[-1][-1])}")


def update_resolution_dropdown(diffusion_enabled, current_resolution_str):
    logging.info(f"Updating resolution dropdown. Diffusion: {diffusion_enabled}, Current: {current_resolution_str}")
    current_h_str, current_w_str = current_resolution_str.split('x')
    current_h, current_w = int(current_h_str), int(current_w_str)

    if diffusion_enabled:
        new_h, new_w = int(current_h) * 2, int(current_w) * 2

        if (new_h, new_w) not in DEFAULT_DIFFUSION_RESOLUTIONS:
            new_h, new_w = DEFAULT_DIFFUSION_RESOLUTIONS[0]
        new_value_str = f"{new_h}x{new_w}"
        return gr.Dropdown.update(choices=[f'{h}x{w}' for h, w in DEFAULT_DIFFUSION_RESOLUTIONS],
                                  value=new_value_str)
    else:
        new_h, new_w = int(current_h) // 2, int(current_w) // 2

        if (new_h, new_w) not in DEFAULT_RESOLUTIONS:
            new_h, new_w = DEFAULT_RESOLUTIONS[0]
        new_value_str = f"{new_h}x{new_w}"

        return gr.Dropdown.update(choices=[f'{h}x{w}' for h, w in DEFAULT_RESOLUTIONS],
                                  value=new_value_str)


# --- Gradio Layout ---
title_markdown = """
<div style="display: flex; align-items: center; padding: 20px; border-radius: 10px; background-color: #f0f0f0;">
  <div>
    <h1 style="margin: 0;"> ILLUME+: Illuminating Unified MLLM with Dual Visual Tokenization and Diffusion Refinement</h1>
    <h2 style="margin: 10px 0;">
      <a href="https://arxiv.org/abs/2504.01934" target="_blank" rel="noopener noreferrer">Paper</a> |
      <a href="https://github.com/illume-unified-mllm/ILLUME_plus" target="_blank" rel="noopener noreferrer">Code</a> |
      <a href="https://huggingface.co/collections/ILLUME-MLLM/illume-models-683b3916f5af2d0a015b3477" target="_blank" rel="noopener noreferrer">Model</a> |
      <a href="https://illume-unified-mllm.github.io/" target="_blank" rel="noopener noreferrer">Project Page</a>
    </h2>
    <ul style="margin: 20px 0; padding-left: 20px;">
      <li><strong>1.</strong> Enter text and/or upload an image.</li>
      <li><strong>2.</strong> Click the 💬 <strong>Chat</strong> button for image inputted conversations</li>
      <li><strong>3.</strong> Click the 🖼️ <strong>Generate</strong> for image generation and image editing.</li>
      <li><strong>4.</strong> (Optional) Enable Diffusion Decoder for image super resolution decoding.
      <li><strong>5.</strong> Adjust generation parameters if needed.
        <br/><strong>💡 Tip 1:</strong> For better image generation quality, we recommend setting <code>temperature = 1.0</code>, <code>top_k = 2048</code>, <code>top_p = 1.0</code>, <code>llm_cfg = 2.0</code>.
        <br/><strong>💡 Tip 2:</strong> For better image editing quality, we recommend setting <code>temperature = 0.7</code>, <code>top_k = 512</code>, <code>top_p = 0.8</code>, <code>llm_cfg = 1.5</code>.
        <br/><strong>💡 Tip 3:</strong> For diffusion decoder, CFG scale of 1.5 or 2.0 is enough.
      </li>
    </ul>
  </div>
</div>
"""

learn_more_markdown = ("""
## Citation


    @article{huang2025illume_plus,
      title={ILLUME+: Illuminating Unified MLLM with Dual Visual Tokenization and Diffusion Refinement},
      author={Huang, Runhui and Wang, Chunwei and Yang, Junwei and Lu, Guansong and Yuan, Yunlong and Han, Jianhua and Hou, Lu and Zhang, Wei and Hong, Lanqing and Zhao, Hengshuang and Xu, Hang}
      journal={arXiv preprint arXiv:2504.01934},
      year={2025}
    }
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
.message-row img {
    max-width: 80%;
    max-height: 400px;
    height: auto;
    display: block;
    margin-top: 10px;
    margin-bottom: 5px;
    border-radius: 5px;
    border: 1px solid #e0e0e0; /* Add a light border */
}
.avatar-container img {
    padding: 0px !important;
}
/* Style for resolution dropdown */
#resolution_dropdown .gradio-dropdown {
    min-width: 150px !important;
}
"""


def build_demo(embed_mode):
    textbox = gr.Textbox(label="Text Input / Prompt", show_label=False,
                         placeholder="Enter text prompt. Ask about the image or request image generation...",
                         container=False, scale=8)

    with gr.Blocks(title="ILLUME Demo", theme=gr.themes.Default(), css=block_css) as demo:
        conversation_state = gr.State()  # Holds conversation state (instance of illume.conversation.Conversation)

        if not embed_mode:
            gr.HTML(title_markdown)

        with gr.Row():
            with gr.Column(scale=2):
                imagebox = gr.Image(type="pil", label="Input Image", height=300)

                # Text Generation Parameters
                with gr.Accordion("Text Generation Parameters", open=True):
                    temperature = gr.Slider(
                        minimum=0.0, maximum=1.5, value=1.0, step=0.1,
                        label="Temperature",
                        info="Controls randomness of the output (higher = more diverse)."
                    )
                    top_k = gr.Slider(
                        minimum=1, maximum=4096, value=128, step=1,
                        label="Top-K",
                    )
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                        label="Top-P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=128, maximum=8192, value=1024, step=128,
                        label="Max Output Tokens",
                    )

                # Image Generation Parameters
                with gr.Accordion("Image Generation Parameters", open=True):
                    image_gen_temperature = gr.Slider(
                        minimum=0.0, maximum=1.5, value=1.0, step=0.1,
                        label="Temperature",
                    )
                    image_gen_top_k = gr.Slider(
                        minimum=1, maximum=4096 * 2, value=2048, step=32,
                        label="Top-K",
                        info="Recommended value for better image generation: 2048."
                    )
                    image_gen_top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=1.0, step=0.05,
                        label="Top-P",
                    )

                    resolution_wh_dropdown = gr.Dropdown(
                        [f'{h}x{w}' for h, w in DEFAULT_RESOLUTIONS],
                        value="512x512",
                        label="Output Resolution (HxW)",
                        elem_id="resolution_dropdown",
                        info="Select target size for generated images."
                    )

                    llm_cfg_scale = gr.Slider(
                        minimum=1.0, maximum=10.0, value=2.0, step=0.1,
                        label="LLM CFG Scale",
                        info="Guidance for text-to-image conditioning (higher = stricter to prompt)."
                    )

                    with gr.Accordion("Diffusion Decoder (Optional)", open=False):
                        use_diffusion_checkbox = gr.Checkbox(
                            value=False, interactive=True,
                            label="Use diffusion decoder for image generation",
                            info="Enable diffusion decoder."
                        )
                        diffusion_cfg_scale = gr.Slider(
                            minimum=1.0, maximum=15.0, value=2.0, step=0.1,
                            label="Diffusion CFG Scale",
                            info="Guidance strength for diffusion decoder."
                        )
                        diffusion_num_inference_steps = gr.Slider(
                            minimum=5, maximum=100, value=20, step=5,
                            label="Diffusion Inference Steps",
                            info="Number of steps during denoising."
                        )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="ILLUME Chat",
                    layout="bubble",
                    height=650,  # Increased height
                    bubble_full_width=False,
                    render_markdown=True  # Crucial for images
                )
                with gr.Row():
                    textbox.render()
                with gr.Row(elem_id="buttons") as button_row:
                    chat_btn = gr.Button(value="💬 Chat", variant="primary")
                    gen_btn = gr.Button(value="🖼️ Generate", variant="secondary")
                with gr.Row(elem_id="additional-buttons") as button_row_additional:
                    regenerate_btn = gr.Button(value="🔄 Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️ Clear History", interactive=False)

        # Update examples for ILLUME
        with gr.Accordion("Examples (Click to Load)", open=True):
            with gr.Row():
                gr.Examples(examples=[
                    ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/1.png",
                     "What are they doing?"],
                    ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/2.png",
                     "Depict the image in detail."],
                    ["../configs/data_configs/test_data_examples/ImageUnderstandingExample/images/3.png",
                     "Parse the table"],
                ], inputs=[imagebox, textbox], label='Image Understanding Examples')

                gr.Examples(examples=[
                    [None, "a cat with a hat."],
                    [None, "a smiling child."],
                    [None, "tiger cub playing with soccer ball"],
                    [None, "screenshot from a 16 bit platform game in a lush green landscape"],
                    [None, "Old car in kandy sri lanka,lake road,flower, bright, sunny, orange sky."],
                    [None, "Create a vibrant painting of a tropical beach at sunset."],
                ], inputs=[imagebox, textbox], label='Image Generation Examples')

                gr.Examples(examples=[
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/0.jpg",
                     "Change the color of the boots to a deep forest green"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/2.jpg",
                     "Remove the dried flowers"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/3.jpg",
                     "Change it into winter"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/4.jpg",
                     "Delete the tennis racket from the man's hand"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/5.jpg",
                     "Show me this as it would appear in a comic book"],
                    ["../configs/data_configs/test_data_examples/EditingSingleTurnExample/images/1.jpg",
                     "Add a hat on the dog"],
                ], inputs=[imagebox, textbox], label='Image Editing Examples')

        if not embed_mode:
            gr.Markdown(learn_more_markdown)

        # Register listeners
        btn_list = [regenerate_btn, clear_btn]
        parameter_chat_inputs = [temperature, top_k, top_p, max_output_tokens]
        parameter_gen_edit_inputs = [temperature, top_k, top_p,
                                     image_gen_temperature, image_gen_top_k, image_gen_top_p, max_output_tokens,
                                     llm_cfg_scale, resolution_wh_dropdown,
                                     use_diffusion_checkbox, diffusion_cfg_scale, diffusion_num_inference_steps]

        regenerate_btn.click(
            regenerate,
            [conversation_state],
            [conversation_state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot_conditional_then,
            [conversation_state] + parameter_gen_edit_inputs,
            [conversation_state, chatbot] + btn_list,
        )

        clear_btn.click(
            clear_history,
            None,
            [conversation_state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        # Default use chat.
        textbox.submit(
            partial(add_text, mode="chat"),
            [conversation_state, textbox, imagebox],
            [conversation_state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_chat_bot,
            [conversation_state] + parameter_chat_inputs,
            [conversation_state, chatbot] + btn_list,
        )

        # Regular Vision-language Chat
        chat_btn.click(partial(add_text, mode="chat"),
                       [conversation_state, textbox, imagebox],
                       [conversation_state, chatbot, textbox, imagebox] + btn_list,
                       queue=False
                       ).then(
            http_chat_bot,
            [conversation_state] + parameter_chat_inputs,
            [conversation_state, chatbot] + btn_list,
        )

        # Image Generation
        gen_btn.click(
            partial(add_text, mode="image-generation"),
            [conversation_state, textbox, imagebox],
            [conversation_state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_gen_edit_bot,
            [conversation_state] + parameter_gen_edit_inputs,
            [conversation_state, chatbot] + btn_list
        )

        use_diffusion_checkbox.change(
            fn=update_resolution_dropdown,
            inputs=[use_diffusion_checkbox, resolution_wh_dropdown],
            outputs=[resolution_wh_dropdown],
            queue=False
        )

        # Load initial state when demo starts
        demo.load(
            load_demo_refresh_model_list,
            None,
            conversation_state,
            queue=False
        )
    return demo


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- Add arguments for ILLUME configs and checkpoints ---
    parser.add_argument("--model_name", type=str, default="ILLUME-MLLM/illume_plus-qwen-2_5-3b-hf",
                        help="Name for builder.")
    parser.add_argument("--torch_dtype", type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'],
                        help="Computation data type.")

    parser.add_argument("--diffusion_decoder_path", type=str, default='ILLUME-MLLM/dualvitok-sdxl-decoder',
                        help="Path to Diffusion Decoder checkpoint (.pt). Required if using diffusion.")

    parser.add_argument("--tokenizer_path", type=str, default='ILLUME-MLLM/dualvitok',
                        help="Path to Tokenizer config file (e.g., tokenizer_config.py).")

    # --- End ILLUME arguments ---
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--embed", action="store_true", help="Run in embed mode (minimal UI)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, cpu).")

    args = parser.parse_args()

    # --- Model Loading ---
    # --- Model Loading ---set
    # Set device
    if "cuda" in args.device and torch.cuda.is_available():
        device = args.device
        local_rank = 0  # Assume single GPU for Gradio unless configured otherwise
        torch.cuda.set_device(local_rank)  # Set default CUDA device
    else:
        device = "cpu"
        local_rank = -1  # Indicate CPU
    logging.info(f"Using device: {device}")

    args.torch_dtype = dict(fp16=torch.float16, fp32=torch.float32, bf16=torch.bfloat16)[args.torch_dtype]

    num_gpus = torch.cuda.device_count()
    logging.info(f"Detected {num_gpus} CUDA devices.")
    # Determine device assignments
    if num_gpus >= 3:
        # Assign to separate GPUs if we have at least 3
        mllm_device = torch.device('cuda:0')
        vq_device = torch.device('cuda:1')
        diffusion_device = torch.device('cuda:2')
        logging.info("Assigning models: MLLM -> cuda:0, VQ -> cuda:1, Diffusion -> cuda:2")
    elif num_gpus == 2:
        # Assign MLLM to GPU 0, VQ and Diffusion to GPU 1
        mllm_device = torch.device('cuda:0')
        vq_device = torch.device('cuda:1')
        diffusion_device = torch.device('cuda:1')
        logging.info("Assigning models: MLLM -> cuda:0, VQ & Diffusion -> cuda:1")
    elif num_gpus == 1:
        # Assign all to the single available GPU
        mllm_device = torch.device('cuda:0')
        vq_device = torch.device('cuda:0')
        diffusion_device = torch.device('cuda:0')
        logging.info("Assigning all models to cuda:0")
    else:
        # Fallback to CPU if no GPUs are available
        mllm_device = torch.device('cpu')
        vq_device = torch.device('cpu')
        diffusion_device = torch.device('cpu')
        logging.info("Warning: No CUDA devices found. Assigning all models to CPU.")

    # Build the ILLUME model instance
    logging.info("Building ILLUME model...")
    # prepare models and processors
    model = AutoModel.from_pretrained(
        args.model_name,
        attn_implementation='flash_attention_2',  # OR 'sdpa' for Ascend NPUs
        torch_dtype=args.torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().to(args.torch_dtype).cuda()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    # set the vision tokenizer for decoding image.
    dualvitok = AutoModel.from_pretrained(args.tokenizer_path,
                                          torch_dtype=args.torch_dtype,
                                          trust_remote_code=True).eval().to(vq_device)
    processor.set_vision_tokenizer(dualvitok)

    # (Optional): set the sdxl diffusion decoder. It will enable upsample 2x image resolution.
    processor.load_diffusion_vision_detokenizer(args.diffusion_decoder_path, device=diffusion_device)

    # Assign device to model for later use
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

    logging.info("ILLUME model built successfully.")

    demo = build_demo(args.embed)
    demo.queue(
        max_size=10,
        api_open=False
    ).launch(
        share=args.share,
        server_name="0.0.0.0"  # Allow network access if not using --share
    )
