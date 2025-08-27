import gradio as gr
import numpy as np
import random
import os
import json
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Event
import atexit
import signal
import requests

mp.set_start_method('spawn', force=True)

from diffusers import DiffusionPipeline, QwenImageEditPipeline
import torch
from src.examples.tools.prompt_utils import rewrite

model_repo_id = "Qwen/Qwen-Image"
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1440

NUM_GPUS_TO_USE = int(os.environ.get("NUM_GPUS_TO_USE", torch.cuda.device_count()))
TASK_QUEUE_SIZE = int(os.environ.get("TASK_QUEUE_SIZE", 100))
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 300))

print(f"Config: Using {NUM_GPUS_TO_USE} GPUs, queue size {TASK_QUEUE_SIZE}, timeout {TASK_TIMEOUT} seconds")


class GPUWorker:
    def __init__(self, gpu_id, model_repo_id, task_queue, result_queue, stop_event):
        self.gpu_id = gpu_id
        self.model_repo_id = model_repo_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        self.pipe = None
        self.pipe_edit = None

    def initialize_model(self):
        """Initialize the model on the specified GPU"""
        try:
            torch.cuda.set_device(self.gpu_id)
            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

            self.pipe = DiffusionPipeline.from_pretrained(self.model_repo_id, torch_dtype=torch_dtype)
            self.pipe = self.pipe.to(self.device)
            self.pipe_edit = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch_dtype).to(self.device)
            print(f"GPU {self.gpu_id} model initialized successfully")
            return True
        except Exception as e:
            print(f"GPU {self.gpu_id} model initialization failed: {e}")
            return False

    def process_task(self, task):
        """Process a single task"""
        try:
            task_id = task['task_id']
            prompt = task['prompt']
            negative_prompt = task['negative_prompt']
            seed = task['seed']
            width = task['width']
            height = task['height']
            guidance_scale = task['guidance_scale']
            num_inference_steps = task['num_inference_steps']
            progress_callback = task['progress_callback']

            lora1 = task['lora1']
            lora1_scale = task['lora1_scale']
            lora2 = task['lora2']
            lora2_scale = task['lora2_scale']
            lora3 = task['lora3']
            lora3_scale = task['lora3_scale']

            if lora1:
                self.pipe.load_lora_weights(lora1, adapter_name=lora1)
            if lora2:
                self.pipe.load_lora_weights(lora2, adapter_name=lora2)
            if lora3:
                self.pipe.load_lora_weights(lora3, adapter_name=lora3)

            adapter_names = []
            adapter_weights = []
            if lora1:
                adapter_names.append(lora1)
                adapter_weights.append(lora1_scale)
            if lora2:
                adapter_names.append(lora2)
                adapter_weights.append(lora2_scale)
            if lora3:
                adapter_names.append(lora3)
                adapter_weights.append(lora3_scale)

            if adapter_names:
                self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

            def step_callback(pipe, i, t, callback_kwargs):
                if progress_callback:
                    progress_callback(0.2 + i / num_inference_steps * 0.8, desc="GPU processing...")
                return callback_kwargs

            generator = torch.Generator(device=self.device).manual_seed(seed)

            with torch.cuda.device(self.gpu_id):
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    true_cfg_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=step_callback if progress_callback else None
                ).images[0]

            return {
                'task_id': task_id,
                'image': image,
                'success': True,
                'gpu_id': self.gpu_id
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'gpu_id': self.gpu_id
            }

    def run(self):
        """Worker main loop"""
        if not self.initialize_model():
            return

        print(f"GPU {self.gpu_id} worker starting")

        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break

                result = self.process_task(task)
                self.result_queue.put(result)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU {self.gpu_id} worker exception: {e}")
                continue

        print(f"GPU {self.gpu_id} worker stopping")

def gpu_worker_process(gpu_id, model_repo_id, task_queue, result_queue, stop_event):
    worker = GPUWorker(gpu_id, model_repo_id, task_queue, result_queue, stop_event)
    worker.run()

def download_lora(url):
    if not url:
        return None
    try:
        lora_dir = "/workspaces/projects/loras"
        os.makedirs(lora_dir, exist_ok=True)
        file_name = os.path.basename(url)
        local_path = os.path.join(lora_dir, file_name)
        if not os.path.exists(local_path):
            print(f"Downloading LoRA from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded LoRA to {local_path}")
        return local_path
    except Exception as e:
        print(f"Failed to download LoRA from {url}: {e}")
        return None

class MultiGPUManager:
    def __init__(self, model_repo_id, num_gpus=None, task_queue_size=100):
        self.model_repo_id = model_repo_id
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.task_queue = Queue(maxsize=task_queue_size)
        self.result_queue = Queue()
        self.stop_event = Event()
        self.worker_processes = []
        self.task_counter = 0
        self.pending_tasks = {}

        print(f"Initializing Multi-GPU Manager with {self.num_gpus} GPUs, queue size {task_queue_size}")

    def start_workers(self):
        for gpu_id in range(self.num_gpus):
            process = Process(target=gpu_worker_process,
                            args=(gpu_id, self.model_repo_id, self.task_queue,
                                  self.result_queue, self.stop_event))
            process.start()
            self.worker_processes.append(process)

        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()

        print(f"All {self.num_gpus} GPU workers have started")

    def _process_results(self):
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=1)
                task_id = result['task_id']

                if task_id in self.pending_tasks:
                    self.pending_tasks[task_id]['result'] = result
                    self.pending_tasks[task_id]['event'].set()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Result processing thread exception: {e}")
                continue

    def submit_task_with_progress(self, prompt, negative_prompt="", seed=42, width=1664, height=928,
                                 guidance_scale=4, num_inference_steps=50, timeout=300, progress_callback=None,
                                 lora1=None, lora1_scale=1.0, lora2=None, lora2_scale=1.0, lora3=None, lora3_scale=1.0):
        task_id = f"task_{self.task_counter}_{time.time()}"
        self.task_counter += 1

        task = {
            'task_id': task_id,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'width': width,
            'height': height,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'progress_callback': progress_callback,
            'lora1': lora1,
            'lora1_scale': lora1_scale,
            'lora2': lora2,
            'lora2_scale': lora2_scale,
            'lora3': lora3,
            'lora3_scale': lora3_scale
        }

        result_event = threading.Event()
        self.pending_tasks[task_id] = {
            'event': result_event,
            'result': None,
            'submitted_time': time.time()
        }

        try:
            self.task_queue.put(task, timeout=10)

            if progress_callback:
                progress_callback(0.2, desc="Task submitted, waiting for GPU processing...")

            start_time = time.time()
            while not result_event.is_set():
                if result_event.wait(timeout=2):
                    break

                if time.time() - start_time > timeout:
                    del self.pending_tasks[task_id]
                    return {'success': False, 'error': 'Task timeout'}

            if progress_callback:
                progress_callback(0.8, desc="GPU processing complete...")

            result = self.pending_tasks[task_id]['result']
            del self.pending_tasks[task_id]
            return result

        except queue.Full:
            del self.pending_tasks[task_id]
            return {'success': False, 'error': 'Task queue is full'}
        except Exception as e:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
            return {'success': False, 'error': str(e)}

    def stop(self):
        print("Stopping Multi-GPU Manager...")
        self.stop_event.set()

        for _ in range(self.num_gpus):
            try:
                self.task_queue.put(None, timeout=1)
            except queue.Full:
                pass

        for process in self.worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()

        print("Multi-GPU Manager has stopped")

gpu_manager = None

def initialize_gpu_manager():
    global gpu_manager
    if gpu_manager is None:
        try:
            if torch.cuda.is_available():
                print(f"Detected {torch.cuda.device_count()} GPUs")

            gpu_manager = MultiGPUManager(
                model_repo_id,
                num_gpus=NUM_GPUS_TO_USE,
                task_queue_size=TASK_QUEUE_SIZE
            )
            gpu_manager.start_workers()
            print("GPU Manager initialized successfully")
        except Exception as e:
            print(f"GPU Manager initialization failed: {e}")
            gpu_manager = None

def get_image_size(aspect_ratio):
    if aspect_ratio == "1:1":
        return 1328, 1328
    elif aspect_ratio == "16:9":
        return 1664, 928
    elif aspect_ratio == "9:16":
        return 928, 1664
    elif aspect_ratio == "4:3":
        return 1472, 1140
    elif aspect_ratio == "3:4":
        return 1140, 1472
    else:
        return 1328, 1328

def infer_text_to_image(
    prompt,
    negative_prompt="",
    seed=42,
    randomize_seed=False,
    aspect_ratio="16:9",
    guidance_scale=5,
    num_inference_steps=50,
    lora1=None,
    lora1_scale=1.0,
    lora2=None,
    lora2_scale=1.0,
    lora3=None,
    lora3_scale=1.0,
    progress=gr.Progress(track_tqdm=True)
):
    lora1_path = download_lora(lora1)
    lora2_path = download_lora(lora2)
    lora3_path = download_lora(lora3)
    global gpu_manager

    if gpu_manager is None:
        if progress:
            progress(0.1, desc="Initializing GPU manager...")
        initialize_gpu_manager()

        if gpu_manager is None:
            print("GPU manager initialization failed, unable to process task")
            from PIL import Image
            error_image = Image.new('RGB', (512, 512), color='gray')
            return error_image, seed

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    width, height = get_image_size(aspect_ratio)
    original_prompt = prompt

    if progress:
        progress(0.1, desc="Optimizing prompt...")
    prompt = rewrite(prompt)
    print(f"Prompt: {prompt}, original_prompt: {original_prompt}")

    if progress:
        progress(0.3, desc="Submitting task to GPU queue...")

    result = gpu_manager.submit_task_with_progress(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        timeout=TASK_TIMEOUT,
        progress_callback=progress,
        lora1=lora1_path,
        lora1_scale=lora1_scale,
        lora2=lora2_path,
        lora2_scale=lora2_scale,
        lora3=lora3_path,
        lora3_scale=lora3_scale,
    )

    if result['success']:
        if progress:
            progress(0.9, desc="Saving result...")
        image = result['image']
        gpu_id = result['gpu_id']
        print(f"Task completed using GPU {gpu_id}")

        if progress:
            progress(1.0, desc="Done!")
        return image, seed
    else:
        print(f"Inference failed: {result['error']}")
        from PIL import Image
        error_image = Image.new('RGB', (512, 512), color='red')
        return error_image, seed

def infer_image_edit(
    image,
    image2,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=50,
    rewrite_prompt=True,
    num_images_per_prompt=1,
    lora1_edit=None,
    lora1_scale_edit=1.0,
    lora2_edit=None,
    lora2_scale_edit=1.0,
    lora3_edit=None,
    lora3_scale_edit=1.0,
    use_lightning_edit=True,
    progress=gr.Progress(track_tqdm=True),
):
    lora1_edit_path = download_lora(lora1_edit)
    lora2_edit_path = download_lora(lora2_edit)
    lora3_edit_path = download_lora(lora3_edit)
    global gpu_manager

    if gpu_manager is None:
        if progress:
            progress(0.1, desc="Initializing GPU manager...")
        initialize_gpu_manager()

        if gpu_manager is None:
            print("GPU manager initialization failed, unable to process task")
            from PIL import Image
            error_image = Image.new('RGB', (512, 512), color='gray')
            return error_image, seed

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    from src.examples.tools.prompt_utils import polish_edit_prompt
    if rewrite_prompt:
        prompt = polish_edit_prompt(prompt, image)
        print(f"Rewritten Prompt: {prompt}")

    if image2:
        from PIL import Image
        composite_image = Image.new('RGB', (image.width + image2.width, max(image.height, image2.height)))
        composite_image.paste(image, (0, 0))
        composite_image.paste(image2, (image.width, 0))
        image = composite_image

    # For now, we are not using the multi-gpu manager for image editing.
    # We will use the first gpu worker's pipe_edit directly.
    if len(gpu_manager.worker_processes) > 0:
        worker = gpu_manager.worker_processes[0]

        if use_lightning_edit:
            worker.pipe_edit.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors", adapter_name="lightning")

        if lora1_edit_path:
            worker.pipe_edit.load_lora_weights(lora1_edit_path, adapter_name=lora1_edit_path)
        if lora2_edit_path:
            worker.pipe_edit.load_lora_weights(lora2_edit_path, adapter_name=lora2_edit_path)
        if lora3_edit_path:
            worker.pipe_edit.load_lora_weights(lora3_edit_path, adapter_name=lora3_edit_path)

        adapter_names = []
        adapter_weights = []

        if use_lightning_edit:
            adapter_names.append("lightning")
            adapter_weights.append(1.0)
        if lora1_edit_path:
            adapter_names.append(lora1_edit_path)
            adapter_weights.append(lora1_scale_edit)
        if lora2_edit_path:
            adapter_names.append(lora2_edit_path)
            adapter_weights.append(lora2_scale_edit)
        if lora3_edit_path:
            adapter_names.append(lora3_edit_path)
            adapter_weights.append(lora3_scale_edit)

        if adapter_names:
            worker.pipe_edit.set_adapters(adapter_names, adapter_weights=adapter_weights)

        generator = torch.Generator(device=worker.device).manual_seed(seed)
        result_image = worker.pipe_edit(
            image=image,
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_images_per_prompt
        ).images[0]
        return result_image, seed
    else:
        print("No GPU workers available")
        from PIL import Image
        error_image = Image.new('RGB', (512, 512), color='red')
        return error_image, seed

with gr.Blocks() as demo:
    gr.Markdown("# Qwen-Image and Qwen-Image-Edit")
    with gr.Tabs():
        with gr.TabItem("Text-to-Image"):
            gr.Markdown("## Text-to-Image")
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0, variant="primary")
            result = gr.Image(label="Result", show_label=False)
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Text(
                    label="Negative prompt",
                    max_lines=1,
                    placeholder="Enter a negative prompt",
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    aspect_ratio = gr.Radio(
                        label="Aspect ratio(width:height)",
                        choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                        value="16:9",
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.0,
                        maximum=7.5,
                        step=0.1,
                        value=4.0,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=50,
                    )
                with gr.Row():
                    lora1 = gr.Textbox(label="LoRA 1 URL")
                    lora1_scale = gr.Slider(label="LoRA 1 Scale", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                with gr.Row():
                    lora2 = gr.Textbox(label="LoRA 2 URL")
                    lora2_scale = gr.Slider(label="LoRA 2 Scale", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                with gr.Row():
                    lora3 = gr.Textbox(label="LoRA 3 URL")
                    lora3_scale = gr.Slider(label="LoRA 3 Scale", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
            
            run_button.click(
                fn=infer_text_to_image,
                inputs=[
                    prompt,
                    negative_prompt,
                    seed,
                    randomize_seed,
                    aspect_ratio,
                    guidance_scale,
                    num_inference_steps,
                    lora1,
                    lora1_scale,
                    lora2,
                    lora2_scale,
                    lora3,
                    lora3_scale,
                ],
                outputs=[result, seed]
            )

            run_button_edit.click(
                fn=infer_image_edit,
                inputs=[
                    input_image,
                    input_image_2,
                    prompt_edit,
                    seed_edit,
                    randomize_seed_edit,
                    true_guidance_scale_edit,
                    num_inference_steps_edit,
                    rewrite_prompt_edit,
                    num_images_per_prompt_edit,
                    lora1_edit,
                    lora1_scale_edit,
                    lora2_edit,
                    lora2_scale_edit,
                    lora3_edit,
                    lora3_scale_edit,
                    use_lightning_edit,
                ],
                outputs=[result_edit, seed_edit]
            )

        with gr.TabItem("Image Editing"):
            gr.Markdown("## Image Editing")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image 1", show_label=False, type="pil")
                with gr.Column():
                    input_image_2 = gr.Image(label="Input Image 2", show_label=False, type="pil")
                result_edit = gr.Gallery(label="Result", show_label=False, type="pil")
            with gr.Row():
                prompt_edit = gr.Text(
                        label="Prompt",
                        show_label=False,
                        placeholder="describe the edit instruction",
                        container=False,
                )
                run_button_edit = gr.Button("Edit!", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                seed_edit = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )

                randomize_seed_edit = gr.Checkbox(label="Randomize seed", value=True)

                with gr.Row():

                    true_guidance_scale_edit = gr.Slider(
                        label="True guidance scale",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=4.0
                    )

                    num_inference_steps_edit = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=50,
                    )
                    
                    num_images_per_prompt_edit = gr.Slider(
                        label="Number of images per prompt",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                    )
                    
                    rewrite_prompt_edit = gr.Checkbox(label="Rewrite prompt", value=True)
                with gr.Row():
                    lora1_edit = gr.Textbox(label="LoRA 1 URL")
                    lora1_scale_edit = gr.Slider(label="LoRA 1 Scale", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                with gr.Row():
                    lora2_edit = gr.Textbox(label="LoRA 2 URL")
                    lora2_scale_edit = gr.Slider(label="LoRA 2 Scale", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                with gr.Row():
                    lora3_edit = gr.Textbox(label="LoRA 3 URL")
                    lora3_scale_edit = gr.Slider(label="LoRA 3 Scale", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
                with gr.Row():
                    use_lightning_edit = gr.Checkbox(label="Use Lightning Model", value=True)

if __name__ == "__main__":
    def cleanup():
        if gpu_manager:
            gpu_manager.stop()

    atexit.register(cleanup)

    def signal_handler(signum, frame):
        print(f"Received signal {signum}, cleaning up resources...")
        cleanup()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        demo.launch(server_name="0.0.0.0")
    except KeyboardInterrupt:
        print("Received interrupt signal, cleaning up resources...")
        cleanup()
    except Exception as e:
        print(f"Application exception: {e}")
        cleanup()
        raise
