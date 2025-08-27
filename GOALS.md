**Project GOALS**

  - Take the github project, https://github.com/QwenLM/Qwen-Image, and
    make a gradio app that will run inferene with the qwen image and
    qwen image edit models.  This will be containerized to run on RunPod
    platform.
  - There is an example, that does not work well,
    https://huggingface.co/spaces/Qwen/Qwen-Image-Edit .  This is
    why a working and and more feature rich version is needed.
  - https://github.com/sruckh/qwenimage-runpod is the github repository for the Runpod version of this project.  Always use SSH to talk to github.
  - Use context7 to get the latest documentation.  For example how to build docker containers for Runpod.
  - Use fetch to access external websites
  - Ask serena for recent memories to get context about the project.
  - 'docker-compose' has been deprecated, use 'docker compose instead'
  - for complext tasks let claude flow launch sub-agents
  - https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files is where Qwen Image models can be downloaded.  
  - These are the models needed for Qwen Image
    - https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors
	  - https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors
	  - https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
  - This is the hugginface page for Qwen Image Edit
    - https://huggingface.co/Qwen/Qwen-Image-Edit

  - see https://huggingface.co/flymy-ai/qwen-image-edit-inscene-lora for how to include LoRAs.  Gradio interface should be able to Load Up to 3 LoRAs and be able to proide a scale for each LoRA 0-2 with two decimal places.
  - https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.1.safetensors should be used to speed up edits.  see https://github.com/ModelTC/Qwen-Image-Lightning/ on how to use model.
  - Use context7 find out how to build docker containers for Runpod.  Also use fetch to see https://github.com/runpod/containers.
  - Outside of the base image and maybe a startup (boot strap script), nothing else should be in the base container.  Everything will be installed at runtime (programs, Dependencies, AI Models, etc)
  - Everything should be installed in the /workspace filesystem.
  - HF_TOKEN and HF_HOME will be environmental variables that will be configured.  HF_TOKEN is the user's hugging face token for access huggingface models.  HF_HOME is the directory where hugging face models will be downloaded.
  - Look at the src/examples folders of https://github.com/QwenLM/Qwen-Image to see how inference calls are made.
  - The gradio app, should support the functionality of the inference.
  - It is desired to integrate https://github.com/ClownsharkBatwing/RES4LYF
  - It is desired to support multi image editing.  For example an image of a person and and image of a purse.  You can prompt qwen image edit with something like "person is holding the purse" and inference will create an image using the two reference images.
  - Never build this docker container on the host.  It is designed for Runpod and will most likely fail trying to build on this host.
  - The github repo https://github.com/sruckh/qwenimage-runpod will be used for this project.  Use SSH to access this repository.
  - Create github action for building container and deploy to dockerhub.
  - The github secrets DOCKER_USERNAME and DOCKER_PASSWORD have been created for accessing the gemneye repository on dockerhub.
  - The correct flash_attn whl file can be installed from https://github.com/Dao-AILab/flash-attention/releases
  - github action needs to be configured to build and deploy container image to docker hub.
    - github secrets DOCKER_USERNAME and DOCKER_PASSWORD have been configured to be used for pushing images to the dockerhub repository gemneye/
  - If you ever have a question, let the user provide an approriate answer before you begin making changes.
