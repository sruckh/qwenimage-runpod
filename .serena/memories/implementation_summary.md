I have completed the implementation of the goals outlined in GOALS.md.

Here is a summary of the work I have done:
- **Image Editing:** I have implemented the image editing functionality in the Gradio app, including a user interface for uploading images and providing prompts.
- **LoRA Support:** I have added support for up to 3 LoRAs for both text-to-image and image editing, with sliders to control their scales.
- **Lightning Model:** I have integrated the Qwen-Image-Lightning model to speed up image edits, with a checkbox to enable or disable it.
- **Multi-Image Editing:** I have added support for multi-image editing, allowing users to combine two images with a text prompt.
- **Dockerfile:** I have created a `Dockerfile` to containerize the application for the RunPod platform.
- **GitHub Action:** I have created a GitHub Action to automatically build and push the Docker image to Docker Hub.
- **Flash Attention:** I have added the installation of the correct `flash_attn` wheel to the `Dockerfile` for performance optimization.

The following goals from `GOALS.md` are instructions, constraints, or require user input, and I have followed them throughout the implementation:
- Not building the Docker container on the host.
- Using SSH to access the GitHub repository.
- Using the provided Docker Hub secrets.
- Asking for user input when needed.

I believe the project is now fully implemented according to the specified goals.