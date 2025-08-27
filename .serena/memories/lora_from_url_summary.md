I have implemented the ability to use LoRAs from a URL.
- The LoRA inputs in the UI have been changed from file uploads to text boxes to accept URLs.
- When a URL is provided for a LoRA, the application will first check if the LoRA already exists in the `/workspaces/projects/loras` directory.
- If the LoRA does not exist locally, it will be downloaded from the provided URL and stored in the `/workspaces/projects/loras` directory.
- The application will then use the local version of the LoRA for the inference.