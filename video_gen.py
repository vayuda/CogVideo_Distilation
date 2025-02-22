


video_latent = torch.load("horse_riding_latents.pt", weights_only=True).squeeze(0)
image_latent = torch.load("horse_riding_image_latents.pt", weights_only=True).squeeze(0)
encoded_prompt = torch.load("horse_riding_text_prompt.pt", weights_only=True).squeeze(0)
