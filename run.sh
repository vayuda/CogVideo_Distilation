export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_sid.py
python wanpipeline.py --mode export --latent_fp checkpoints/start_video.pt --video_fp samples/test
python wanpipeline.py --mode export --latent_fp checkpoints/gen_latents_latest.pt --video_fp samples/epoch