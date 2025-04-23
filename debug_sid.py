import os
import time
import torch
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger
import wandb # Import wandb directly for potential debug flags

# Optional: Set WANDB_DEBUG environment variable for verbose logging
# os.environ['WANDB_DEBUG'] = 'true'

def main():
    # --- Configuration ---
    wandb_project = "Wan-SiD-Distillation" # <--- Replace with your project name
    wandb_entity = "SiD_pawan" # <--- Replace with your entity name
    num_steps_to_log = 20 # Log for 20 steps
    log_interval_seconds = 1 # Log every 1 second

    # --- Setup Wandb Logger ---
    # Initialize the logger. Fabric will handle the distributed setup.
    logger = WandbLogger(
        project=wandb_project,
        entity=wandb_entity,
        name="minimal-fabric-test-run",
        group="minimal_tests",
        # Setting reinit=True can sometimes help in complex setups,
        # but shouldn't be necessary for a minimal test.
        # reinit=True,
        log_model=False # Don't log models for this test
    )

    # --- Setup Fabric ---
    # Use DDP strategy with multiple devices (adjust devices as needed)
    fabric = Fabric(
        loggers=[logger],
        accelerator="cuda", # Or "cpu" if not using GPU
        devices=2, # <--- Adjust to the number of devices you want to test with
        strategy='ddp',
        precision="bf16-mixed" # Match your training script's precision
    )

    # --- Launch Fabric ---
    # This starts the distributed processes.
    fabric.launch()

    # --- Logging Loop (only on global rank 0) ---
    if fabric.is_global_zero:
        print(f"[{fabric.global_rank}] Global rank 0 process started.")
        print(f"[{fabric.global_rank}] Logging to Wandb project: {wandb_project}, entity: {wandb_entity}")
        print(f"[{fabric.global_rank}] Starting logging loop for {num_steps_to_log} steps...")

        for step in range(num_steps_to_log):
            current_step = step + 1 # Start step count from 1
            print(f"[{fabric.global_rank}] Logging step {current_step}...")

            # Log a simple counter
            log_dict = {"minimal_step_counter": current_step}
            fabric.log()
            fabric.log_dict(log_dict) # Use fabric.log for Fabric's logger integration

            print(f"[{fabric.global_rank}] Logged step {current_step}. Sleeping...")
            time.sleep(log_interval_seconds) # Pause briefly

        print(f"[{fabric.global_rank}] Logging loop finished.")

    # --- Barrier to ensure all processes finish ---
    fabric.barrier()

    # --- Finish Wandb run (optional, Fabric might handle this) ---
    # This ensures all buffered data is sent.
    if fabric.is_global_zero:
        print(f"[{fabric.global_rank}] Global rank 0 finishing Wandb run.")
        wandb.finish() # Use direct wandb.finish() on rank 0

    print(f"[{fabric.global_rank}] Process finished.")

if __name__ == "__main__":
    # Ensure script is run correctly with torchrun or similar launcher for DDP
    # Example: torchrun --nproc_per_node=2 your_script_name.py
    main()
