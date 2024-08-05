from pipeline_rf import RectifiedFlowPipeline
import torch
import argparse
import os 
from tqdm import tqdm


def main(args):
    os.makedirs(args.image_dir, exist_ok=True)
    pipe = RectifiedFlowPipeline.from_pretrained("XCLIU/2_rectified_flow_from_sd_1_5", torch_dtype=torch.float16) 
    ### switch to torch.float32 for higher quality
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe.to(device)  ### if GPU is not available, comment this line

    prompts = [line.strip("\n") for line in open(args.caption_file)][:args.num_prompts]

    timesteps = [float(i) * 1000 for i in args.timesteps]
    if len(timesteps) == 0:
        timesteps = [(1. - i/args.steps) * 1000. for i in range(args.steps)]

    count = 0
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        prompt_batch = prompts[i : i + args.batch_size]
        images = pipe(prompt=prompt_batch, 
                guidance_scale=1.5,
                timesteps=timesteps).images 
        for i, image in enumerate(images):
            image.save(os.path.join(args.image_dir, f"{count + i:06d}.png"))
        count += len(prompt_batch)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sample arguments')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--caption_file', type=str)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_prompts', type=int, default=10000)
    parser.add_argument('--timesteps', nargs='+')
    parser.add_argument('--steps', type=int)

    args = parser.parse_args()

    main(args)
