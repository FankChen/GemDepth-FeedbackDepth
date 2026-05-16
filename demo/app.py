import gc
import os
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    spaces = _SpacesFallback()


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR
if not (PROJECT_DIR / "model").exists():
    PROJECT_DIR = ROOT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model.gemdepth import GemDepth  # noqa: E402
from model.utils.dc_utils import read_video_frames  # noqa: E402


MODEL_CONFIGS = {
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoint" / "gemdepth.pth"
MAX_DEMO_FRAMES = 300


def resolve_checkpoint() -> str:
    checkpoint_path = os.getenv("GEMDEPTH_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    repo_id = os.getenv("GEMDEPTH_MODEL_REPO_ID")
    filename = os.getenv("GEMDEPTH_MODEL_FILENAME", "gemdepth.pth")
    if repo_id:
        return hf_hub_download(repo_id=repo_id, filename=filename)

    raise gr.Error(
        "Model checkpoint not found. Upload the weights to checkpoint/gemdepth.pth, "
        "or set GEMDEPTH_MODEL_REPO_ID / GEMDEPTH_MODEL_FILENAME to download them from the Hugging Face Hub."
    )


def normalize_state_dict(checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise gr.Error("Invalid checkpoint format: could not read a state_dict.")

    return {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state_dict.items()
    }


@lru_cache(maxsize=3)
def load_model(encoder: str) -> GemDepth:
    checkpoint_path = resolve_checkpoint()
    model = GemDepth(**MODEL_CONFIGS[encoder])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(normalize_state_dict(checkpoint), strict=True)
    model.eval().requires_grad_(False)
    return model


def make_combined_frame(frame, depth, frame_w, frame_h, depth_h, d_min, d_max, grayscale=False):
    frame_img = np.asarray(frame)
    if frame_img.dtype != np.uint8:
        frame_img = (frame_img * 255 if frame_img.max() <= 1.0 else frame_img).astype(np.uint8)
    if frame_img.ndim == 2:
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2RGB)
    else:
        frame_img = frame_img[:, :, :3]
    frame_img = cv2.resize(frame_img, (frame_w, frame_h))

    depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_uint8 = (np.clip(depth_norm, 0, 1) * 255).astype(np.uint8)
    if grayscale:
        depth_img = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2RGB)
    else:
        depth_img = cv2.cvtColor(cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    depth_img = cv2.resize(depth_img, (frame_w, depth_h))

    return np.vstack([frame_img, depth_img])


def save_combined_video(frames, depths, output_path, fps=12, grayscale=False):
    if len(frames) == 0 or len(depths) == 0:
        raise gr.Error("The input video does not contain any processable frames.")

    min_len = min(len(frames), len(depths))
    frames, depths = frames[:min_len], depths[:min_len]
    frame = np.asarray(frames[0])
    frame_h, frame_w = frame.shape[:2]
    depth_h, depth_w = depths[0].shape[:2]
    scaled_depth_h = int(depth_h * (frame_w / depth_w))
    total_h = frame_h + scaled_depth_h

    if frame_w % 2:
        frame_w -= 1
    if total_h % 2:
        total_h -= 1

    all_depths = np.concatenate([depth.reshape(-1) for depth in depths])
    d_min, d_max = np.percentile(all_depths, 2), np.percentile(all_depths, 98)
    if d_max <= d_min:
        d_min, d_max = 0.0, 1.0

    fps = max(float(fps), 1.0)
    depth_vis_h = total_h - frame_h

    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "faststart"],
        )
        for frame, depth in zip(frames, depths):
            writer.append_data(
                make_combined_frame(frame, depth, frame_w, frame_h, depth_vis_h, d_min, d_max, grayscale)
            )
        writer.close()
        return
    except Exception as exc:
        print(f"imageio video writer unavailable, falling back to OpenCV: {exc}")

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, total_h))
    if not writer.isOpened():
        raise gr.Error("Could not create the output video. Please install imageio and imageio-ffmpeg.")

    for frame, depth in zip(frames, depths):
        combined = make_combined_frame(frame, depth, frame_w, frame_h, depth_vis_h, d_min, d_max, grayscale)
        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    writer.release()


def get_uploaded_path(video_file) -> str:
    if isinstance(video_file, str):
        return video_file
    if hasattr(video_file, "name"):
        return video_file.name
    if isinstance(video_file, dict) and "path" in video_file:
        return video_file["path"]
    raise gr.Error("Could not read the uploaded video file.")


def copy_input_video(video_file) -> str:
    video_path = get_uploaded_path(video_file)
    suffix = Path(video_path).suffix or ".mp4"
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_input.close()
    shutil.copy(video_path, temp_input.name)
    return temp_input.name


@spaces.GPU(duration=180)
def run_demo(video_file, max_frames, target_fps, input_size, grayscale, fp32):
    if video_file is None:
        raise gr.Error("Please upload a video first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_length = int(min(max_frames, MAX_DEMO_FRAMES))
    requested_fps = -1 if int(target_fps) <= 0 else int(target_fps)
    input_size = int(input_size)

    temp_video = copy_input_video(video_file)
    output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    output_depth = tempfile.NamedTemporaryFile(delete=False, suffix=".npz").name

    model = load_model("vitl")
    model.to(device)

    try:
        frames, fps = read_video_frames(
            temp_video,
            process_length=process_length,
            target_fps=requested_fps,
            max_res=1280,
        )
        depths, fps = model.infer_video_depth(
            frames,
            fps,
            input_size=input_size,
            device=device,
            fp32=bool(fp32 or device == "cpu"),
        )
        save_combined_video(frames, depths, output_video, fps=fps, grayscale=grayscale)
        np.savez_compressed(output_depth, depth=depths.astype(np.float32), fps=np.array(fps))
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if os.path.exists(temp_video):
            os.remove(temp_video)

    return output_video, output_video, output_depth


with gr.Blocks(title="GemDepth Demo") as demo:
    gr.Markdown(
        """
        # GemDepth Video Depth Demo

        Upload a short video file to generate a side-by-side source/depth visualization.
        The predicted depth array is also available as a compressed `.npz` file.
        """
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="Input video file",
                file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"],
                type="filepath",
            )
            gr.Markdown("**Encoder:** `vitl`")
            max_frames_input = gr.Slider(
                minimum=16,
                maximum=MAX_DEMO_FRAMES,
                step=1,
                value=48,
                label="Max frames",
            )
            target_fps_input = gr.Slider(
                minimum=0,
                maximum=24,
                step=1,
                value=8,
                label="Target FPS (0 = original FPS)",
            )
            input_size_input = gr.Slider(
                minimum=280,
                maximum=518,
                step=14,
                value=420,
                label="Inference size",
            )
            grayscale_input = gr.Checkbox(value=False, label="Use grayscale depth")
            fp32_input = gr.Checkbox(value=False, label="Force FP32 inference")
            run_button = gr.Button("Run GemDepth", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Combined result", format="mp4")
            result_file_output = gr.File(label="Result MP4")
            depth_output = gr.File(label="Depth NPZ")

    run_button.click(
        fn=run_demo,
        inputs=[
            video_input,
            max_frames_input,
            target_fps_input,
            input_size_input,
            grayscale_input,
            fp32_input,
        ],
        outputs=[video_output, result_file_output, depth_output],
    )


if __name__ == "__main__":
    demo.queue(max_size=4).launch()
