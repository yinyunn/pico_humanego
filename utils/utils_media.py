import os
import cv2
from tqdm import tqdm
import subprocess
import imageio


def add_audio_to_video(video_path_no_audio, audio_path, save_path):
    if not audio_path or not os.path.exists(audio_path):
        print(f"Audio file not found at {audio_path}, skipping merge for {os.path.basename(save_path)}.")
        os.rename(video_path_no_audio, save_path)
        return

    print(f"Adding audio to {os.path.basename(save_path)}...")
    command = [
        'ffmpeg', '-i', video_path_no_audio, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', save_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.remove(video_path_no_audio)
        print(f"  -> Successfully created video with audio: {os.path.basename(save_path)}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  -> ERROR: FFmpeg failed to add audio. The video will be silent.")
        if isinstance(e, FileNotFoundError):
            print("  -> FFmpeg command not found. Please make sure FFmpeg is installed and in your system's PATH.")
        else:
            print(f"  -> FFmpeg stderr: {e.stderr.decode()}")
        os.rename(video_path_no_audio, save_path)

def create_video_from_frames(frames, save_path, fps=10, export_gif=True, ratio=10):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not frames: 
        print("Warning: frames is empty!")
        return

    first_frame = frames[0]
    if len(first_frame.shape) == 2:
        h, w = first_frame.shape
        is_grayscale = True
    else:
        h, w, _ = first_frame.shape
        is_grayscale = False
    
    size = (w, h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, size)
    
    if export_gif:
        gif_frames = []

    print(f"Generating Video and GIF (Resolution={w}x{h}, FPS={fps})...")
    
    i = 0
    for frame in tqdm(frames):
        i+=1
        if is_grayscale:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
        
        if export_gif:
            if i%ratio == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                gif_frames.append(frame_rgb)

    out.release()
    print(f"Video Saved to: {save_path}")
    if export_gif and gif_frames:
        output_gif_path = save_path.replace('.mp4', '.gif')
        print(f"Gif Saved to: {output_gif_path}")
        try:
            imageio.mimsave(output_gif_path, gif_frames, fps=fps//ratio, loop=0)
        except Exception as e:
            print(f"Failed: {e}")
