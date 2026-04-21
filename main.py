import os
import sys
import subprocess
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

def get_ffmpeg_path():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    
    local_ffmpeg = Path(__file__).parent / "ffmpeg_bin" / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    
    return None

def install_ffmpeg():
    print("[!] FFmpeg not found. Attempting to install based on your OS...")
    
    if sys.platform.startswith("linux"):
        if shutil.which("pacman"):
            print("[+] Detected Arch-based system. Running pacman...")
            subprocess.run(["sudo", "pacman", "-Sy", "ffmpeg", "--noconfirm"], check=True)
        elif shutil.which("apt"):
            print("[+] Detected Debian/Ubuntu-based system. Running apt...")
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
        else:
            print("[-] Unsupported Linux package manager. Please install ffmpeg manually.")
            sys.exit(1)
            
    elif sys.platform == "darwin":
        if shutil.which("brew"):
            print("[+] Detected macOS. Running Homebrew...")
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
        else:
            print("[-] Homebrew not found. Please install Homebrew or install ffmpeg manually.")
            sys.exit(1)
            
    elif sys.platform == "win32":
        print("[+] Detected Windows. Downloading FFmpeg static build...")
        url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        zip_path = Path("ffmpeg_temp.zip")
        extract_dir = Path("ffmpeg_bin")
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith("ffmpeg.exe"):
                        file_info.filename = "ffmpeg.exe"
                        zip_ref.extract(file_info, extract_dir)
            
            zip_path.unlink()
            print(f"[+] FFmpeg downloaded and extracted locally to {extract_dir.absolute()}")
        except Exception as e:
            print(f"[-] Failed to download FFmpeg: {e}")
            sys.exit(1)
    else:
        print("[-] Unsupported operating system.")
        sys.exit(1)

def extract_frames(directory_path):
    target_dir = Path(directory_path)
    
    if not target_dir.is_dir():
        print(f"[-] Error: '{directory_path}' is not a valid directory.")
        sys.exit(1)

    supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_formats]

    if not video_files:
        print(f"[-] No supported video files found in '{directory_path}'.")
        return

    ffmpeg_cmd = get_ffmpeg_path()
    if not ffmpeg_cmd:
        print("[-] Critical Error: FFmpeg could not be found or installed.")
        sys.exit(1)

    for video_file in video_files:
        output_folder = target_dir / video_file.stem
        output_folder.mkdir(parents=True, exist_ok=True)
        
        output_pattern = str(output_folder / "frame_%04d.jpg")
        
        print(f"\n[+] Processing: {video_file.name}")
        print(f"    Extracting to: {output_folder.name}/")

        command = [
            ffmpeg_cmd,
            "-y",
            "-i", str(video_file),
            "-vf", "fps=2",
            "-qscale:v", "2",
            output_pattern
        ]

        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            print("    Success!")
        except subprocess.CalledProcessError as e:
            print(f"[-] Error processing {video_file.name}:")
            print(e.stderr.decode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 2 frames per second from all videos in a target directory.")
    parser.add_argument("directory", help="Path to the directory containing video files")
    args = parser.parse_args()

    if not get_ffmpeg_path():
        install_ffmpeg()

    extract_frames(args.directory)