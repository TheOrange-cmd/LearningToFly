'''
This script is used to review a video and mark sections of the video that are considered "bad" or unusable.
The user can play/pause the video, mark the start and end of bad sections, and navigate through frames. The marked sections are saved in a JSON file for later reference.
Author: Daniel Rugge (2025) created with help of Claude Sonnet 3.5
'''

import cv2
import json
from pathlib import Path

def review_video(video_path, output_json):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad_sections = []
    current_start = None
    paused = False
    frame_number = 0

    print("Controls:")
    print("  p - Play/Pause")
    print("  s - Mark start of bad section")
    print("  e - Mark end of bad section")
    print("  ←/→ - Move frame backward/forward (when paused)")
    print("  ESC - Save and exit")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Create display frame
        display = frame.copy()
        status = "PAUSED" if paused else "PLAYING"
        color = (0, 255, 0) if not paused else (0, 0, 255)
        
        # Show current section markers
        cv2.putText(display, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Frame: {frame_number}/{total_frames}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if current_start is not None:
            cv2.putText(display, f"Marking: {current_start}-?", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show bad sections
        for start, end in bad_sections:
            if start <= frame_number <= end:
                cv2.rectangle(display, (0, 0), 
                            (display.shape[1], display.shape[0]),
                            (0, 0, 255), 5)

        cv2.imshow("Video Review", display)

        key = cv2.waitKey(0 if paused else 1) & 0xFF  # Get key code
        
        if key == 27:  # ESC - Save and exit
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            current_start = frame_number
        elif key == ord('e') and current_start is not None:
            bad_sections.append((min(current_start, frame_number),
                               max(current_start, frame_number)))
            current_start = None
        elif key == ord('n'):  # Left arrow (standard or numpad)
            if paused and frame_number > 0:
                frame_number = max(0, frame_number - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
        elif key == ord('m'):  # Right arrow (standard or numpad)
            if paused and frame_number < total_frames - 1:
                frame_number = min(total_frames - 1, frame_number + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

    # Save results as ranges
    with open(output_json, 'w') as f:
        json.dump({
            "video_path": str(video_path),
            "bad_sections": bad_sections
        }, f, indent=2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_dir = Path("C:\\Users\\danie\\Documents\\GitHub\\Autonomous_Flight\\comparison_videos")
    output_dir = Path("C:\\Users\\danie\\Documents\\GitHub\\Autonomous_Flight\\annotations")
    output_dir.mkdir(exist_ok=True)
    
    for video_path in video_dir.glob("*.mp4"):
        output_json = output_dir / f"{video_path.stem}_bad_frames.json"
        print(f"Reviewing: {video_path.name}")
        review_video(video_path, output_json)