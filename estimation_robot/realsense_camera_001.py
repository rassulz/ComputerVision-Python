import cv2
import torch
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# === Load YOLO Model ===
MODEL_PATH = "cubeCamera.pt"  # Ensure your trained model is in the same directory
model = YOLO(MODEL_PATH).to('cuda')  # Run on GPU if available

# === Configure Intel RealSense Camera ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth Stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color Stream

# Start streaming
pipeline.start(config)
align = rs.align(rs.stream.color)  # Align depth and color frames

# Get camera intrinsic parameters (for converting depth to 3D points)
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # Get depth scale for converting depth units
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

def get_3d_coordinates(px, py, depth_frame):
    """
    Converts 2D pixel coordinates (px, py) to 3D real-world coordinates (X, Y, Z).
    """
    depth = depth_frame.get_distance(px, py)  # Get depth in meters
    if depth > 0:
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], depth)
        return point[0], point[1], point[2]  # (X, Y, Z) coordinates in meters
    return None, None, None  # If invalid depth

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)  # Align depth and color frames
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Convert color frame to numpy array
    color_image = np.asanyarray(color_frame.get_data())

    # === Run YOLO Detection ===
    results = model(color_image, conf=0.6, iou=0.4)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            label = result.names[int(box.cls[0])]  # Class label

            # Compute center of detected object
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Get 3D world coordinates (X, Y, Z)
            x_real, y_real, z_real = get_3d_coordinates(center_x, center_y, depth_frame)
            
            # Check if the coordinates are valid
            if x_real is None or y_real is None or z_real is None:
                continue

            # Print coordinates in the terminal
            print(f"Object: {label} | Confidence: {conf:.2f}")
            print(f"Bounding Box: x1={x1}, \n y1={y1}, \n x2={x2},\n y2={y2}")
            print(f"3D Coordinates: X={x_real:.3f}m,\n Y={y_real:.3f}m,\n Z={z_real:.3f}m\n")

            # === Draw bounding box and coordinates ===
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f} | X:{x_real:.2f}m Y:{y_real:.2f}m Z:{z_real:.2f}m"
            cv2.putText(color_image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLO + RealSense 3D Detection", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the RealSense pipeline and close OpenCV window
pipeline.stop()
cv2.destroyAllWindows()
