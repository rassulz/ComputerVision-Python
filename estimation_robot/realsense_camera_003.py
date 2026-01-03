import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)  # Increase timeout
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("No frames received, restarting pipeline...")
                pipeline.stop()
                time.sleep(1)
                pipeline.start(config)
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap to depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Display images
            cv2.imshow('RealSense Depth', depth_colormap)
            cv2.imshow('RealSense Color', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except RuntimeError as e:
            print(f"Error: {e}. Restarting pipeline...")
            pipeline.stop()
            time.sleep(1)
            pipeline.start(config)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
