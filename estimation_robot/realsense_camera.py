import pyrealsense2 as rs
import numpy as np
import cv2

# Configure the depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# Enable depth stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the depth image to 8-bit per pixel and apply a colormap
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

        # Stack the color image and the depth colormap side-by-side
        images = np.hstack((color_image, depth_colormap))

        # Display the combined image
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit
            break
finally:
    # Stop streaming
    pipeline.stop()
