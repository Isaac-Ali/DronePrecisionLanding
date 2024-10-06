import cv2
import numpy as np
from dronekit import connect
import time
from picamera2 import Picamera2
from scipy.spatial.transform import Rotation as R
import threading

SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 921600
FRAME_RATE = 60.0
MARKER_SIZE = 0.1  # Real-world size of the ArUco marker in meters
FOV_X = 62.2
FOV_Y = 48.8

# Connect to the vehicle using DroneKit
vehicle = connect(SERIAL_PORT, baud=BAUD_RATE, wait_ready=True)
print("Connected to Pixhawk")

# Camera parameters
scalar = 0.5
frame_width, frame_height = int(3280 * scalar), int(2464 * scalar)
focal_length = frame_width / (2 * np.tan(FOV_X * (np.pi / 180) / 2))

# Precompute camera matrix and distortion coefficients
camera_matrix = np.array([[focal_length, 0, frame_width / 2],
                          [0, focal_length, frame_height / 2],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# Initialize Picamera2 for ArUco detection
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format": 'RGB888', "size": (frame_width, frame_height)})
picam2.configure(config)
picam2.set_controls({"FrameRate": FRAME_RATE})
picam2.start()

# Load the ArUco dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()
parameters.useAruco3Detection = 1

# Shared variables for frame capture
latest_frame = None
frame_lock = threading.Lock()
frame_ready = threading.Event()

def capture_frames():
    global latest_frame
    while True:
        frame = picam2.capture_array()  # Capture the frame as a NumPy array
        with frame_lock:
            latest_frame = frame  # Update the latest frame
            frame_ready.set()  # Signal that a new frame is ready

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True  # Daemonize thread
capture_thread.start()

def send_landing_target_with_aruco(angle_x, angle_y, distance, x, y, z, q):
    timestamp = int(time.time() * 1e6)
    msg = vehicle.message_factory.landing_target_encode(
        timestamp,  # Timestamp in microseconds
        0,  # target_num, not used
        0,  # frame, MAV_FRAME_BODY_NED
        angle_x,  # angle_x
        angle_y,  # angle_y
        distance,  # distance
        MARKER_SIZE,  # size_x
        MARKER_SIZE,  # size_y
        x,  # x
        y,  # y
        z,  # z
        q,  # q (quaternion)
        0,  # Fiducial marker
        True # Position Valid
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()
    print(f"""
    Sending LANDING_TARGET:
        angle_x = {angle_x}
        angle_y = {angle_y}
        distance = {distance}
        x = {x}
        y = {y}
        z = {z}
        q = {q}
    """)

def calculate_marker_pose(frame):
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None and len(corners) > 0:
        center_x, center_y = np.mean(corners[0][0], axis=0).astype(int)  # Center of the marker in pixels        
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
        estimated_marker_altitude = tvecs[0][0][2]  # Altitude (z-axis) in meters
        distance_to_marker = np.linalg.norm(tvecs[0][0])  # Total distance to the marker in meters

        # Convert rotation vector to quaternion using scipy
        rotation = R.from_rotvec(rvecs[0][0])  # From rotation vector
        q = rotation.as_quat()  # Convert to quaternion (x, y, z, w)

        # Rearrange to MAVLink's quaternion format (w, x, y, z)
        q = [q[3], q[0], q[1], q[2]]

        return estimated_marker_altitude, distance_to_marker, center_x, center_y, q
    return None, None, None, None, None

def pixel_to_angle(pixel_x, pixel_y, frame_width, frame_height, fov_x, fov_y):
    angle_x = (pixel_x - frame_width / 2) * (fov_x / frame_width)
    angle_y = (pixel_y - frame_height / 2) * (fov_y / frame_height)
    return angle_x, angle_y

# Main loop
while True:
    frame_ready.wait()  # Wait for a frame to be ready
    with frame_lock:
        frame = latest_frame  # Get the latest frame
        frame_ready.clear()  # Reset the event until the next frame is ready

        # Calculate marker pose
        estimated_marker_altitude, distance_to_marker, center_x, center_y, q = calculate_marker_pose(frame)

        if estimated_marker_altitude is not None:
            # Convert pixel coordinates to angles
            angle_x, angle_y = pixel_to_angle(center_x, center_y, frame_width, frame_height, FOV_X, FOV_Y)

            # Send LANDING_TARGET MAVLink message with ArUco detection data
            send_landing_target_with_aruco(angle_x, angle_y, distance_to_marker, center_x, center_y, estimated_marker_altitude, q)


