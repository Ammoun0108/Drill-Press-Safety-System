import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
import time
import serial
import tkinter as tk

# --- Serial controller class to manage RTS signal ---
class SerialController:
 def __init__(self, port='/dev/ttyUSB0'):
 try:
 self.ser = serial.Serial(port)
 print(f"Serial port {port} opened.")
 except serial.SerialException as e:
 print(f"Could not open {port}: {e}")
 self.ser = None

 def set_rts(self, status: bool):
 if self.ser:
 self.ser.setRTS(status)
 print(f"RTS set to {'high' if status else 'low'}.")

 def close(self):
 if self.ser:
 self.ser.close()
 print("Serial port closed.")

# --- Calibration constants ---
image_depth = 12
PIXEL_TO_inches = 11.3 / 640 / 12
PX_CENTER, PY_CENTER = 320, 240
P_TOP = np.array([10, 0, 52])
P_BOTTOM = P_TOP
safeD = 5 # safety distance threshold

# --- Coordinate transformation helpers ---
def pixels_to_xy(polygon, depth_m):
 zin_in = depth_m * 39.37 # depth in inches
 polygon_out = polygon.copy()
 polygon_out[:, 0] = (polygon_out[:, 0] - PX_CENTER) * PIXEL_TO_inches * zin_in
 polygon_out[:, 1] = (polygon_out[:, 1] - PY_CENTER) * PIXEL_TO_inches * zin_in
 return polygon_out

def get_shapes(label, masks, classes, names):
 return [masks[i] for i, c in enumerate(classes) if names[c] == label]

def polygon_to_points(polygon, depth_frame, class_name):
 if polygon.size == 0:
 return None
 avg = np.mean(polygon, axis=0)
 d_m = depth_frame.get_distance(int(avg[0]), int(avg[1]))
 if d_m <= 0:
 return None
 pts_xy = pixels_to_xy(polygon, d_m)
 d_in = d_m * 39.37
 pts_xyz = np.hstack([pts_xy, d_in * np.ones((pts_xy.shape[0], 1))])
 return pts_xyz

def fit_sphere_world(points, transform_matrix):
 points_h = np.hstack([points, np.ones((points.shape[0], 1))])
 points_world_h = (transform_matrix @ points_h.T).T
 points_world = points_world_h[:, :3]
 sorted_pts = points_world[np.argsort(points_world[:, 2])]
 return sorted_pts

def global_points(points):
 sorted_pts = points[np.argsort(points[:, 2])]
 c1, c2 = sorted_pts[0], sorted_pts[-1]
 c1 = np.array([P_TOP[0], P_TOP[1], c1[2]])
 c2 = P_TOP
 return c1, c2, (c1 + c2) / 2

def global_points_drill(points):
 sorted_pts = points[np.argsort(points[:, 2])]
 c1, c2 = sorted_pts[0], sorted_pts[-1]
 c1 = np.array([P_BOTTOM[0], P_BOTTOM[1], c1[2]])
 c2 = P_BOTTOM
 return c1, c2, (c1 + c2) / 2

def sphere_to_cylinder_distance(hand_c, r_h, c1, c2, r_cyl):
 v = c2 - c1
 v_unit = v / np.linalg.norm(v)
 m = hand_c - c1
 d_perp = np.linalg.norm(np.cross(m, v_unit))
 pp = c1 + np.dot(m, v_unit) * v_unit
 d1 = np.linalg.norm(pp - c1)
 d2 = np.linalg.norm(pp - c2)
 lsegment = np.linalg.norm(v)
 within = (d1 < lsegment) and (d2 < lsegment)
 return d_perp if within else min(d1, d2)

def sphere_to_sphere_distance(c1, r1, c2, r2):
 return np.linalg.norm(c1 - c2) - (r1 + r2)

# Function to check for clamps, with dialog loop
def check_clamps(pipeline, align, model):
 global clamps_checked

 while True:
 frames = pipeline.wait_for_frames()
 aligned = align.process(frames)
 color_frame = aligned.get_color_frame()
 depth_frame = aligned.get_depth_frame()
 if not color_frame or not depth_frame:
 continue

 color_image = np.asanyarray(color_frame.get_data())
 results = model(color_image)[0]

 if results.masks is None:
 continue

 masks = results.masks.xy
 classes = results.boxes.cls.cpu().numpy().astype(int)
 names = results.names
 clamp_shapes = get_shapes("clamp", masks, classes, names)

 if len(clamp_shapes) >= 2:
 clamps_checked = True
 return

 # Not enough clamps: ask user what to do
 proceed = [0]

 def proceed_without():
 proceed[0] = 1
 root.destroy()

 def clamps_added():
 proceed[0] = 2
 root.destroy()

 root = tk.Tk()
 root.title("Clamps Confirmation")
 root.attributes("-fullscreen", True)

 canvas = tk.Canvas(root, bg="black")
 canvas.pack(fill=tk.BOTH, expand=True)

 label = tk.Label(root, text="Less than 2 clamps detected.\nWhat do you want to do?",
 font=("Helvetica", 24), bg="black", fg="white")
 label.place(relx=0.5, rely=0.4, anchor="center")

 yes_btn = tk.Button(root, text="Proceed Without Clamps", font=("Helvetica", 20),
 bg="green", fg="white", width=20, height=2, command=proceed_without)
 yes_btn.place(relx=0.3, rely=0.55, anchor="center")

 retry_btn = tk.Button(root, text="Clamps Added, Check Again", font=("Helvetica", 20),
 bg="blue", fg="white", width=20, height=2, command=clamps_added)
 retry_btn.place(relx=0.7, rely=0.55, anchor="center")

 root.mainloop()

 if proceed[0] == 1:
 clamps_checked = True
 return

# --- Setup serial and robot state ---
serial_ctrl = SerialController('/dev/ttyUSB0')
serial_ctrl.set_rts(True) # robot OFF initially

# --- Camera calibration matrix: user prompt restored ---
print("Initial Conditions: ")
print("camera X rotation angle (degrees): -120")
print("camera Z rotation angle (degrees): 35")
print("camera X offset (inches): 23")
print("camera Y offset (inches): -26")
print("camera Z offset (inches): 57")
resp = input("Have you changed the camera position? (y/n): ").strip().lower()

if resp == 'y':
 angle_x = float(input("Enter camera X rotation angle (degrees): "))
 angle_z = float(input("Enter camera Z rotation angle (degrees): "))
 tx1 = float(input("Enter camera X offset (inches): "))
 ty1 = float(input("Enter camera Y offset (inches): "))
 tz1 = float(input("Enter camera Z offset (inches): "))
else:
 angle_x, angle_z = -120, 35
 tx1, ty1, tz1 = 23, -26, 57

r1 = R.from_euler('x', angle_x, degrees=True).as_matrix()
r2 = R.from_euler('z', angle_z, degrees=True).as_matrix()
rotation = np.dot(r2, r1)
transform_matrix = np.eye(4)
transform_matrix[:3, :3] = rotation
transform_matrix[:3, 3] = [tx1, ty1, tz1]

# --- RealSense pipeline and YOLO model setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
model = YOLO('bestdrill.pt')

clamps_checked = False
check_clamps(pipeline, align, model)
serial_ctrl.set_rts(False) # robot ON

# --- Main loop ---
try:
 while True:
 T=time.time()
 print(T)
 frames = pipeline.wait_for_frames()
 aligned = align.process(frames)
 color_frame = aligned.get_color_frame()
 depth_frame = aligned.get_depth_frame()
 if not color_frame or not depth_frame:
 continue

 color_image = np.asanyarray(color_frame.get_data())
 results = model(color_image)[0]

 if results.masks is None:
 if cv2.waitKey(1) & 0xFF == ord("q"):
 break
 continue

 annotated_image = results.plot()
 cv2.imshow("YOLO Detection", annotated_image)

 masks = results.masks.xy
 classes = results.boxes.cls.cpu().numpy().astype(int)
 names = results.names

 hand_shapes = get_shapes("hand", masks, classes, names)
 drill_shapes = get_shapes("drill", masks, classes, names)
 chuck_shapes = get_shapes("chuck", masks, classes, names)

 robot_should_stop = False

 for hand_poly in hand_shapes:
 hand_pts_c = polygon_to_points(hand_poly, depth_frame, 'hand')
 if hand_pts_c is None:
 continue
 hand_pts_w = fit_sphere_world(hand_pts_c, transform_matrix)
 hand_c = np.mean(hand_pts_w, axis=0)
 r_h = np.max(np.linalg.norm(hand_pts_w - hand_c, axis=1))

 for chuck_poly in chuck_shapes:
 chuck_pts = polygon_to_points(chuck_poly, depth_frame, 'chuck')
 if chuck_pts is None:
 continue
 chuck_pts = fit_sphere_world(chuck_pts, transform_matrix)
 c1, c2, chuck_c_corr = global_points(chuck_pts)
 r_sphere = np.linalg.norm(c2 - c1) / 2
 clear_sphere = sphere_to_sphere_distance(hand_c, r_h, chuck_c_corr, r_sphere)
 if clear_sphere < safeD:
 clear_cyl = sphere_to_cylinder_distance(hand_c, r_h, c1, c2, 1.115)
 if clear_cyl < safeD:
 serial_ctrl.set_rts(True)
 robot_should_stop = True
 break
 if robot_should_stop:
 break

 for drill_poly in drill_shapes:
 drill_pts = polygon_to_points(drill_poly, depth_frame, 'drill')
 if drill_pts is None:
 continue
 drill_pts = fit_sphere_world(drill_pts, transform_matrix)
 c1, c2, drill_c_corr = global_points_drill(drill_pts)
 drill_r = (c2[2] - c1[2]) / 2
 clear_sphere = sphere_to_sphere_distance(hand_c, r_h, drill_c_corr, drill_r)
 if clear_sphere < safeD:
 clear_cyl = sphere_to_cylinder_distance(hand_c, r_h, c1, c2, 1.0)
 if clear_cyl < safeD:
 serial_ctrl.set_rts(True)
 robot_should_stop = True
 break
 if robot_should_stop:
 break

 if robot_should_stop:
 root = tk.Tk()
 root.title("Safety Shutdown")
 root.attributes("-fullscreen", True)

 canvas = tk.Canvas(root, bg="black")
 canvas.pack(fill=tk.BOTH, expand=True)

 label = tk.Label(root, text="SAFETY SHUTDOWN - PRESS RED BUTTON ON DRILL PRESS",
 font=("Helvetica", 24), bg="black", fg="white")
 label.place(relx=0.5, rely=0.4, anchor="center")

 confirm_btn = tk.Button(root, text="CONFIRMED RED BUTTON PRESSED", font=("Helvetica", 20),
 bg="green", fg="white", width=40, height=4,
 command=lambda: root.destroy())
 confirm_btn.place(relx=0.5, rely=0.55, anchor="center")

 root.mainloop()

 serial_ctrl.set_rts(False)
 check_clamps(pipeline, align, model)

 cv2.imshow("RGB", color_image)
 if cv2.waitKey(1) == ord("q"):
 break

finally:
 pipeline.stop()
 cv2.destroyAllWindows()
 serial_ctrl.close()

 
