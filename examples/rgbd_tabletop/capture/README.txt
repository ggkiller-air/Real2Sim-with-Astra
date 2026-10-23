One D455 raw depth frame and one RGB frame from each currently connected camera.
image_external_rgb.jpg: D455; cropped/resized as specified in capture_metadata.json.
image_external_2_rgb.jpg: D435; native 640x480 RGB.
image_wrist_rgb.jpg: ZED Mini left eye; resized and padded to 640x480.
d455_depth_raw.png: uint16 native depth; multiply by depth_scale_m for metres; zero is invalid.
d455_depth_m.npy: metric depth, with zero still invalid.
calibration_bundle_20261016.json: original historical bundle, copied without modification.
d455_calibration.json: extracted D455 calibration plus current factory depth intrinsics/extrinsics.
capture_metadata.json: receive timestamps, camera identity, preprocessing and freshness.
RGB and depth were requested concurrently but are not hardware synchronized or pixel aligned.
The bundle contains historical D435/ZED calibration too; this capture does not validate current mounting.
