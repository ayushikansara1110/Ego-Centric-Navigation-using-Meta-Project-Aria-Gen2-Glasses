import aria.sdk_gen2 as sdk_gen2
from aria.sdk_gen2 import (
    DeviceClient, DeviceTarget, HttpStreamingConfig, StreamingInterface,
    StreamDataInterface, AriaGen2HttpServer, HttpServerConfig
)
import aria.oss_data_converter as data_converter

DEVICE_IP = "192.168.225.7"  # not used for USB, but DeviceTarget can stay empty for auto-detect over USB

# --- 1. Connect ---
client = DeviceClient()
device = client.connect(DeviceTarget())  # USB auto-detect

# --- 2. Set up data converter (calibration set once we receive it) ---
converter = data_converter.OssDataConverter(enable_image_decoding=False)
calibration_ready = False

def on_calib(device_calibration):
    global calibration_ready
    converter.set_calibration(device_calibration.get_json())  # method name may vary — check calib object's serialization method
    calibration_ready = True
    print("Calibration received, VIO decoding enabled")

def on_vio(frontend_output):
    if not calibration_ready:
        return
    pose = frontend_output.transform_odometry_bodyimu
    t = pose.translation()
    r = pose.rotation().log()
    print(f"VIO pose  t={t}  r={r}")

def on_vio_high_freq(vio_poses):
    if not calibration_ready:
        return
    for p in vio_poses:
        t = p.transform_odometry_device.translation()
        print(f"HF VIO  t={t}")

# --- 3. Register callbacks ---
stream_interface = StreamDataInterface(enable_image_decoding=False, enable_raw_stream=False)
stream_interface.register_device_calib_callback(on_calib)
stream_interface.register_vio_callback(on_vio)
stream_interface.register_vio_high_frequency_callback(on_vio_high_freq)  # optional, drop if not needed

# Optional: bump queue size if you see drops
stream_interface.set_vio_queue_size(30)

# --- 4. Configure streaming over USB ---
config = HttpStreamingConfig()
config.profile_name = "profile_name_with_vio"  # replace after checking device.device_profiles()
config.streaming_interface = StreamingInterface.USB_NCM  # use USB_RNDIS on Windows
device.set_streaming_config(config)

# --- 5. Start HTTP server to receive the stream, then start streaming ---
server_config = HttpServerConfig()
server_config.address = "0.0.0.0"
server_config.port = 8080
server = AriaGen2HttpServer(server_config, stream_interface)

device.install_streaming_certs()
device.start_streaming()

try:
    server.join()
except KeyboardInterrupt:
    device.stop_streaming()
    server.stop()
    client.disconnect(device)