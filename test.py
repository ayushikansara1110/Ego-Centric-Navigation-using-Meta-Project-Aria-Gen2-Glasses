import time

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver

from projectaria_tools.core.sensor_data import FrontendOutput


# ============================================================
# GLOBALS
# ============================================================

vio_count = 0
start_time = time.time()


# ============================================================
# VIO CALLBACK
# ============================================================

def vio_callback(vio_data: FrontendOutput):
    global vio_count

    vio_count += 1

    timestamp_ns = vio_data.capture_timestamp_ns

    T_odom_bodyimu = vio_data.transform_odometry_bodyimu

    rotation = T_odom_bodyimu.rotation().log()
    translation = T_odom_bodyimu.translation()

    elapsed = time.time() - start_time

    print(
        f"[VIO #{vio_count:05d}] "
        f"t={elapsed:6.2f}s | "
        f"timestamp={timestamp_ns} | "
        f"position={translation} | "
        f"rotation={rotation}"
    )


# ============================================================
# CONNECT
# ============================================================

print("Connecting to Aria Gen2...")

device_client = sdk_gen2.DeviceClient()

client_config = sdk_gen2.DeviceClientConfig()
device_client.set_client_config(client_config)

device = device_client.connect()

print("Connected.")


# ============================================================
# STREAM CONFIG
# ============================================================

streaming_config = sdk_gen2.HttpStreamingConfig()

# Official MP streaming profile:
# includes VIO / eye gaze / hand tracking
streaming_config.profile_name = "mp_streaming_demo"

# IMPORTANT:
# Do NOT manually set USB_NCM for this test.
# Let the SDK use its default USB streaming configuration.

device.set_streaming_config(streaming_config)

print("Profile configured: mp_streaming_demo")


# ============================================================
# START DEVICE STREAMING FIRST
# ============================================================

stream_started = False
stream_receiver = None

try:

    print("Starting streaming on glasses...")

    device.start_streaming()

    stream_started = True

    print("Device streaming started.")


    # ========================================================
    # NOW CREATE RECEIVER
    # ========================================================

    print("Creating StreamReceiver...")

    server_config = sdk_gen2.HttpServerConfig()

    server_config.address = "0.0.0.0"
    server_config.port = 6768

    # Use default constructor just like official example
    stream_receiver = receiver.StreamReceiver()

    stream_receiver.set_server_config(server_config)


    # ========================================================
    # REGISTER VIO
    # ========================================================

    stream_receiver.register_vio_callback(vio_callback)

    print("VIO callback registered.")


    # ========================================================
    # START SERVER
    # ========================================================

    print("Starting receiver server on port 6768...")

    stream_receiver.start_server()

    print("Receiver started.")


    # ========================================================
    # TEST
    # ========================================================

    print()
    print("=" * 70)
    print("LIVE GEN2 VIO TEST")
    print("=" * 70)
    print()
    print("Waiting for VIO...")
    print("Press Ctrl+C to stop.")
    print()
    print("=" * 70)


    last_count = 0

    while True:

        time.sleep(2)

        current = vio_count
        received = current - last_count

        print(
            f"[STATUS] "
            f"VIO total={current} | "
            f"last 2 sec={received}"
        )

        last_count = current


except KeyboardInterrupt:

    print("\nStopping...")


except Exception as e:

    print(
        f"\nERROR: {type(e).__name__}: {e}"
    )


finally:

    print("\nCleaning up...")


    # --------------------------------------------------------
    # STOP RECEIVER
    # --------------------------------------------------------

    if stream_receiver is not None:

        try:

            stream_receiver.stop_server()

            print("Receiver stopped.")

        except Exception as e:

            print(
                f"Receiver shutdown error: {e}"
            )


    # --------------------------------------------------------
    # STOP DEVICE STREAM
    # --------------------------------------------------------

    if stream_started:

        try:

            device.stop_streaming()

            print("Device streaming stopped.")

        except Exception as e:

            print(
                f"Device stop error: {e}"
            )


    # --------------------------------------------------------
    # DISCONNECT
    # --------------------------------------------------------

    try:

        device_client.disconnect(device)

        print("Device disconnected.")

    except Exception as e:

        print(
            f"Disconnect error: {e}"
        )


    print("Done.")