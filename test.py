# introspect StreamReceiver callback names
import aria.stream_receiver as receiver
print([m for m in dir(receiver.StreamReceiver) if not m.startswith('_')])

# introspect calibration extrinsic accessor
from projectaria_tools.core import data_provider
provider = data_provider.create_vrs_data_provider("/home/ayushi/aria_gen2/vrs_files/test1.vrs")
device_calib = provider.get_device_calibration()
print([m for m in dir(device_calib) if 'transform' in m.lower() or 'extrinsic' in m.lower()])