    
# Other good ones:
# en-US-AriaNeural
# en-US-JennyNeural
# en-IN-NeerjaNeural
# en-AU-NatashaNeural

from projectaria_tools.core import data_provider

provider = data_provider.create_vrs_data_provider(
    "/home/ayushi/aria_gen2/vrs_files/test1.vrs"
)

print("Streams:")
for stream in provider.get_all_streams():
    print(stream, provider.get_label_from_stream_id(stream))
    
from projectaria_tools.core import data_provider

provider = data_provider.create_vrs_data_provider(
    "/home/ayushi/aria_gen2/vrs_files/test1.vrs"
)

for stream in provider.get_all_streams():
    sid = stream
    label = provider.get_label_from_stream_id(stream)
    print("--------------")
    print(label)
    print(provider.get_num_data(sid))