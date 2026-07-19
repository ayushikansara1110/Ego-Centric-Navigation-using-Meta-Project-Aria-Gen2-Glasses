from projectaria_tools.core import data_provider
reader = data_provider.create_vrs_data_provider("vrs_files/new/Corridor.vrs")
print(reader.get_all_streams())