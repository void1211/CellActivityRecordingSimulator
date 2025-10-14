import importlib.metadata

# from .BaseObject import BaseObject
# from .BaseSettings import BaseSettings
# from .BaseSignal import BaseSignal
# from .Unit import Unit
# from .Settings import Settings
# from .Contact import Contact
# from .Template import BaseTemplate, GaborTemplate, ExponentialTemplate
# from .Noise import RandomNoise, DriftNoise, PowerLineNoise
# from .calculate import calculate_scaled_spike_amplitude, calculate_distance_two_objects
# from .generate import make_background_activity
# from .plot.main import plot_main
# from .tools import addSpikeToSignal, make_save_dir
# from .main import main, run
# from .carsIO import load_settings_from_json, load_units_from_json, load_contacts_from_json, save_data, load_noise_file, load_spike_templates

__version__ = importlib.metadata.version("cellactivityrecodingsimulator")

# __all__ = [
#     'BaseObject',
#     'BaseSettings',
#     'BaseSignal',
#     'Unit',
#     'Settings',
#     'Contact',
#     'BaseTemplate',
#     'GaborTemplate',
#     'ExponentialTemplate',
#     'RandomNoise',
#     'DriftNoise',
#     'PowerLineNoise',
#     'calculate_spike_max_amplitude',
#     'calculate_scaled_spike_amplitude',
#     'calculate_distance_two_objects',
#     'make_noise_units',
#     'make_background_activity',
#     'make_spike_times',
#     'plot_main',
#     'addSpikeToSignal',
#     'make_save_dir',
#     'main',
#     'run',
#     'load_settings_file',
#     'load_units_from_json',
#     'load_contacts_from_json',
#     'save_data',
#     'load_noise_file',
#     'load_spike_templates'
# ]
