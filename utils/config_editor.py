from ast import literal_eval
from utils.writable import Writable

def write_recursively(setting, config_parser):
    section_name = setting.get_name()

    config_parser.add_section(section_name)

    for attrib in dir(setting):
        # ignore methods
        val = getattr(setting, attrib)
        if callable(val) or attrib.startswith("_"):
            continue
        if issubclass(val.__class__, Writable):
            print("Extracting parameters of class: ", val.get_name())
            write_recursively(val, config_parser)
        else:
            config_parser.set(section_name, attrib, str(val))

def read_recursively(setting, config_parser):
    section_name = setting.get_name()

    for attrib in dir(setting):
        val = getattr(setting, attrib)

        if callable(val) or attrib.startswith("_"):
            continue
        if issubclass(val.__class__, Writable):
            # print("Updating parameters of class: ", val.get_name()) NOTE: add to see the classes being updated
            read_recursively(val, config_parser)
        elif config_parser.has_option(section_name, attrib):
            value_in_config = config_parser.get(section_name, attrib)
            value_in_config = do_type_checking(value_in_config)
            setattr(setting, attrib, value_in_config)
    
    for option in config_parser.options(section_name):
        if option not in dir(setting):
            value_in_config = config_parser.get(section_name, option)
            value_in_config = do_type_checking(value_in_config)
            setattr(setting, option, value_in_config)

def do_type_checking(value):
    if value.isdigit() or value == "0":
        value = int(value)
    elif value == "None":
        value = None
    elif value == "True":
        value = True
    elif value == "False":
        value = False
    elif "{" in value:  # if string contains a "{", the string is probably a dictionary (e.g. boundary conditions)
        evald = literal_eval(value)
        if isinstance(evald, dict):
            value = evald
    else:
        try:
            value = float(value)
        except ValueError:
            pass
    return value


# def read_recursively(setting, config_parser):
#     section_name = setting.get_name()

#     for attrib in dir(setting):
#         val = getattr(setting, attrib)

#         if callable(val) or attrib.startswith("_"):
#             continue
#         if issubclass(val.__class__, Writable):
#             read_recursively(val, config_parser)
#         elif config_parser.has_option(section_name, attrib):
#             value_in_config = config_parser.get(section_name, attrib)

#             # Do type checking
#             if value_in_config.isdigit() or value_in_config == "0":
#                 value_in_config = int(value_in_config)
#             elif value_in_config == "None":
#                 value_in_config = None
#             elif value_in_config == "True":
#                 value_in_config = True
#             elif value_in_config == "False":
#                 value_in_config = False
#             elif "{" in value_in_config:  # if string contains a "{", the string is probably a dictionary (e.g. boundary conditions)
#                 evald = literal_eval(value_in_config)
#                 if isinstance(evald, dict):
#                     value_in_config = evald
#             else:
#                 try:
#                     value_in_config = float(value_in_config)
#                 except ValueError:
#                     pass
            
#             setattr(setting, attrib, value_in_config)




