from .pix2pix_model import Pix2PixModel

def get_option_setter():
    return Pix2PixModel.modify_commandline_options
