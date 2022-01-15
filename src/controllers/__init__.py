REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_v import BasicMAC_V
from .basic_controller_gan import BasicMACGan
from .basic_controller_ali import BasicMACAli

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_mac_ali"] = BasicMACAli
REGISTRY["basic_mac_v"] = BasicMAC_V
REGISTRY["basic_mac_gan"] = BasicMACGan
