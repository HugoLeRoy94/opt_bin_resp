# core/__init__.py

from .loss import (BaseInformationLoss,
                    ProxyInformationLoss,
                    ExactInformationLoss)

from .bin_loss import (compute_shannon_joint_entropy,
                       compute_renyi_joint_entropy,
                        DiscreteProxyLoss,
                        DiscreteExactLoss)
#from .tolerant_bin_loss import TolerantDiscreteProxyLoss
                   
from .family_mi_loss import MaximizeMutualInformationFamilyLoss
from .concentration_mi_loss import MaximizeMutualInformationConcentrationLoss

__all__ = ["BaseInformationLoss",
            "ProxyInformationLoss",
            "ExactInformationLoss",
            "compute_shannon_joint_entropy",
            "compute_renyi_joint_entropy",
            "DiscreteProxyLoss",
            #"TolerantDiscreteProxyLoss",
            "DiscreteExactLoss",
            "MaximizeMutualInformationFamilyLoss",
            "MaximizeMutualInformationConcentrationLoss"]