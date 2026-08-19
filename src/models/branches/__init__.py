"""The four specialized forensic branches."""
from .face_crop_branch import FaceCropBranch
from .full_frame_branch import FullFrameBranch
from .frequency_branch import FrequencyBranch
from .motion_branch import MotionBranch

__all__ = [
    "FaceCropBranch",
    "FullFrameBranch",
    "FrequencyBranch",
    "MotionBranch",
]
