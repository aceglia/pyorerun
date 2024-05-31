import numpy as np
import rerun as rr

from ..abstract.abstract_class import ExperimentalData


class DepthImage(ExperimentalData):
    def __init__(self, name, depth_image: np.ndarray):
        self.name: str = name + "/depth_image"
        self.depth_image: np.ndarray = depth_image

    @property
    def size_x(self):
        return self.depth_image.shape[0]

    @property
    def size_y(self):
        return self.depth_image.shape[1]

    @property
    def nb_frames(self):
        return self.depth_image.shape[2]

    @property
    def nb_components(self):
        return 1

    def to_rerun(self, frame: int) -> None:
        depth_image_frame = self.depth_image[:, :, frame]
        fx = 419.5417175292969
        fy = 419.13189697265625
        cx = 418.1453857421875
        cy = 245.16981506347656
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        rr.log(
            self.name,
            rr.Pinhole(
                image_from_camera=intrinsics,
                width=depth_image_frame.shape[1],
                height=depth_image_frame.shape[0],
                # focal_length=200,
            ),
        )

        # Log the tensor.
        rr.log(f"{self.name}/depth", rr.DepthImage(depth_image_frame, meter=1000))
