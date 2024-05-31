import numpy as np
import rerun as rr

from ..abstract.abstract_class import Component
from ..abstract.markers import MarkerProperties


class MarkersUpdater(Component):
    def __init__(self, name, marker_properties: MarkerProperties, callable_markers: callable, timeless=False):
        self.name = name + "/model_markers"
        self.marker_properties = marker_properties
        self.callable_markers = callable_markers
        self.timeless=timeless

    @property
    def nb_markers(self) -> int:
        return self.marker_properties.nb_markers

    @property
    def nb_components(self) -> int:
        return 1

    def to_rerun(self, q: np.ndarray) -> None:
        rr.log(
            self.name,
            rr.Points3D(
                positions=self.callable_markers(q),
                radii=self.marker_properties.radius_to_rerun(),
                colors=self.marker_properties.color_to_rerun(),
                labels=self.marker_properties.markers_names,
            ),
            timeless=self.timeless
        )


def from_pyo_to_rerun(maker_positions: np.ndarray) -> np.ndarray:
    """[3 x N] to [N x 3]"""
    return np.transpose(maker_positions, (1, 0))
