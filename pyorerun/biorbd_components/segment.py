import numpy as np

from pyorerun.abstract.abstract_class import Component
from .local_frame import LocalFrameUpdater
from .mesh import TransformableMeshUpdater


class SegmentUpdater(Component):
    def __init__(self, name, transform_callable: callable, mesh: TransformableMeshUpdater, timeless=False):
        self.name = name
        self.transform_callable = transform_callable
        self.mesh = mesh
        self.local_frame = LocalFrameUpdater(name + "/frame", transform_callable)
        self.timeless=timeless

    @property
    def nb_components(self):
        nb_components = 0
        for component in self.components:
            nb_components += component.nb_components()

    @property
    def components(self) -> list[Component]:
        return [self.mesh, self.local_frame]

    def to_rerun(self, q: np.ndarray) -> None:
        for component in self.components:
            component.to_rerun(q
)

    @property
    def component_names(self) -> list[str]:
        return [component.name for component in self.components]
