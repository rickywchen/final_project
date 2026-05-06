"""
Object Loader - Load and analyze 3D objects for grasp planning
Extended with tall box support for testing antipodal grasps
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ObjectProperties:
    """Physical and geometric properties of an object"""
    name: str
    mass: float  # kg
    center_of_mass: np.ndarray  # (3,) relative to object frame
    inertia_matrix: np.ndarray  # (3, 3)
    bounding_box: Dict[str, np.ndarray]  # min, max, center, size
    volume: float  # m^3
    surface_area: float  # m^2
    is_graspable: bool
    stability_score: float  # 0-1, higher is more stable


class ObjectLoader:
    """Load and analyze objects for manipulation"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.objects_dir = root_dir / "assets" / "objects"
        self.default_density = 400.0  # kg/m^3 (wood-like)
    
    def create_simple_cube(self, size: float = 0.05) -> Tuple[trimesh.Trimesh, ObjectProperties]:
        """
        Create a simple cube for testing
        
        Args:
            size: Cube edge length (m)
        
        Returns:
            (mesh, properties)
        """
        # Create cube mesh centered at origin
        mesh = trimesh.creation.box(extents=[size, size, size])
        
        # Compute properties
        props = self._compute_properties(mesh, name=f"cube_{int(size*100)}cm")
        
        return mesh, props
    
    def create_simple_cylinder(self, radius: float = 0.03, 
                              height: float = 0.10) -> Tuple[trimesh.Trimesh, ObjectProperties]:
        """
        Create a simple cylinder for testing tall object grasps
        
        Args:
            radius: Cylinder radius (m)
            height: Cylinder height (m)
        
        Returns:
            (mesh, properties)
        """
        # Create cylinder mesh centered at origin
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
        
        # Compute properties
        props = self._compute_properties(mesh, name=f"cylinder_r{int(radius*100)}_h{int(height*100)}")
        
        return mesh, props
    
    def create_simple_sphere(self, radius: float = 0.03) -> Tuple[trimesh.Trimesh, ObjectProperties]:
        """
        Create a simple sphere for testing
        
        Args:
            radius: Sphere radius (m)
        
        Returns:
            (mesh, properties)
        """
        # Create sphere mesh centered at origin
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        
        # Compute properties
        props = self._compute_properties(mesh, name=f"sphere_{int(radius*100)}cm")
        
        return mesh, props
    
    def create_tall_box(self, width: float = 0.05, depth: float = 0.05, 
                       height: float = 0.12) -> Tuple[trimesh.Trimesh, ObjectProperties]:
        """
        Create a tall rectangular box for testing tall object grasps
        
        Args:
            width: Box width (X, m)
            depth: Box depth (Y, m)
            height: Box height (Z, m)
        
        Returns:
            (mesh, properties)
        """
        # Create box mesh centered at origin
        mesh = trimesh.creation.box(extents=[width, depth, height])
        
        # Compute properties
        props = self._compute_properties(mesh, name=f"tall_box_{int(height*100)}cm")
        
        return mesh, props
    
    def _compute_properties(self, mesh: trimesh.Trimesh, 
                           name: str = "object") -> ObjectProperties:
        """Compute physical properties from mesh"""
        
        # Mass (assume uniform density)
        volume = mesh.volume
        mass = volume * self.default_density
        
        # Center of mass (assume uniform)
        com = mesh.center_mass
        
        # Inertia (from trimesh, relative to COM)
        inertia = mesh.moment_inertia
        
        # Bounding box
        bounds = mesh.bounds  # (2, 3) array: [min, max]
        bbox_min = bounds[0]
        bbox_max = bounds[1]
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min
        
        bounding_box = {
            'min': bbox_min,
            'max': bbox_max,
            'center': bbox_center,
            'size': bbox_size
        }
        
        # Surface area
        surface_area = mesh.area
        
        # Graspability (simple heuristic)
        max_dim = np.max(bbox_size)
        min_dim = np.min(bbox_size)
        is_graspable = (min_dim > 0.01 and max_dim < 0.30)  # 1cm - 30cm range
        
        # Stability (based on base area vs height)
        base_area = bbox_size[0] * bbox_size[1]
        height = bbox_size[2]
        stability = min(1.0, base_area / (height * 0.01)) if height > 0 else 1.0
        
        return ObjectProperties(
            name=name,
            mass=mass,
            center_of_mass=com,
            inertia_matrix=inertia,
            bounding_box=bounding_box,
            volume=volume,
            surface_area=surface_area,
            is_graspable=is_graspable,
            stability_score=stability
        )


if __name__ == "__main__":
    # Test object loader
    ROOT = Path(__file__).resolve().parent.parent
    loader = ObjectLoader(ROOT)
    
    print("\n" + "="*70)
    print("OBJECT LOADER TEST")
    print("="*70)
    
    # Test cube
    mesh, props = loader.create_simple_cube(size=0.05)
    print(f"\nCube (5cm):")
    print(f"  Mass: {props.mass*1000:.1f}g")
    print(f"  Volume: {props.volume*1e6:.1f}cm³")
    print(f"  Bounding box: {props.bounding_box['size']}")
    print(f"  Graspable: {props.is_graspable}")
    print(f"  Stability: {props.stability_score:.2f}")
    
    # Test cylinder
    mesh, props = loader.create_simple_cylinder(radius=0.03, height=0.10)
    print(f"\nCylinder (r=3cm, h=10cm):")
    print(f"  Mass: {props.mass*1000:.1f}g")
    print(f"  Volume: {props.volume*1e6:.1f}cm³")
    print(f"  Bounding box: {props.bounding_box['size']}")
    print(f"  Graspable: {props.is_graspable}")
    print(f"  Stability: {props.stability_score:.2f}")
    
    # Test sphere
    mesh, props = loader.create_simple_sphere(radius=0.03)
    print(f"\nSphere (r=3cm):")
    print(f"  Mass: {props.mass*1000:.1f}g")
    print(f"  Volume: {props.volume*1e6:.1f}cm³")
    print(f"  Bounding box: {props.bounding_box['size']}")
    print(f"  Graspable: {props.is_graspable}")
    print(f"  Stability: {props.stability_score:.2f}")
    
    # Test tall box
    mesh, props = loader.create_tall_box(width=0.05, depth=0.05, height=0.12)
    print(f"\nTall Box (5x5x12cm):")
    print(f"  Mass: {props.mass*1000:.1f}g")
    print(f"  Volume: {props.volume*1e6:.1f}cm³")
    print(f"  Bounding box: {props.bounding_box['size']}")
    print(f"  Graspable: {props.is_graspable}")
    print(f"  Stability: {props.stability_score:.2f}")
    
    print("\n✓ Object loader test complete!")
    print("="*70 + "\n")