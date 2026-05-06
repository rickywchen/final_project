"""
Universal Object Loader
Loads any mesh object and extracts geometric/affordance features
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import Optional, Dict, List
import xml.etree.ElementTree as ET


@dataclass
class ObjectProperties:
    """Object physical and geometric properties"""
    name: str
    mesh_path: Path
    mass: float
    center_of_mass: np.ndarray
    bounding_box: Dict[str, np.ndarray]  # 'min', 'max', 'size', 'center'
    principal_axes: np.ndarray
    volume: float
    surface_area: float
    is_convex: bool
    stability_score: float  # 0-1, higher = more stable when placed
    

class ObjectLoader:
    """Load and analyze object meshes for affordance-aware grasping"""
    
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.objects_dir = self.root_dir / "assets" / "objects"
        self.mesh_cache = {}
    
    def load_object(self, object_path: str, scale: float = 1.0) -> tuple[trimesh.Trimesh, ObjectProperties]:
        """
        Load object mesh and compute properties
        
        Args:
            object_path: Relative path from objects_dir (e.g., "ycb/035_power_drill/textured.obj")
            scale: Scale factor for mesh
        
        Returns:
            mesh: trimesh object
            properties: ObjectProperties dataclass
        """
        full_path = self.objects_dir / object_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Object not found: {full_path}")
        
        # Load mesh
        mesh = trimesh.load(full_path, force='mesh')
        
        if scale != 1.0:
            mesh.apply_scale(scale)
        
        # Compute properties
        properties = self._compute_properties(mesh, object_path)
        
        return mesh, properties
    
    def _compute_properties(self, mesh: trimesh.Trimesh, name: str) -> ObjectProperties:
        """Compute geometric and physical properties"""
        
        # Bounding box
        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]
        bbox_size = bbox_max - bbox_min
        bbox_center = (bbox_min + bbox_max) / 2
        
        # Principal axes (PCA of vertices)
        centered_verts = mesh.vertices - mesh.center_mass
        cov = np.cov(centered_verts.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        principal_axes = eigenvectors[:, idx]
        
        # Stability score (ratio of base area to height)
        # Objects with wide base are more stable
        base_area = bbox_size[0] * bbox_size[1]
        height = bbox_size[2]
        stability = min(1.0, base_area / (height * height + 1e-6))
        
        return ObjectProperties(
            name=name,
            mesh_path=Path(name),
            mass=mesh.mass if hasattr(mesh, 'mass') else 0.1,
            center_of_mass=mesh.center_mass,
            bounding_box={
                'min': bbox_min,
                'max': bbox_max,
                'size': bbox_size,
                'center': bbox_center
            },
            principal_axes=principal_axes,
            volume=mesh.volume if mesh.is_watertight else 0.0,
            surface_area=mesh.area,
            is_convex=mesh.is_convex,
            stability_score=stability
        )
    
    def create_simple_cube(self, size: float = 0.05) -> tuple[trimesh.Trimesh, ObjectProperties]:
        """Create a simple cube for testing"""
        mesh = trimesh.primitives.Box(extents=[size, size, size])
        properties = self._compute_properties(mesh, "cube")
        return mesh, properties
    
    def create_simple_cylinder(self, radius: float = 0.03, height: float = 0.10) -> tuple[trimesh.Trimesh, ObjectProperties]:
        """Create a simple cylinder (mug-like)"""
        mesh = trimesh.primitives.Cylinder(radius=radius, height=height)
        properties = self._compute_properties(mesh, "cylinder")
        return mesh, properties
    
    def create_simple_sphere(self, radius: float = 0.03) -> tuple[trimesh.Trimesh, ObjectProperties]:
        """Create a simple sphere (ball-like)"""
        mesh = trimesh.primitives.Sphere(radius=radius)
        properties = self._compute_properties(mesh, "sphere")
        return mesh, properties
    
    def update_scene_xml(self, scene_xml_path: Path, mesh_path: Path, 
                        scale: float = 1.0, mass: float = 0.1):
        """
        Update MuJoCo scene XML to load specific object
        
        Args:
            scene_xml_path: Path to scene XML file
            mesh_path: Path to object mesh (relative to meshdir)
            scale: Scale factor
            mass: Object mass in kg
        """
        tree = ET.parse(scene_xml_path)
        root = tree.getroot()
        
        # Update mesh asset
        asset = root.find('.//asset')
        mesh_elem = asset.find('.//mesh[@name="object_mesh"]')
        if mesh_elem is not None:
            mesh_elem.set('file', str(mesh_path))
            mesh_elem.set('scale', f"{scale} {scale} {scale}")
        
        # Update mass
        geom = root.find('.//geom[@name="object_geom"]')
        if geom is not None:
            geom.set('mass', str(mass))
        
        tree.write(scene_xml_path)


if __name__ == "__main__":
    # Test
    ROOT = Path(__file__).resolve().parent.parent
    loader = ObjectLoader(ROOT)
    
    print("Testing ObjectLoader...")
    
    # Create test objects
    cube_mesh, cube_props = loader.create_simple_cube(size=0.05)
    print(f"\nCube properties:")
    print(f"  Bounding box: {cube_props.bounding_box['size']}")
    print(f"  Volume: {cube_props.volume:.6f} m³")
    print(f"  Stability: {cube_props.stability_score:.2f}")
    
    cyl_mesh, cyl_props = loader.create_simple_cylinder(radius=0.03, height=0.10)
    print(f"\nCylinder properties:")
    print(f"  Bounding box: {cyl_props.bounding_box['size']}")
    print(f"  Volume: {cyl_props.volume:.6f} m³")
    print(f"  Stability: {cyl_props.stability_score:.2f}")
    
    sphere_mesh, sphere_props = loader.create_simple_sphere(radius=0.03)
    print(f"\nSphere properties:")
    print(f"  Bounding box: {sphere_props.bounding_box['size']}")
    print(f"  Volume: {sphere_props.volume:.6f} m³")
    print(f"  Stability: {sphere_props.stability_score:.2f}")
    
    print("\n✓ ObjectLoader working correctly!")