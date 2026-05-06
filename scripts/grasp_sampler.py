"""
Grasp Sampler - Generate grasp candidates based on object geometry
Production version following Contact-GraspNet methodology
Uses uniform surface sampling + multi-strategy evaluation
FIXED: Correct gripper orientation for Franka Panda
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.spatial import KDTree

# Fix imports
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from object_loader import ObjectProperties


@dataclass
class GraspCandidate:
    """Single grasp candidate with all parameters"""
    grasp_id: int
    strategy: str  # 'top_down', 'antipodal', 'lateral'
    position: np.ndarray  # Grasp center position (3,) in WORLD coordinates
    approach_direction: np.ndarray  # Direction to approach from (3,)
    orientation: np.ndarray  # Quaternion (w, x, y, z)
    width: float  # Required gripper width (m)
    quality_score: float  # 0-1, higher is better
    contact_points: Tuple[np.ndarray, np.ndarray]  # (left_contact, right_contact)
    surface_normals: Tuple[np.ndarray, np.ndarray]  # Normals at contact points


class GraspSampler:
    """
    Production-grade grasp sampler
    Following Contact-GraspNet + Dex-Net best practices
    FIXED: Proper Franka Panda gripper orientation
    """
    
    def __init__(self, max_gripper_width: float = 0.08):
        """
        Args:
            max_gripper_width: Maximum gripper opening (Panda: 0.08m)
        """
        self.max_gripper_width = max_gripper_width
        self.min_gripper_width = 0.01  # Minimum stable grasp
        
        # Grasp height offsets (how high above surface to position gripper)
        self.top_down_height = 0.08  # 8cm above top surface
        self.antipodal_height = 0.05  # 5cm above grasp center
        self.lateral_height = 0.03    # 3cm offset
        
    def sample_grasps(self, mesh: trimesh.Trimesh, 
                     properties: ObjectProperties,
                     num_samples: int = 100) -> List[GraspCandidate]:
        """
        Sample grasp candidates using multiple strategies
        
        Industry standard approach:
        1. Uniform surface sampling (scale-invariant)
        2. Multi-strategy evaluation
        3. Quality-based filtering
        
        Args:
            mesh: Object mesh (in object-local coordinates)
            properties: Object properties
            num_samples: Approximate number of grasps to generate
        
        Returns:
            List of GraspCandidate objects with positions in OBJECT-LOCAL coords
            (caller must transform to world coordinates)
        """
        grasps = []
        grasp_id = 0
        
        # Step 1: Dense uniform surface sampling (Contact-GraspNet approach)
        num_surface_samples = num_samples * 5  # Oversample 5x
        surface_points, face_indices = mesh.sample(
            num_surface_samples, 
            return_index=True
        )
        surface_normals = mesh.face_normals[face_indices]
        
        print(f"  Sampled {num_surface_samples} surface points")
        
        # Step 2: Apply multiple grasp strategies
        
        # Strategy 1: Top-down grasps (vertical approach from above)
        # Best for: flat surfaces, stable grasps
        top_down_samples = max(10, num_samples // 4)
        top_down_grasps = self._sample_top_down_grasps(
            surface_points, surface_normals, properties, top_down_samples
        )
        for grasp_data in top_down_grasps:
            grasps.append(self._create_grasp_candidate(
                grasp_id, 'top_down', grasp_data
            ))
            grasp_id += 1
        
        # Strategy 2: Antipodal grasps (pinch from opposite sides)
        # Best for: symmetric objects, power grasps
        antipodal_samples = max(20, num_samples // 2)
        antipodal_grasps = self._sample_antipodal_grasps(
            surface_points, surface_normals, properties, antipodal_samples
        )
        for grasp_data in antipodal_grasps:
            grasps.append(self._create_grasp_candidate(
                grasp_id, 'antipodal', grasp_data
            ))
            grasp_id += 1
        
        # Strategy 3: Lateral grasps (horizontal approach)
        # Best for: tall objects, cylindrical objects
        lateral_samples = max(10, num_samples // 4)
        lateral_grasps = self._sample_lateral_grasps(
            surface_points, surface_normals, properties, lateral_samples
        )
        for grasp_data in lateral_grasps:
            grasps.append(self._create_grasp_candidate(
                grasp_id, 'lateral', grasp_data
            ))
            grasp_id += 1
        
        print(f"  Generated {len(grasps)} grasp candidates:")
        print(f"    - Top-down: {len(top_down_grasps)}")
        print(f"    - Antipodal: {len(antipodal_grasps)}")
        print(f"    - Lateral: {len(lateral_grasps)}")
        
        return grasps
    
    def _sample_top_down_grasps(self, points: np.ndarray, 
                                normals: np.ndarray,
                                properties: ObjectProperties,
                                num_samples: int) -> List[dict]:
        """
        Top-down grasp strategy
        
        Approach: Gripper descends vertically from above
        Position: Centered above object for small objects, surface points for large objects
        """
        grasps = []
        
        bbox = properties.bounding_box
        bbox_top = bbox['max'][2]
        bbox_center = bbox['center']
        object_size = np.linalg.norm(bbox['size'][:2])  # XY size
        
        # For small objects (<10cm), use centered grasps only
        # For large objects, sample surface points
        use_centered = object_size < 0.10
        
        if use_centered:
            # Generate multiple centered top-down grasps with small random offsets
            # This ensures gripper is always near center
            for i in range(min(num_samples, 10)):
                # Small random offset (±1cm) for diversity
                offset_xy = (np.random.rand(2) - 0.5) * 0.02
                
                grasp_pos = np.array([
                    bbox_center[0] + offset_xy[0],
                    bbox_center[1] + offset_xy[1],
                    bbox_top + self.top_down_height
                ])
                
                approach_dir = np.array([0, 0, -1])
                width = min(bbox['size'][0], bbox['size'][1])
                
                if width > self.max_gripper_width * 2.0:
                    continue
                
                quality = 1.0 - (np.linalg.norm(offset_xy) / 0.01)  # Higher if more centered
                
                contact_left = np.array([bbox_center[0] - width/2, bbox_center[1], bbox_top])
                contact_right = np.array([bbox_center[0] + width/2, bbox_center[1], bbox_top])
                
                grasps.append({
                    'position': grasp_pos,
                    'approach_direction': approach_dir,
                    'width': width,
                    'quality': quality,
                    'contact_points': (contact_left, contact_right),
                    'surface_normals': (np.array([0, 0, 1]), np.array([0, 0, 1]))
                })
        
        else:
            # Original approach for large objects: sample surface points
            up_direction = np.array([0, 0, 1])
            dot_products = np.dot(normals, up_direction)
            
            threshold = max(0.5, np.percentile(dot_products, 70))
            upward_mask = dot_products > threshold
            
            top_points = points[upward_mask]
            top_normals = normals[upward_mask]
            
            if len(top_points) == 0:
                z_threshold = np.percentile(points[:, 2], 80)
                top_mask = points[:, 2] >= z_threshold
                top_points = points[top_mask]
                top_normals = normals[top_mask]
            
            if len(top_points) == 0:
                return grasps
            
            sample_count = min(num_samples, len(top_points))
            sample_indices = np.random.choice(
                len(top_points), sample_count, replace=False
            )
            
            for idx in sample_indices:
                point = top_points[idx]
                normal = top_normals[idx]
                
                grasp_pos = np.array([
                    point[0],
                    point[1],
                    bbox_top + self.top_down_height
                ])
                
                approach_dir = np.array([0, 0, -1])
                width = min(bbox['size'][0], bbox['size'][1])
                
                if width > self.max_gripper_width * 2.0:
                    continue
                
                flatness = max(0.0, np.dot(normal, up_direction))
                distance_from_center = np.linalg.norm(point[:2] - bbox_center[:2])
                max_distance = max(0.001, np.linalg.norm(bbox['size'][:2]) / 2)
                centrality = 1.0 - min(1.0, distance_from_center / max_distance)
                
                quality = 0.6 * flatness + 0.4 * centrality
                
                contact_left = np.array([point[0] - width/2, point[1], point[2]])
                contact_right = np.array([point[0] + width/2, point[1], point[2]])
                
                grasps.append({
                    'position': grasp_pos,
                    'approach_direction': approach_dir,
                    'width': width,
                    'quality': quality,
                    'contact_points': (contact_left, contact_right),
                    'surface_normals': (normal, normal)
                })
        
        return grasps
    
    def _sample_antipodal_grasps(self, points: np.ndarray,
                                 normals: np.ndarray,
                                 properties: ObjectProperties,
                                 num_samples: int) -> List[dict]:
        """
        Antipodal grasp strategy
        
        Approach: Pinch object from opposite sides
        Position: Between contact points, with vertical offset
        """
        grasps = []
        
        if len(points) < 2:
            return grasps
        
        # Build KD-tree for fast opposite point search
        tree = KDTree(points)
        
        sample_count = min(num_samples, len(points))
        sample_indices = np.random.choice(
            len(points), sample_count, replace=False
        )
        
        bbox = properties.bounding_box
        bbox_center_z = bbox['center'][2]
        
        for idx in sample_indices:
            point1 = points[idx]
            normal1 = normals[idx]
            
            # Find antipodal point (opposite side, facing inward)
            opposite_direction = -normal1
            dot_products = np.dot(normals, opposite_direction)
            
            threshold = max(0.5, np.percentile(dot_products, 70))
            opposite_mask = dot_products > threshold
            opposite_indices = np.where(opposite_mask)[0]
            
            if len(opposite_indices) == 0:
                continue
            
            # Find closest opposite point
            opposite_points = points[opposite_indices]
            distances = np.linalg.norm(opposite_points - point1, axis=1)
            
            if len(distances) == 0:
                continue
            
            closest_idx = opposite_indices[np.argmin(distances)]
            point2 = points[closest_idx]
            normal2 = normals[closest_idx]
            
            # Grasp width
            width = np.linalg.norm(point2 - point1)
            
            if width < self.min_gripper_width or width > self.max_gripper_width * 1.5:
                continue
            
            # Grasp position: center of contact points + small upward offset
            grasp_center = (point1 + point2) / 2
            grasp_pos = np.array([
                grasp_center[0],
                grasp_center[1],
                max(grasp_center[2], bbox_center_z) + self.antipodal_height
            ])
            
            # Approach direction (average of inward normals)
            approach_dir = (normal1 - normal2) / 2
            if np.linalg.norm(approach_dir) > 1e-6:
                approach_dir = approach_dir / np.linalg.norm(approach_dir)
            else:
                approach_dir = np.array([0, 0, 1])  # Default: from above
            
            # Quality score
            normal_alignment = -np.dot(normal1, normal2)
            width_score = 1.0 - min(1.0, width / self.max_gripper_width)
            quality = 0.7 * max(0.0, normal_alignment) + 0.3 * width_score
            
            grasps.append({
                'position': grasp_pos,
                'approach_direction': approach_dir,
                'width': width,
                'quality': quality,
                'contact_points': (point1, point2),
                'surface_normals': (normal1, normal2)
            })
        
        return grasps
    
    def _sample_lateral_grasps(self, points: np.ndarray,
                               normals: np.ndarray,
                               properties: ObjectProperties,
                               num_samples: int) -> List[dict]:
        """
        Lateral grasp strategy
        
        Approach: Horizontal approach from side
        Position: Middle height of object
        """
        grasps = []
        
        bbox = properties.bounding_box
        z_min, z_max = bbox['min'][2], bbox['max'][2]
        z_mid = (z_min + z_max) / 2
        z_height = max(0.01, z_max - z_min)
        
        # Focus on middle third of object height
        middle_mask = np.abs(points[:, 2] - z_mid) < (z_height / 3)
        middle_points = points[middle_mask]
        middle_normals = normals[middle_mask]
        
        if len(middle_points) == 0:
            return grasps
        
        # Filter for horizontally-facing normals (not vertical)
        # Accept normals with Z component < 0.5 (more horizontal than vertical)
        lateral_mask = np.abs(middle_normals[:, 2]) < 0.5
        
        lateral_points = middle_points[lateral_mask]
        lateral_normals = middle_normals[lateral_mask]
        
        if len(lateral_points) == 0:
            return grasps
        
        sample_count = min(num_samples, len(lateral_points))
        sample_indices = np.random.choice(
            len(lateral_points), sample_count, replace=False
        )
        
        for idx in sample_indices:
            point = lateral_points[idx]
            normal = lateral_normals[idx]
            
            # Grasp position: offset from surface + small upward adjustment
            grasp_pos = point + normal * 0.03 + np.array([0, 0, self.lateral_height])
            
            # Approach direction (horizontal, toward object)
            approach_dir = -normal
            approach_dir[2] = 0  # Force horizontal
            
            if np.linalg.norm(approach_dir) > 1e-6:
                approach_dir = approach_dir / np.linalg.norm(approach_dir)
            else:
                continue
            
            # Gripper width
            width = min(bbox['size'][0], bbox['size'][1])
            
            if width > self.max_gripper_width * 1.5:
                continue
            
            # Quality: prefer more horizontal approaches
            horizontalness = 1.0 - abs(normal[2])
            quality = 0.5 + 0.5 * horizontalness
            
            # Contact points
            contact_left = point + np.array([0, -width/2, 0])
            contact_right = point + np.array([0, width/2, 0])
            
            grasps.append({
                'position': grasp_pos,
                'approach_direction': approach_dir,
                'width': width,
                'quality': quality,
                'contact_points': (contact_left, contact_right),
                'surface_normals': (normal, -normal)
            })
        
        return grasps
    
    def _create_grasp_candidate(self, grasp_id: int, strategy: str, 
                               grasp_data: dict) -> GraspCandidate:
        """Convert grasp data dict to GraspCandidate object"""
        
        # Compute orientation quaternion from approach direction
        approach_dir = grasp_data['approach_direction']
        orientation = self._compute_grasp_orientation(approach_dir)
        
        return GraspCandidate(
            grasp_id=grasp_id,
            strategy=strategy,
            position=grasp_data['position'],
            approach_direction=approach_dir,
            orientation=orientation,
            width=grasp_data['width'],
            quality_score=grasp_data['quality'],
            contact_points=grasp_data['contact_points'],
            surface_normals=grasp_data['surface_normals']
        )
    
    def _compute_grasp_orientation(self, approach_direction: np.ndarray) -> np.ndarray:
        """
        Compute gripper orientation quaternion from approach direction
        
        FIXED: Returns correct orientation for Franka Panda gripper
        Convention: Gripper fingers point along Y-axis, approach along -Z
        
        Returns:
            quaternion (w, x, y, z)
        """
        # Normalize approach direction
        approach = approach_direction / np.linalg.norm(approach_direction)
        
        # Default gripper orientation (pointing down -Z)
        default_approach = np.array([0, 0, -1])
        
        # Check if approach is downward (most common case)
        dot_product = np.dot(approach, default_approach)
        
        if dot_product > 0.999:
            # Pointing down: use 180° rotation around Y axis
            # This matches the working orientation from pickup_clean.py
            return np.array([0.0, 1.0, 0.0, 0.0])
        
        if dot_product < -0.999:
            # Pointing up: 180° rotation around X
            return np.array([0.0, 1.0, 0.0, 0.0])
        
        # For other directions, compute rotation
        axis = np.cross(default_approach, approach)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            # Fallback for edge cases
            return np.array([0.0, 1.0, 0.0, 0.0])
        
        # Normalize axis
        axis = axis / axis_norm
        
        # Rotation angle
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # Convert axis-angle to quaternion
        half_angle = angle / 2
        s = np.sin(half_angle)
        quat = np.array([
            np.cos(half_angle),  # w
            axis[0] * s,          # x
            axis[1] * s,          # y
            axis[2] * s           # z
        ])
        
        return quat
    
    def filter_feasible_grasps(self, grasps: List[GraspCandidate],
                              table_height: float = 0.42,
                              workspace_bounds: Optional[dict] = None) -> List[GraspCandidate]:
        """
        Filter grasps for kinematic and physical feasibility
        
        Checks:
        1. Table collision (approach path must be above table)
        2. Workspace reachability (within robot reach)
        3. Minimum quality threshold
        
        Args:
            grasps: List of grasp candidates (in WORLD coordinates)
            table_height: Height of table surface (m)
            workspace_bounds: Dict with 'x', 'y', 'z' bounds as (min, max) tuples
        
        Returns:
            Filtered list of feasible grasps
        """
        feasible = []
        
        for grasp in grasps:
            # Check 1: Table collision
            # Approach point must be above table with clearance
            approach_distance = 0.15
            approach_point = grasp.position - approach_distance * grasp.approach_direction
            
            if approach_point[2] < table_height + 0.05:  # 5cm clearance
                continue
            
            # Check 2: Workspace bounds
            if workspace_bounds is not None:
                pos = grasp.position
                if not (workspace_bounds['x'][0] <= pos[0] <= workspace_bounds['x'][1] and
                       workspace_bounds['y'][0] <= pos[1] <= workspace_bounds['y'][1] and
                       workspace_bounds['z'][0] <= pos[2] <= workspace_bounds['z'][1]):
                    continue
            
            # Check 3: Quality threshold
            if grasp.quality_score < 0.3:
                continue
            
            feasible.append(grasp)
        
        return feasible


if __name__ == "__main__":
    # Test grasp sampler
    from object_loader import ObjectLoader
    
    ROOT = Path(__file__).resolve().parent.parent
    loader = ObjectLoader(ROOT)
    sampler = GraspSampler()
    
    print("\n" + "="*70)
    print("PRODUCTION GRASP SAMPLER TEST (FIXED)")
    print("="*70)
    
    # Test with various object sizes
    test_objects = [
        ("Small Cube (3cm)", loader.create_simple_cube(size=0.03)),
        ("Medium Cube (5cm)", loader.create_simple_cube(size=0.05)),
        ("Large Cube (8cm)", loader.create_simple_cube(size=0.08)),
        ("Cylinder", loader.create_simple_cylinder(radius=0.03, height=0.10)),
        ("Sphere", loader.create_simple_sphere(radius=0.03)),
    ]
    
    for name, (mesh, props) in test_objects:
        print(f"\n{name}:")
        print(f"  Bounding box: {props.bounding_box['size']}")
        grasps = sampler.sample_grasps(mesh, props, num_samples=50)
        
        if len(grasps) > 0:
            avg_quality = np.mean([g.quality_score for g in grasps])
            print(f"  ✓ {len(grasps)} grasps, avg quality: {avg_quality:.2f}")
            
            # Show orientation
            sample_grasp = grasps[0]
            print(f"  Sample grasp orientation: {sample_grasp.orientation}")
            print(f"  Should be [0, 1, 0, 0] for top-down")
        else:
            print(f"  ✗ No grasps generated!")
    
    print("\n✓ Production grasp sampler test complete!")
    print("="*70 + "\n")