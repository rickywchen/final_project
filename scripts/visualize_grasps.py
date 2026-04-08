"""
Visualize grasp candidates on object mesh
Shows grasp positions and approach directions in 3D
"""

from pathlib import Path
import sys
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from object_loader import ObjectLoader
from grasp_sampler import GraspSampler


def visualize_grasps_3d(mesh, grasps, max_grasps=50, strategy_filter=None):
    """
    Visualize grasp candidates on 3D mesh
    
    Args:
        mesh: Object mesh
        grasps: List of GraspCandidate
        max_grasps: Maximum grasps to show
        strategy_filter: Only show specific strategy ('top_down', 'antipodal', 'lateral')
    """
    
    # Filter by strategy if specified
    if strategy_filter:
        grasps = [g for g in grasps if g.strategy == strategy_filter]
    
    # Limit number
    grasps = grasps[:max_grasps]
    
    if len(grasps) == 0:
        print(f"No grasps to visualize (filter: {strategy_filter})")
        return
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Top view
    ax1 = fig.add_subplot(131, projection='3d')
    plot_grasps_on_mesh(ax1, mesh, grasps, view='top')
    ax1.set_title(f'Top View ({len(grasps)} grasps)')
    
    # Plot 2: Side view
    ax2 = fig.add_subplot(132, projection='3d')
    plot_grasps_on_mesh(ax2, mesh, grasps, view='side')
    ax2.set_title('Side View')
    
    # Plot 3: Front view
    ax3 = fig.add_subplot(133, projection='3d')
    plot_grasps_on_mesh(ax3, mesh, grasps, view='front')
    ax3.set_title('Front View')
    
    plt.tight_layout()
    plt.show()


def plot_grasps_on_mesh(ax, mesh, grasps, view='top'):
    """Plot mesh with grasp arrows"""
    
    # Plot mesh
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Plot mesh wireframe
    for face in faces[::10]:  # Subsample faces for clarity
        triangle = vertices[face]
        ax.plot3D(*triangle.T, 'k-', alpha=0.1, linewidth=0.5)
    
    # Color map for strategies
    strategy_colors = {
        'top_down': 'blue',
        'antipodal': 'red',
        'lateral': 'green'
    }
    
    # Plot grasps
    for grasp in grasps:
        color = strategy_colors.get(grasp.strategy, 'gray')
        
        # Grasp position
        pos = grasp.position
        
        # Approach arrow (showing where gripper comes from)
        arrow_length = 0.03
        arrow_end = pos - arrow_length * grasp.approach_direction
        
        # Draw approach arrow
        ax.quiver(
            arrow_end[0], arrow_end[1], arrow_end[2],
            arrow_length * grasp.approach_direction[0],
            arrow_length * grasp.approach_direction[1],
            arrow_length * grasp.approach_direction[2],
            color=color, alpha=0.7, arrow_length_ratio=0.3, linewidth=2
        )
        
        # Draw grasp point
        ax.scatter(*pos, color=color, s=20, alpha=0.8)
    
    # Set view
    bbox = mesh.bounds
    center = (bbox[0] + bbox[1]) / 2
    size = np.max(bbox[1] - bbox[0])
    
    ax.set_xlim(center[0] - size, center[0] + size)
    ax.set_ylim(center[1] - size, center[1] + size)
    ax.set_zlim(center[2] - size, center[2] + size)
    
    # Set camera angle
    if view == 'top':
        ax.view_init(elev=90, azim=0)
    elif view == 'side':
        ax.view_init(elev=0, azim=90)
    elif view == 'front':
        ax.view_init(elev=0, azim=0)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')


def visualize_strategy_comparison(mesh, grasps):
    """Show different strategies separately"""
    
    strategies = {}
    for g in grasps:
        if g.strategy not in strategies:
            strategies[g.strategy] = []
        strategies[g.strategy].append(g)
    
    print(f"\nStrategy breakdown:")
    for strategy, strategy_grasps in strategies.items():
        print(f"  {strategy}: {len(strategy_grasps)} grasps")
    
    # Plot each strategy
    for strategy, strategy_grasps in strategies.items():
        if len(strategy_grasps) == 0:
            continue
        
        print(f"\nVisualizing {strategy} grasps...")
        visualize_grasps_3d(mesh, strategy_grasps, max_grasps=20, strategy_filter=strategy)


def visualize_interactive(mesh, grasps):
    """Interactive 3D visualization using trimesh viewer"""
    
    scene = trimesh.Scene()
    
    # Add mesh
    scene.add_geometry(mesh)
    
    # Color map
    strategy_colors = {
        'top_down': [0, 0, 255, 255],    # Blue
        'antipodal': [255, 0, 0, 255],   # Red
        'lateral': [0, 255, 0, 255]      # Green
    }
    
    # Add grasp arrows
    for i, grasp in enumerate(grasps[:50]):  # Limit to 50
        color = strategy_colors.get(grasp.strategy, [128, 128, 128, 255])
        
        # Create arrow mesh
        arrow_length = 0.03
        arrow_start = grasp.position - arrow_length * grasp.approach_direction
        arrow_end = grasp.position
        
        # Create cylinder for arrow shaft
        direction = arrow_end - arrow_start
        height = np.linalg.norm(direction)
        
        if height < 1e-6:
            continue
        
        # Create arrow as cylinder + cone
        arrow = trimesh.creation.cylinder(
            radius=0.002,
            height=height,
            sections=8
        )
        
        # Rotate and translate arrow
        direction_normalized = direction / height
        z_axis = np.array([0, 0, 1])
        
        if not np.allclose(direction_normalized, z_axis):
            rotation_axis = np.cross(z_axis, direction_normalized)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(z_axis, direction_normalized))
            
            arrow.apply_transform(
                trimesh.transformations.rotation_matrix(
                    rotation_angle, rotation_axis
                )
            )
        
        arrow.apply_translation(arrow_start + direction / 2)
        arrow.visual.vertex_colors = color
        
        scene.add_geometry(arrow, node_name=f"grasp_{i}")
    
    # Show interactive viewer
    scene.show()


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    loader = ObjectLoader(ROOT)
    sampler = GraspSampler()
    
    print("="*70)
    print("GRASP VISUALIZATION TEST")
    print("="*70)
    
    # Test object
    print("\nCreating test object...")
    mesh, props = loader.create_simple_cube(size=0.05)
    
    # Sample grasps
    print("\nSampling grasps...")
    grasps = sampler.sample_grasps(mesh, props, num_samples=50)
    
    print(f"\nGenerated {len(grasps)} grasps")
    print(f"  Top-down: {len([g for g in grasps if g.strategy == 'top_down'])}")
    print(f"  Antipodal: {len([g for g in grasps if g.strategy == 'antipodal'])}")
    print(f"  Lateral: {len([g for g in grasps if g.strategy == 'lateral'])}")
    
    # Visualize
    print("\n" + "="*70)
    print("VISUALIZATION OPTIONS")
    print("="*70)
    print("1. Static 3D plots (matplotlib)")
    print("2. Interactive 3D viewer (trimesh)")
    print("3. Strategy comparison")
    choice = input("\nChoose (1/2/3): ").strip()
    
    if choice == "1":
        visualize_grasps_3d(mesh, grasps, max_grasps=30)
    elif choice == "2":
        visualize_interactive(mesh, grasps)
    elif choice == "3":
        visualize_strategy_comparison(mesh, grasps)
    else:
        print("Invalid choice, showing static plots...")
        visualize_grasps_3d(mesh, grasps, max_grasps=30)