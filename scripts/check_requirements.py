"""
Check if all required files and dependencies exist
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

print("\n" + "="*70)
print("PROJECT REQUIREMENTS CHECK")
print("="*70)

# Check files
required_files = {
    "Core Scripts": [
        "scripts/object_loader.py",
        "scripts/grasp_sampler.py",
        "scripts/pickup_clean.py",
        "scripts/affordance_pipeline.py",
    ],
    "Scene Files": [
        "assets/scenes/scene_grasp_clean.xml",
        "assets/models/franka_emika_panda/panda.xml",
    ],
    "Test Scripts": [
        "scripts/test_pipeline_debug.py",
    ]
}

all_good = True

for category, files in required_files.items():
    print(f"\n{category}:")
    for file_path in files:
        full_path = ROOT / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING!")
            all_good = False

# Check Python imports
print("\nPython Dependencies:")
required_modules = [
    "numpy",
    "mujoco",
    "trimesh",
    "scipy",
    "matplotlib"
]

for module in required_modules:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError:
        print(f"  ✗ {module} - NOT INSTALLED!")
        all_good = False

# Check if imports work
print("\nModule Imports:")
try:
    sys.path.insert(0, str(ROOT / "scripts"))
    from object_loader import ObjectLoader
    print("  ✓ object_loader imports successfully")
except Exception as e:
    print(f"  ✗ object_loader import failed: {e}")
    all_good = False

try:
    from grasp_sampler import GraspSampler
    print("  ✓ grasp_sampler imports successfully")
except Exception as e:
    print(f"  ✗ grasp_sampler import failed: {e}")
    all_good = False

try:
    from affordance_pipeline import AffordancePipeline
    print("  ✓ affordance_pipeline imports successfully")
except Exception as e:
    print(f"  ✗ affordance_pipeline import failed: {e}")
    all_good = False

# Summary
print("\n" + "="*70)
if all_good:
    print("✓✓✓ ALL REQUIREMENTS SATISFIED ✓✓✓")
    print("You're ready to run the affordance pipeline!")
else:
    print("✗✗✗ SOME REQUIREMENTS MISSING ✗✗✗")
    print("Fix the issues above before proceeding.")
print("="*70 + "\n")