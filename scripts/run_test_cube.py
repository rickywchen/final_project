"""
Quick test: run empirical trials on a simple cube.
Used to verify the grasp sampler works correctly on an easy object
before testing on the harder bottle scene.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from empirical_grasp_sampler import run_trials

SCENE = str(ROOT / "assets" / "scenes" / "scene_test_cube.xml")

results = run_trials(
    xml_path       = SCENE,
    obj_name       = "cube",
    n_top          = 10,
    n_side_heights = 4,       # only 4 heights — cube is short
    n_side_angles  = 12,
    show_viewer    = True,
    verbose        = True,
    rng_seed       = 42,
)

# Quick summary
attempted = [r for r in results if r.fail_reason != "unreachable"]
successes = [r for r in results if r.success]
print(f"\n  Quick test: {len(successes)}/{len(attempted)} succeeded "
      f"({len(successes)/max(len(attempted),1)*100:.0f}%)")

# Check for table collision symptoms
bad_lifts = [r for r in attempted if r.lift_m < -0.10]
if bad_lifts:
    print(f"\n  ⚠ WARNING: {len(bad_lifts)} trials with lift < -10cm")
    print(f"  These indicate table collision (arm pushing object down):")
    for r in bad_lifts:
        print(f"    Trial {r.trial_id}: z={r.contact_height:.3f} "
              f"angle={r.approach_angle*180/3.14159:.0f}° "
              f"lift={r.lift_cm:.1f}cm  {r.fail_reason}")
else:
    print(f"\n  ✓ No table collision detected (no lift < -10cm)")

descent_fails = [r for r in attempted if r.fail_reason == "failed_descent"]
if descent_fails:
    print(f"\n  ⚠ {len(descent_fails)} failed_descent trials:")
    for r in descent_fails:
        print(f"    Trial {r.trial_id}: z={r.contact_height:.3f} "
              f"angle={r.approach_angle*180/3.14159:.0f}°")
else:
    print(f"  ✓ No failed_descent trials")