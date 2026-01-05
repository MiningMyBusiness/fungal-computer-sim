"""Quick test script to verify system identification protocols work correctly."""

from realistic_sim import RealisticFungalComputer
import numpy as np

print("Testing system identification protocols...")
print("="*70)

# Create a small test environment
env = RealisticFungalComputer(num_nodes=20, random_seed=42)

# Set some test parameters
env.tau_v = 50.0
env.tau_w = 800.0
env.a = 0.7
env.b = 0.8
env.v_scale = 5.0
env.R_off = 100.0
env.R_on = 10.0
env.alpha = 0.01

print("\n1. Testing Step Response Protocol...")
try:
    step_result = env.step_response_protocol(voltage=2.0, pulse_duration=3000.0, 
                                             probe_distance=5.0, sim_time=5000.0)
    print(f"   ✓ Success!")
    print(f"   Rise time: {step_result['rise_time']:.1f} ms")
    print(f"   Saturation voltage: {step_result['saturation_voltage']:.3f} V")
    print(f"   Oscillation index: {step_result['oscillation_index']:.4f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n2. Testing Paired-Pulse Protocol...")
try:
    pp_result = env.paired_pulse_protocol(voltage=2.0, pulse_width=50.0,
                                          probe_distance=5.0, 
                                          delays=[200.0, 800.0, 2000.0])
    print(f"   ✓ Success!")
    print(f"   Recovery ratios: {pp_result['recovery_ratios']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n3. Testing Triangle Sweep Protocol...")
try:
    tri_result = env.triangle_sweep_protocol(v_max=5.0, sweep_rate=0.01,
                                             probe_distance=5.0)
    print(f"   ✓ Success!")
    print(f"   Hysteresis area: {tri_result['hysteresis_area']:.4f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "="*70)
print("All tests completed!")
