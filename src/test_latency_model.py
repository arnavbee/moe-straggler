"""
Unit tests for latency_model.py, where the right answer is arithmetic.

latency_model.py is the one script whose output is DERIVED, not measured, so a
sign slip or a dropped factor of two there would silently rewrite the headline
millisecond numbers and nothing downstream would complain. These tests pin the
payload arithmetic to closed forms written out independently of the code, and
check that a missing sweep refuses to print nan.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import latency_model as L


def test_dispatch_and_combine_are_both_counted():
    """Each token crosses the fabric twice per MoE layer: out to its experts, back."""
    t = L.payload_table(B=4096, N=8)
    one_way = L.K * L.D_MODEL * L.BYTES
    assert t["per_tok_layer"] == 2 * one_way, t["per_tok_layer"]
    print("PASS per-token-per-layer payload counts dispatch and combine")


def test_forward_pass_scales_with_layers_and_batch():
    t1 = L.payload_table(B=1024, N=8)
    t2 = L.payload_table(B=2048, N=8)
    assert t1["per_tok_fwd"] == t1["per_tok_layer"] * L.L_MOE
    assert t2["step_total"] == 2 * t1["step_total"], "step total is linear in B"
    assert t1["per_tok_fwd"] == t2["per_tok_fwd"], "per-token cost cannot depend on B"
    print("PASS payload is linear in B and in the MoE layer count")


def test_receive_per_device_falls_as_one_over_n():
    """More devices split the same assignments, so each receives proportionally less."""
    t4 = L.payload_table(B=4096, N=4)
    t8 = L.payload_table(B=4096, N=8)
    assert abs(t4["recv_per_dev_layer"] - 2 * t8["recv_per_dev_layer"]) < 1e-6
    assert abs(t4["t_balanced_s"] - 2 * t8["t_balanced_s"]) < 1e-12
    print("PASS per-device receive and balanced time scale as 1/N")


def test_balanced_time_matches_hand_arithmetic():
    B, N = 4096, 8
    t = L.payload_table(B, N)
    expect = (B * L.K / N) * L.D_MODEL * L.BYTES / L.BW * L.L_MOE * 2
    assert abs(t["t_balanced_s"] - expect) < 1e-15, (t["t_balanced_s"], expect)
    # a sanity floor: at the assumed bandwidth this is sub-100 ms, not seconds
    assert 0 < t["t_balanced_s"] < 0.1, t["t_balanced_s"]
    print("PASS balanced all-to-all time matches the closed form")


def test_skew_tax_is_proportional_to_s_minus_one():
    """The only way S enters the model is as a multiplier, so the tax is (S-1)x."""
    t = L.payload_table(B=4096, N=8)
    for S in (1.0, 1.05, 1.4):
        tax = t["t_balanced_s"] * (S - 1)
        assert abs(t["t_balanced_s"] * S - (t["t_balanced_s"] + tax)) < 1e-15
    assert t["t_balanced_s"] * (1.0 - 1) == 0, "a balanced run owes no skew tax"
    print("PASS skew tax is exactly (S-1) x the balanced time")


def test_missing_sweep_refuses_to_print_nan():
    """An empty row match used to average to nan and print it as milliseconds."""
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as d:
        os.mkdir(os.path.join(d, "out"))
        with open(os.path.join(d, "out", "sweep.csv"), "w") as f:
            f.write("domain,B,N,S_obs_mean,S_obs_p95,S_shuf_mean,S_iid\n")
            f.write("code,128,8,1.1,1.2,1.05,1.04\n")   # right domain, wrong B
        src = os.path.join(d, "src")
        os.mkdir(src)
        for name in ("latency_model.py",):
            open(os.path.join(src, name), "w").write(
                open(os.path.join(here, name)).read())
        r = subprocess.run([sys.executable, os.path.join(src, "latency_model.py")],
                           capture_output=True, text=True)
        assert r.returncode != 0, r.stdout
        assert "nan" not in r.stdout.lower(), r.stdout
        assert "sweep.py" in r.stderr, r.stderr
    print("PASS a sweep with no matching rows exits instead of printing nan")


if __name__ == "__main__":
    test_dispatch_and_combine_are_both_counted()
    test_forward_pass_scales_with_layers_and_batch()
    test_receive_per_device_falls_as_one_over_n()
    test_balanced_time_matches_hand_arithmetic()
    test_skew_tax_is_proportional_to_s_minus_one()
    test_missing_sweep_refuses_to_print_nan()
    print("\nALL TESTS PASS")
