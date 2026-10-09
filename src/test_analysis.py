"""
Unit tests for analysis.py against SYNTHETIC data, where the right answer is known.

The load-bearing one is test_shuffle_matches_iid: if the shuffle control and the
closed form disagree on data that really is iid, then one of them is wrong and
every number in the real experiment would be uninterpretable.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A

E, K = 32, 8
rng = np.random.default_rng(7)


def synth_iid(T, p=None):
    """T tokens, each choosing K distinct experts without replacement from marginal p."""
    if p is None:
        p = np.ones(E) / E
    out = np.empty((T, K), dtype=np.int8)
    # vectorised Gumbel top-k: sampling K without replacement with weights p
    g = rng.gumbel(size=(T, E)) + np.log(p)[None, :]
    out[:] = np.argsort(-g, axis=1)[:, :K]
    return out


def synth_correlated(T, run_len, p=None):
    """Same marginals, but tokens come in runs that repeat one token's choice."""
    base = synth_iid(T // run_len + 1, p)
    return np.repeat(base, run_len, axis=0)[:T]


def test_counts_conserve():
    x = synth_iid(4096)
    c = A.expert_counts(x, 512, n_experts=E)
    assert (c.sum(axis=1) == 512 * K).all(), "counts must sum to B*K"
    print("PASS counts conserve mass")


def test_straggler_bounds():
    x = synth_iid(4096)
    c = A.expert_counts(x, 512, n_experts=E)
    for N in (2, 4, 8, 16, 32):
        s = A.straggler(c, N)
        assert (s >= 1.0 - 1e-9).all(), "S < 1 at N=%d" % N
        assert (s <= N + 1e-9).all()
    # S <= M whenever E/N > 1
    m = A.expert_max_ratio(c)
    for N in (2, 4, 8, 16):
        assert (A.straggler(c, N) <= m + 1e-9).all(), "S > M at N=%d" % N
    assert np.allclose(A.straggler(c, E), m), "S must equal M at N=E"
    print("PASS 1 <= S <= M, and S == M at N = E")


def test_shuffle_matches_iid():
    """The load-bearing test."""
    print("\n  B      N   S_shuf   S_iid    diff     [S_iid_asym, the uncorrected form]")
    worst = 0.0
    # One large fixed corpus for every B: the closed form needs max_j P_j, and a
    # marginal estimated from only B*40 tokens is itself upward-biased, which
    # would show up as a fake disagreement at small B.
    x_big = synth_iid(600_000)
    for N in (4, 8, 16):
        for B in (128, 512, 2048, 8192):
            x = x_big
            r = A.three_curves(x, B, N, n_experts=E, n_shuffles=8,
                               rng=np.random.default_rng(1))
            d = abs(r["S_shuf_mean"] - r["S_iid"])
            worst = max(worst, d)
            print("  %-6d %-3d %.4f   %.4f   %+.4f   [%.4f]"
                  % (B, N, r["S_shuf_mean"], r["S_iid"],
                     r["S_shuf_mean"] - r["S_iid"], r["S_iid_asym"]))
    print("  worst |S_shuf - S_iid| = %.4f" % worst)
    assert worst < 0.02, "closed form and shuffler disagree by %.4f" % worst
    print("PASS S_shuf tracks S_iid on synthetic iid data")


def test_shuffle_kills_correlation():
    """On correlated data S_obs must exceed S_shuf, and S_shuf must still match S_iid."""
    print("\n  run_len  S_obs    S_shuf   S_iid")
    for run_len in (1, 4, 16, 64):
        x = synth_correlated(2048 * 40, run_len)
        r = A.three_curves(x, 2048, 8, n_experts=E, n_shuffles=20,
                           rng=np.random.default_rng(2))
        print("  %-8d %.4f   %.4f   %.4f"
              % (run_len, r["S_obs_mean"], r["S_shuf_mean"], r["S_iid"]))
        if run_len == 1:
            assert abs(r["S_obs_mean"] - r["S_shuf_mean"]) < 0.02
        if run_len >= 16:
            assert r["S_obs_mean"] > r["S_shuf_mean"] + 0.01, \
                "shuffle failed to detect injected correlation at run_len=%d" % run_len
        # shuffling preserves marginals, so S_shuf should stay near the closed form
        assert abs(r["S_shuf_mean"] - r["S_iid"]) < 0.02
    print("PASS shuffle control detects injected correlation, marginals preserved")


def test_skewed_marginals():
    """With a skewed marginal, S_shuf must converge to N*max_j P_j, not to 1."""
    p = np.ones(E)
    p[:4] = 6.0                       # four hot experts, all on device 0 at N=8
    p /= p.sum()
    x = synth_iid(8192 * 40, p)
    r = A.three_curves(x, 8192, 8, n_experts=E, n_shuffles=20,
                       rng=np.random.default_rng(3))
    print("\n  skewed marginal: S_shuf=%.4f  N*maxP=%.4f  S_iid=%.4f"
          % (r["S_shuf_mean"], r["N_maxP"], r["S_iid"]))
    assert abs(r["S_shuf_mean"] - r["N_maxP"]) < 0.03
    assert r["S_shuf_mean"] > 1.2, "expected a real floor from skewed marginals"
    print("PASS S_shuf converges to the marginal+placement floor N*max_j P_j")


def test_bad_expert_id_raises():
    """
    A stray id >= n_experts used to be absorbed silently: the offset-bincount
    trick in expert_counts binned it into the NEXT micro-batch, so the counts
    no longer summed to B*K and S was computed off corrupted loads.  Both the
    too-large and the negative case must raise instead.
    """
    x = synth_iid(1024).astype(np.int64)
    x[0, 0] = E + 3                       # stray id in the FIRST micro-batch
    for bad in (E + 3, -1):
        x[0, 0] = bad
        try:
            A.expert_counts(x, 512, n_experts=E)
        except ValueError as e:
            assert "out of range" in str(e)
        else:
            raise AssertionError("expert_counts accepted id %d" % bad)
    # and the same id is still fine when n_experts is actually that large
    x[0, 0] = E + 3
    c = A.expert_counts(x, 512, n_experts=E + 8)
    assert (c.sum(axis=1) == 512 * K).all()
    print("PASS out-of-range expert ids raise instead of bleeding across micro-batches")


def test_shuffle_preserves_marginals_exactly():
    x = synth_iid(10000)
    a = np.bincount(x.ravel().astype(np.int64), minlength=E)
    order = np.random.default_rng(4).permutation(len(x))
    b = np.bincount(x[order].ravel().astype(np.int64), minlength=E)
    assert (a == b).all()
    print("PASS shuffling preserves the expert marginal exactly")


if __name__ == "__main__":
    test_counts_conserve()
    test_straggler_bounds()
    test_bad_expert_id_raises()
    test_shuffle_preserves_marginals_exactly()
    test_shuffle_matches_iid()
    test_shuffle_kills_correlation()
    test_skewed_marginals()
    print("\nALL TESTS PASS")
