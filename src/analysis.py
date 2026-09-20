"""
Offline analysis of a routing trace.  Pure numpy, no model, no torch.

Definitions follow the scope note:

  n_e   assignments to expert e within a micro-batch of B tokens; sum_e n_e = B*k
  p_e   n_e / (B*k)
  M     expert-level max/mean load ratio  = E * max_e p_e     (the capacity factor)
  L_j   device load = sum over the E/N experts placed on device j
  S     straggler factor = max_j L_j / (B*k/N) = N * max_j P_j

Three curves, all as a function of micro-batch size B:
  S_obs   contiguous micro-batches, tokens in natural order   (what a server sees)
  S_shuf  same assignment multiset, tokens shuffled across micro-batches
          (kills token-to-token correlation, preserves expert marginals exactly)
  S_iid   closed form under independent sampling from the corpus marginal
"""
import numpy as np

# ----------------------------------------------------------------------------
# core counting
# ----------------------------------------------------------------------------


def expert_counts(topk_layer, B, order=None, n_experts=32):
    """
    topk_layer : (T, K) int, expert ids for one layer
    returns    : (n_micro, E) int64 counts per micro-batch
    """
    T, K = topk_layer.shape
    if order is not None:
        topk_layer = topk_layer[order]
    n_micro = T // B
    if n_micro == 0:
        return np.zeros((0, n_experts), dtype=np.int64)
    v = topk_layer[: n_micro * B].reshape(n_micro, B * K).astype(np.int64)
    # bincount per row without a python loop
    offs = (np.arange(n_micro, dtype=np.int64) * n_experts)[:, None]
    flat = np.bincount((v + offs).ravel(), minlength=n_micro * n_experts)
    return flat.reshape(n_micro, n_experts)


def device_loads(counts, N):
    """Contiguous placement: experts [j*E/N, (j+1)*E/N) live on device j."""
    n_micro, E = counts.shape
    if E % N:
        raise ValueError("E=%d not divisible by N=%d" % (E, N))
    return counts.reshape(n_micro, N, E // N).sum(axis=2)


def straggler(counts, N):
    """S = max_j L_j / (B*k/N), one value per micro-batch."""
    loads = device_loads(counts, N)
    total = counts.sum(axis=1, keepdims=True)          # = B*k
    return loads.max(axis=1) / (total[:, 0] / N)


def expert_max_ratio(counts):
    """M = E * max_e p_e, one value per micro-batch."""
    E = counts.shape[1]
    return counts.max(axis=1) / (counts.sum(axis=1) / E)


def entropy_norm(counts):
    E = counts.shape[1]
    p = counts / counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    return h / np.log(E)


def gini(counts):
    p = np.sort(counts / counts.sum(axis=1, keepdims=True), axis=1)
    E = counts.shape[1]
    i = np.arange(1, E + 1)
    return (2.0 * (p * i).sum(axis=1)) / E - (E + 1.0) / E


def dead_experts(counts):
    return (counts == 0).sum(axis=1)


# ----------------------------------------------------------------------------
# the closed-form null
# ----------------------------------------------------------------------------


_EMAX_CACHE = {}


def expected_max_normal(N):
    """
    E[max of N iid standard normals], by quadrature.

    The scope note used the asymptotic z_N = sqrt(2 ln N).  That is an upper
    bound and a loose one at the N we care about: it overstates the true
    expected max by 1.42x (N=32) to 1.68x (N=2).  Using it would have made the
    iid null look LARGER than the measured shuffle control, which would have
    read as a spurious "better than independent" result.  See tests.
    """
    if N < 2:
        return 0.0
    if N in _EMAX_CACHE:
        return _EMAX_CACHE[N]
    x = np.linspace(-12.0, 14.0, 600001)
    phi = np.exp(-x ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
    Phi = np.concatenate([[0.0], np.cumsum((phi[1:] + phi[:-1]) / 2.0 * np.diff(x))])
    Phi /= Phi[-1]
    val = float(np.trapezoid(x * N * phi * Phi ** (N - 1), x))
    _EMAX_CACHE[N] = val
    return val


def s_iid(B, K, N, P=None, E=32):
    """
    Closed-form straggler factor under independent sampling from the corpus
    marginal -- i.e. the same expert popularity, but no token-to-token
    correlation whatsoever.

        S_iid ~= N*max_j P_j  +  z_eff * N * sqrt( Pbar(1-Pbar) / (B*K) )

    with three corrections over the naive form, each verified against the
    shuffle control on synthetic iid data (test_analysis.py):

      1. z_eff uses E[max of N standard normals], not the asymptotic
         sqrt(2 ln N).  Dominant correction: 1.42x-1.68x.
      2. x sqrt(N/(N-1)) because the N device loads sum to exactly B*K, so
         they are negatively correlated rather than independent.
      3. x sqrt((E-K)/(E-1)) because each token picks K *distinct* experts.
         Sampling without replacement within a token shrinks the per-token
         variance below the binomial value.

    Together these bring the closed form within ~2% of the measured shuffle
    control across N in {4,8,16} and B in {128..8192}.
    """
    n = B * K
    if N <= 1:
        return 1.0
    if P is None:
        Pmax, Pbar = 1.0 / N, 1.0 / N
    else:
        P = np.asarray(P, dtype=float)
        Pmax, Pbar = float(P.max()), 1.0 / N
    z = expected_max_normal(N)
    z *= np.sqrt(N / (N - 1.0))            # sum-to-constant correction
    z *= np.sqrt(max(E - K, 1) / (E - 1.0))  # top-k without replacement
    return N * Pmax + z * N * np.sqrt(Pbar * (1.0 - Pbar) / n)


def s_iid_asymptotic(B, K, N):
    """
    The uncorrected form from the scope note, kept so the writeup can show how
    far off it is:  1 + sqrt(2 lnN) * sqrt((N-1)/(B*K)).
    """
    if N <= 1:
        return 1.0
    return 1.0 + np.sqrt(2.0 * np.log(N)) * np.sqrt((N - 1.0) / (B * K))


def s_iid_montecarlo(topk_layer, B, N, n_experts=32, n_draws=200, rng=None):
    """
    Independent check on the closed form with no analytics at all: resample
    whole top-k tuples i.i.d. WITH replacement from the empirical distribution
    of tuples.  This preserves the expert marginal and the within-token
    structure exactly, and destroys token-to-token correlation completely.
    """
    rng = rng or np.random.default_rng(0)
    T = topk_layer.shape[0]
    out = np.empty(n_draws)
    for i in range(n_draws):
        idx = rng.integers(0, T, size=B)
        c = expert_counts(topk_layer[idx], B, None, n_experts)
        out[i] = straggler(c, N)[0]
    return float(out.mean())


# ----------------------------------------------------------------------------
# the three-curve sweep
# ----------------------------------------------------------------------------


def three_curves(topk_layer, B, N, n_experts=32, n_shuffles=20, rng=None,
                 pcts=(5, 50, 95)):
    """
    Returns a dict of S_obs / S_shuf / S_iid statistics for one layer, one B, one N.
    """
    rng = rng or np.random.default_rng(0)
    T, K = topk_layer.shape

    c_obs = expert_counts(topk_layer, B, None, n_experts)
    if c_obs.shape[0] == 0:
        return None
    s_obs = straggler(c_obs, N)

    s_shuf_means = []
    s_shuf_all = []
    for _ in range(n_shuffles):
        order = rng.permutation(T)
        c = expert_counts(topk_layer, B, order, n_experts)
        s = straggler(c, N)
        s_shuf_means.append(s.mean())
        s_shuf_all.append(s)
    s_shuf_all = np.concatenate(s_shuf_all)

    # corpus-level marginal -> device marginal, for the non-uniform closed form
    tot = np.bincount(topk_layer.ravel().astype(np.int64), minlength=n_experts)
    p = tot / tot.sum()
    P = p.reshape(N, n_experts // N).sum(axis=1)

    return {
        "B": B, "N": N, "n_micro": int(c_obs.shape[0]),
        "S_obs_mean": float(s_obs.mean()),
        "S_obs_p5": float(np.percentile(s_obs, pcts[0])),
        "S_obs_p50": float(np.percentile(s_obs, pcts[1])),
        "S_obs_p95": float(np.percentile(s_obs, pcts[2])),
        "S_obs_max": float(s_obs.max()),
        "S_shuf_mean": float(np.mean(s_shuf_means)),
        "S_shuf_p5": float(np.percentile(s_shuf_all, pcts[0])),
        "S_shuf_p95": float(np.percentile(s_shuf_all, pcts[2])),
        "S_iid": float(s_iid(B, K, N, P, E=n_experts)),
        "S_iid_uniform": float(s_iid(B, K, N, None, E=n_experts)),
        "S_iid_asym": float(s_iid_asymptotic(B, K, N)),
        "N_maxP": float(N * P.max()),
        "M_obs_mean": float(expert_max_ratio(c_obs).mean()),
        "H_obs_mean": float(entropy_norm(c_obs).mean()),
        "G_obs_mean": float(gini(c_obs).mean()),
        "dead_obs_mean": float(dead_experts(c_obs).mean()),
    }


def adjacent_repeat_rate(topk_layer):
    """
    Mixtral's statistic: fraction of adjacent token pairs sharing their
    first-choice expert.  Random baseline is 1/E.  Included so the measurement
    here can be compared against the published number directly.
    NOTE: assumes column 0 of topk is the top-1 expert (torch.topk returns sorted).
    """
    a = topk_layer[:-1, 0]
    b = topk_layer[1:, 0]
    return float((a == b).mean())
