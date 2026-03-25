from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


Q16 = 1 << 16
Q30 = 1 << 30
T_MIN_Q16 = 1
Z_MIN_Q16 = -(12 << 16)

E_Q30: Tuple[int, ...] = (
    1073741824,
    395007542,
    145315154,
    53458458,
    19666268,
    7234816,
    2661540,
    979126,
    360200,
    132510,
    48748,
    17933,
    6597,
)


@dataclass(frozen=True)
class DecodeStepResult:
    y: int
    Ws: int
    R: int


def _clamp_u32(x: int) -> int:
    return max(0, min(0xFFFFFFFF, x))


def _clamp_i32(x: int) -> int:
    return max(-0x80000000, min(0x7FFFFFFF, x))


def _clamp_i64(x: int) -> int:
    return max(-0x8000000000000000, min(0x7FFFFFFFFFFFFFFF, x))


def _mul_q30(a_q30: int, b_q30: int) -> int:
    return (a_q30 * b_q30) >> 30


def _exp_poly5_q16_16_to_q30(r_q16: int) -> int:
    r_q30 = r_q16 << 14

    r2 = _mul_q30(r_q30, r_q30)
    r3 = _mul_q30(r2, r_q30)
    r4 = _mul_q30(r3, r_q30)
    r5 = _mul_q30(r4, r_q30)

    p = Q30
    p += r_q30
    p += r2 // 2
    p += r3 // 6
    p += r4 // 24
    p += r5 // 120

    if p < 0:
        return 0
    if p > Q30:
        return Q30
    return int(p)


def decode_step(*, K: int, top_k: int, top_p_q16: int, T_q16: int, token_id: Sequence[int], logit_q16: Sequence[int], U_t: int) -> DecodeStepResult:
    if K <= 0:
        raise ValueError("K must be positive")
    if len(token_id) != K or len(logit_q16) != K:
        raise ValueError("token_id and logit_q16 must have length K")
    if not (1 <= top_k <= K):
        raise ValueError("top_k must satisfy 1 <= top_k <= K")
    if not (0 < top_p_q16 <= Q16):
        raise ValueError("top_p_q16 must satisfy 0 < top_p_q16 <= 1.0")

    T_clamped = max(int(T_q16), T_MIN_Q16)

    scaled: List[int] = []
    for l in logit_q16:
        num = int(l) << 16
        s = num // T_clamped
        scaled.append(_clamp_i64(s))

    items = list(zip(token_id, scaled))
    items.sort(key=lambda x: (-x[1], x[0]))

    sid = [int(tid) for tid, _ in items]
    slog = [int(slogit) for _, slogit in items]

    k = int(top_k)
    m = slog[0]

    w: List[int] = [0] * K
    for i in range(k):
        z = slog[i] - m
        if z < Z_MIN_Q16:
            z = Z_MIN_Q16

        neg_z = -z
        n = neg_z >> 16
        if n < 0:
            n = 0
        if n > 12:
            n = 12

        r = z + (n << 16)

        p = _exp_poly5_q16_16_to_q30(r)
        wi = _mul_q30(E_Q30[n], p)
        if wi < 0:
            wi = 0
        w[i] = int(wi)

    Wk = int(sum(w[:k]))
    TH = (int(top_p_q16) * Wk) >> 16

    prefix = 0
    s = 1
    for i in range(k):
        prefix += w[i]
        if prefix >= TH:
            s = i + 1
            break

    Ws = int(sum(w[:s]))

    U = int(U_t) & 0xFFFFFFFFFFFFFFFF
    R = (U * Ws) >> 64

    prefix2 = 0
    j = 0
    for i in range(s):
        prefix2 += w[i]
        if prefix2 > R:
            j = i
            break

    y = int(sid[j])

    return DecodeStepResult(y=y, Ws=Ws, R=int(R))
