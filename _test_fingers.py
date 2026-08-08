"""
Sanity test for the palm-scale _fingers_up logic.
Simulates landmark positions for 0–5 fingers on right and left hands,
and two-hand totals for 6–10.

Run:  python _test_fingers.py
"""
import sys
sys.path.insert(0, r'd:\gesture_control_project')


class FakeLM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def make_right_hand(fingers_raised):
    """
    Right hand (as seen in mirrored frame).
    Thumb extends to the LEFT (lower x) when raised.
    Index MCP at x=0.50, pinky MCP at x=0.62 → lateral vector points right.
    Curled tips are placed BELOW their MCPs (higher y = lower on screen).
    """
    lm = [None] * 21
    lm[0]  = FakeLM(0.50, 0.85)                                          # wrist
    lm[1]  = FakeLM(0.42, 0.78); lm[2] = FakeLM(0.38, 0.72)
    lm[3]  = FakeLM(0.34, 0.67)
    lm[4]  = FakeLM(0.26, 0.64) if 0 in fingers_raised else FakeLM(0.44, 0.68)
    lm[5]  = FakeLM(0.50, 0.65); lm[6]  = FakeLM(0.50, 0.55)
    lm[7]  = FakeLM(0.50, 0.46)
    lm[8]  = FakeLM(0.50, 0.30) if 1 in fingers_raised else FakeLM(0.50, 0.72)  # curled: below MCP
    lm[9]  = FakeLM(0.54, 0.60); lm[10] = FakeLM(0.54, 0.50)            # palm anchor
    lm[11] = FakeLM(0.54, 0.41)
    lm[12] = FakeLM(0.54, 0.25) if 2 in fingers_raised else FakeLM(0.54, 0.70)  # curled: below MCP
    lm[13] = FakeLM(0.58, 0.62); lm[14] = FakeLM(0.58, 0.52)
    lm[15] = FakeLM(0.58, 0.43)
    lm[16] = FakeLM(0.58, 0.27) if 3 in fingers_raised else FakeLM(0.58, 0.71)  # curled: below MCP
    lm[17] = FakeLM(0.62, 0.64); lm[18] = FakeLM(0.62, 0.55)
    lm[19] = FakeLM(0.62, 0.48)
    lm[20] = FakeLM(0.62, 0.33) if 4 in fingers_raised else FakeLM(0.62, 0.73)  # curled: below MCP
    return lm


def make_left_hand(fingers_raised):
    """
    Left hand (as seen in mirrored frame) — mirror of right hand.
    Index MCP at x=0.50, pinky MCP at x=0.38 → lateral vector points left.
    Thumb extends to the RIGHT (higher x) when raised.
    Curled tips are placed BELOW their MCPs (higher y = lower on screen).
    """
    lm = [None] * 21
    lm[0]  = FakeLM(0.50, 0.85)                                          # wrist
    lm[1]  = FakeLM(0.58, 0.78); lm[2] = FakeLM(0.62, 0.72)
    lm[3]  = FakeLM(0.66, 0.67)
    lm[4]  = FakeLM(0.74, 0.64) if 0 in fingers_raised else FakeLM(0.56, 0.68)
    lm[5]  = FakeLM(0.50, 0.65); lm[6]  = FakeLM(0.50, 0.55)
    lm[7]  = FakeLM(0.50, 0.46)
    lm[8]  = FakeLM(0.50, 0.30) if 1 in fingers_raised else FakeLM(0.50, 0.72)  # curled: below MCP
    lm[9]  = FakeLM(0.46, 0.60); lm[10] = FakeLM(0.46, 0.50)            # palm anchor
    lm[11] = FakeLM(0.46, 0.41)
    lm[12] = FakeLM(0.46, 0.25) if 2 in fingers_raised else FakeLM(0.46, 0.70)  # curled: below MCP
    lm[13] = FakeLM(0.42, 0.62); lm[14] = FakeLM(0.42, 0.52)
    lm[15] = FakeLM(0.42, 0.43)
    lm[16] = FakeLM(0.42, 0.27) if 3 in fingers_raised else FakeLM(0.42, 0.71)  # curled: below MCP
    lm[17] = FakeLM(0.38, 0.64); lm[18] = FakeLM(0.38, 0.55)
    lm[19] = FakeLM(0.38, 0.48)
    lm[20] = FakeLM(0.38, 0.33) if 4 in fingers_raised else FakeLM(0.38, 0.73)  # curled: below MCP
    return lm


# ── Import controller without running __init__ ─────────────────────────────────
import gesture_control as gc

ctrl = object.__new__(gc.GestureController)

PASS = FAIL = 0


def test(label, lm, expected, handedness="Right"):
    global PASS, FAIL
    got = ctrl._fingers_up(lm, handedness)
    ok  = got == expected
    PASS += ok
    FAIL += not ok
    tag = "PASS" if ok else f"FAIL  (expected {expected})"
    print(f"  {tag:30s}  {label}: {got}")


print("\n── Right-hand tests ─────────────────────────────────────")
test("0 fingers — fist",          make_right_hand(set()),          0)
test("1 finger  — index",         make_right_hand({1}),            1)
test("2 fingers — index+middle",  make_right_hand({1, 2}),         2)
test("3 fingers — i+m+r",         make_right_hand({1, 2, 3}),      3)
test("4 fingers — no thumb",      make_right_hand({1, 2, 3, 4}),   4)
test("5 fingers — all",           make_right_hand({0,1,2,3,4}),    5)
test("thumb only",                make_right_hand({0}),            1)
test("pinky only",                make_right_hand({4}),            1)
test("ring+pinky",                make_right_hand({3, 4}),         2)

print("\n── Left-hand tests ──────────────────────────────────────")
test("0 fingers — fist",          make_left_hand(set()),           0, "Left")
test("1 finger  — index",         make_left_hand({1}),             1, "Left")
test("2 fingers — index+middle",  make_left_hand({1, 2}),          2, "Left")
test("3 fingers — i+m+r",         make_left_hand({1, 2, 3}),       3, "Left")
test("4 fingers — no thumb",      make_left_hand({1, 2, 3, 4}),    4, "Left")
test("5 fingers — all",           make_left_hand({0,1,2,3,4}),     5, "Left")
test("thumb only",                make_left_hand({0}),             1, "Left")

print("\n── Two-hand totals ──────────────────────────────────────")
combos = [
    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
    (4, 2), (4, 3), (3, 3), (4, 1), (3, 4),
]
for a, b in combos:
    # Build finger sets: use non-thumb fingers first, add thumb if needed
    def finger_set(n):
        # Use fingers 1..4 first (index→pinky), then thumb (0) if n==5
        base = list(range(1, min(n+1, 5)))   # up to 4 non-thumb fingers
        if n == 5:
            base.append(0)
        return set(base)

    lmA = make_right_hand(finger_set(a))
    lmB = make_left_hand(finger_set(b))
    tA  = ctrl._fingers_up(lmA, "Right")
    tB  = ctrl._fingers_up(lmB, "Left")
    total = tA + tB
    exp   = a + b
    ok    = total == exp
    PASS += ok
    FAIL += not ok
    tag   = "PASS" if ok else f"FAIL (expected {exp}, R={tA} L={tB})"
    print(f"  {tag:45s}  {a}+{b} = {total}")

print(f"\n  Results: {PASS} passed, {FAIL} failed\n")
