#!/usr/bin/env python3
"""One definition of how RIR comparison curves are drawn.

Every figure in this programme plots the same three things -- a reference, an
averaged estimate and a jointly conditioned one -- and each was styled where it
happened to be written. That produced two defects independently, twice each:

* the reference was drawn UNDER the estimates. A good estimate tracks the target
  almost everywhere, so the reference was invisible exactly where a reader needs
  to compare them;
* the joint estimate was drawn over the averaged one. On harder examples its
  amplitude is several times larger, so it erased the curve underneath.

Both are ordering problems, so the order lives here as named constants rather
than as a number typed at each call site.

    reference   black, dashed, on top          Z_REFERENCE
    averaged    blue, solid, middle            Z_AVERAGE
    joint       mid grey, dashed, bottom       Z_JOINT

Grey rather than red: red beside a black dashed reference reads as an alarm
colour and competes with it for attention. The joint curve is also translucent,
so where the two overlap the one beneath shows through instead of relying on
z-order alone.
"""
from __future__ import annotations

#: Bottom to top.
Z_JOINT, Z_AVERAGE, Z_REFERENCE = 2, 3, 5

#: Dash patterns chosen to stay distinguishable from each other: the reference
#: is long-dashed, the joint estimate finely dashed.
REFERENCE_DASH = (0, (4, 2))
JOINT_DASH = (0, (1.6, 1.4))

MID_GREY = "#7f7f7f"
AVERAGE_BLUE = "#1f77b4"

#: Spread as **kwargs into a plot call.
REFERENCE = dict(color="black", linestyle=REFERENCE_DASH, zorder=Z_REFERENCE)
AVERAGE = dict(color=AVERAGE_BLUE, zorder=Z_AVERAGE, alpha=.95)
JOINT = dict(color=MID_GREY, linestyle=JOINT_DASH, zorder=Z_JOINT, alpha=.65)

#: Spheres studio shorthand, expanded. Vcl is Violoncello (cello), Vla is Viola,
#: and Main L/C/R is the main array rather than three rooms.
POSITION = {
    "Vcl": "cello", "Vla": "viola", "Vln_1": "1st violin", "Vln_2": "2nd violin",
    "DBass_1": "double bass 1", "DBass_2": "double bass 2", "Flute": "flute",
    "Oboe": "oboe", "Clarinet": "clarinet", "Bassoon": "bassoon", "Horns": "horns",
    "Trumpets": "trumpets", "Trombones": "trombones", "Tuba": "tuba",
    "Timpani": "timpani", "Bass_Drum": "bass drum", "Harp": "harp",
}
MICROPHONE = {"Main_L": "main left", "Main_C": "main centre", "Main_R": "main right"}


def readable(mic: str, position: str) -> str:
    """``"Main_C", "Vcl"`` -> ``"cello position, main centre mic"``."""
    return (f"{POSITION.get(position, position)} position, "
            f"{MICROPHONE.get(mic, mic)} mic")
