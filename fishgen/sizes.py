"""Real adult sizes of every species preset, from published measurements.

Each entry is a typical adult (``typical``) and the range most adults fall in (``low`` to
``high``), all in centimetres. Records and freak giants are left out: an individual's size is
drawn from a triangular distribution over that range, peaking at the typical size.

What is measured depends on the animal, the way biologists measure it:

- fish, sharks, whales and dolphins: total length, snout to tail tip
- rays: disc width, wingtip to wingtip
- squid and cuttlefish: total length, fin tip to arm tip (giant squid: to the tip of the two
  long tentacles)
- octopus: total length, mantle tip to arm tip
- sea urchins: diameter of the shell (test), without the spines

Species whose sexes differ a lot in size (orcas, sperm whales) have a range per sex.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Size:
    typical: float
    low: float
    high: float
    source: str

    def draw(self, r: random.Random) -> float:
        return r.triangular(self.low, self.high, self.typical)


S = Size

SIZES: dict[str, Size] = {
    # ------------------------------------------------------------------ fish
    "Goldfish": S(18, 12, 28, "common/fancy goldfish, adults 10-30 cm (aquariumscience.org, hepper.com)"),
    "Clownfish": S(8.5, 7, 11, "Amphiprion ocellaris, most adults 7-10 cm, max 11 cm (Florida Museum)"),
    "Angelfish": S(13, 10, 15, "Pterophyllum scalare, max 15 cm TL, taller than long with fins (FishBase)"),
    "Tuna": S(150, 100, 200, "yellowfin, common 150 cm FL, max 239 cm (FishBase)"),
    "Eel": S(70, 50, 100, "European eel, adults normally 60-80 cm, rarely over 1 m"),
    "Pufferfish": S(25, 18, 33, "dog-faced / blackspotted puffer, max 33 cm TL (FishBase)"),
    "Betta": S(6, 5, 7, "Betta splendens, max 6.5 cm TL (FishBase); 6-6.3 cm average"),
    "Neon tetra": S(3.5, 3, 4, "Paracheirodon innesi, adults 3-4 cm"),
    "Koi": S(60, 45, 80, "average adult 50-60 cm, large fish 76-91 cm"),
    "Discus": S(15, 12, 20, "Symphysodon, 12-15 cm, 15-20 cm TL in captivity (Wikipedia, UF/IFAS)"),
    "Mackerel": S(30, 20, 45, "Atlantic mackerel, common 30 cm FL, max 60 cm (FishBase)"),
    "Piranha": S(28, 20, 35, "red-bellied piranha, rarely over 35 cm (Wikipedia, Smithsonian Zoo)"),
    "Yellow tang": S(17, 13, 20, "Zebrasoma flavescens, about 20 cm (Britannica)"),
    "Blue tang": S(25, 18, 31, "regal blue tang, max 31 cm TL (Reef Life Survey, TFH)"),
    "Rainbow trout": S(55, 40, 76, "adults 51-76 cm"),
    "Atlantic salmon": S(80, 65, 100, "sea-run adults 71-102 cm"),
    "Great barracuda": S(120, 80, 160, "common 140 cm TL, max 200 cm (FishBase)"),
    "Ocean sunfish": S(180, 140, 250, "average adult 1.8 m long, 2.5 m fin to fin (Wikipedia)"),
    # ------------------------------------------------------------------ sharks
    "Tiger shark": S(375, 325, 450, "average adult 3.25-4.25 m, max about 5.5 m"),
    "Great white shark": S(450, 340, 550, "males 3.4-4.0 m, females 4.6-4.9 m, max about 6.1 m"),
    "Hammerhead shark": S(430, 330, 550, "great hammerhead, average 4.6 m, max 6.1 m (Wikipedia, MarAlliance)"),
    "Whale shark": S(1000, 800, 1400, "males level off at 8-9 m, females about 14 m (AIMS 2020)"),
    "Mako shark": S(250, 200, 320, "shortfin mako, males 2.0-2.15 m, females 2.75-2.9 m (Florida Museum)"),
    "Blacktip reef shark": S(140, 100, 180, "most adults under 1.6 m, max 2 m (FishBase, sharks.org)"),
    "Blue shark": S(220, 170, 310, "most adults 1.7-2.2 m, a few near 3.8 m (Britannica)"),
    "Nurse shark": S(220, 190, 300, "mean adult 1.93 m (ADW), 2.3-2.45 m (Florida Museum)"),
    # ------------------------------------------------------------------ whales & dolphins
    "Orca": S(700, 560, 850, "females average about 7 m, max 8.5 m (ADW, Mammals of Texas)"),
    "Orca male": S(800, 680, 980, "males average 8-8.2 m, max 9.8 m (ADW, Mammals of Texas)"),
    "Bottlenose dolphin": S(280, 230, 350, "adults 2.5-3.5 m in most of the world"),
    "Common dolphin": S(200, 170, 240, "short-beaked common dolphin, 1.7-2.4 m"),
    "Beluga": S(420, 330, 550, "males 4-5.5 m, females 3-4.1 m (MarineBio, COSEWIC)"),
    "Humpback whale": S(1400, 1250, 1600, "adults 14-15 m, females 1-1.5 m longer than males (Wikipedia, UBC)"),
    "Sperm whale": S(1100, 1000, 1200, "females mature at 10.6-11 m, rarely over 12 m (Wikipedia)"),
    "Sperm whale male": S(1600, 1450, 1800, "males mature at 15-16 m, large ones 18-19 m (Wikipedia)"),
    # ------------------------------------------------------------------ rays (disc width)
    "Cownose ray": S(90, 75, 110, "Atlantic cownose ray, 76-114 cm across; mature females average 96 cm"),
    "Manta ray": S(350, 280, 480, "reef manta 3-3.5 m across (Manta Trust), oceanic 4-5 m"),
    "Spotted eagle ray": S(150, 110, 230, "common width 1.2-1.5 m, max about 3.3 m (Georgia Aquarium)"),
    "Stingray": S(90, 55, 150, "southern stingray, males to 67 cm, females to 150 cm across (Wikipedia)"),
    "Bluespotted stingray": S(25, 20, 35, "bluespotted ribbontail, max 30-35 cm across (Florida Museum)"),
    # ------------------------------------------------------------------ squid & cuttlefish
    "Squid": S(40, 25, 60, "longfin inshore squid, mantle 10-30 cm plus arms and tentacles (SeaLifeBase)"),
    "Giant squid": S(1000, 750, 1300, "females to 12-13 m, males to 10 m with tentacles (Wikipedia)"),
    "Cuttlefish": S(30, 20, 45, "common cuttlefish, mantle commonly 15-25 cm plus arms (SeaLifeBase)"),
    # ------------------------------------------------------------------ octopus
    "Octopus": S(60, 35, 100, "common octopus, commonly 30-50 cm, max 1-1.3 m total length"),
    "Blue-ringed octopus": S(15, 12, 22, "body 4-6 cm, arms 7-10 cm, about 20 cm arm span"),
    "Dumbo octopus": S(25, 20, 30, "usually 20-30 cm (Aquarium of the Pacific)"),
    # ------------------------------------------------------------------ urchins (shell diameter)
    "Purple sea urchin": S(7, 5, 9, "Strongylocentrotus purpuratus, test to about 9 cm"),
    "Long-spined urchin": S(7, 5, 9, "Diadema, test about 7 cm, max 8.4 cm; spines to 30 cm"),
    "Red pencil urchin": S(8, 6, 10, "Heterocentrotus mamillatus, test 8-10 cm"),
    "Green sea urchin": S(5, 4, 8, "Strongylocentrotus droebachiensis, typically about 5 cm, max 8.7 cm"),
}


def size_of(species: str) -> Size | None:
    return SIZES.get(species)
