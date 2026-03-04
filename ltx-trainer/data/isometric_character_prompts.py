#!/usr/bin/env python3
"""
Isometric 3D Character-Action Prompt Generator

Every prompt features a specific character doing a specific thing in a specific place.
Better temporal signal for SCD training than ambient/environmental prompts.

Usage:
    python data/isometric_character_prompts.py --count 300 > data/isometric_character_prompts.txt
    python data/isometric_character_prompts.py --count 300 --jsonl data/isometric_character_prompts.jsonl
"""

import argparse
import json
import random
from pathlib import Path

# =============================================================================
# CHARACTERS — who is doing the thing
# =============================================================================

CHARACTERS = [
    # --- Fantasy ---
    "a tiny wizard in a purple robe",
    "a dwarf blacksmith with a braided beard",
    "an elf ranger with a bow",
    "a goblin merchant carrying a sack",
    "a knight in shining armor",
    "a fairy with glowing wings",
    "a necromancer in tattered robes",
    "a barbarian warrior with a huge axe",
    "a halfling thief in a cloak",
    "a druid with antlers growing from their head",
    "a dragon hatchling the size of a cat",
    "a skeleton warrior with a rusty sword",
    "a troll bridge keeper",
    "a phoenix perched on a stone pillar",
    "a centaur archer",
    # --- Sci-Fi ---
    "a small robot with tank treads",
    "a space marine in power armor",
    "a cyborg mechanic with a glowing eye",
    "an alien diplomat with four arms",
    "a drone repair technician",
    "a holographic AI assistant",
    "a bounty hunter in a jetpack",
    "a android bartender",
    "a tiny mech pilot climbing into a cockpit",
    "a space pirate with a laser cutlass",
    # --- Modern / Everyday ---
    "a chef in a tall white hat",
    "a firefighter in full gear",
    "a postal worker with a heavy bag",
    "a street musician with a guitar",
    "a skateboarder in a hoodie",
    "a construction worker in a hard hat",
    "a barista with an apron",
    "a detective in a trench coat",
    "a park ranger with binoculars",
    "a delivery driver with stacked boxes",
    "a painter covered in paint splatters",
    "a mechanic wiping hands on a rag",
    "a teacher writing on a chalkboard",
    "a photographer with a large camera",
    "a fisherman in rubber boots",
    # --- Historical ---
    "a Viking raider with a round shield",
    "a Roman centurion in a red cape",
    "a samurai in lacquered armor",
    "a pirate captain with a tricorn hat",
    "a Wild West sheriff with a star badge",
    "a pharaoh with a golden headdress",
    "a medieval monk with a quill pen",
    "a Spartan hoplite with a bronze shield",
    "a musketeer with a feathered hat",
    "a ninja crouching in the shadows",
    # --- Animals / Creatures ---
    "a fat orange cat",
    "a golden retriever puppy",
    "a penguin in a tiny top hat",
    "a raccoon standing on hind legs",
    "a fox wearing a scarf",
    "an owl perched on a bookshelf",
    "a bear in a lumberjack outfit",
    "a frog sitting on a lily pad throne",
    "a mouse with a tiny backpack",
    "a parrot on a pirate's shoulder",
]

# =============================================================================
# ACTIONS — what the character is actively doing (strong temporal signal)
# =============================================================================

ACTIONS = [
    # --- Physical / Movement ---
    "walks along a cobblestone path",
    "runs across the scene chasing a butterfly",
    "climbs a tall ladder to the roof",
    "jumps over a puddle",
    "slides down a banister",
    "pushes a heavy cart uphill",
    "carries a large crate across the room",
    "sweeps the floor with a broom",
    "hammers a nail into a wooden plank",
    "chops firewood with an axe",
    "digs a hole in the ground with a shovel",
    "pulls a lever on a large machine",
    "rolls a barrel across the floor",
    "drags a fishing net from the water",
    "stacks boxes into a neat pile",
    # --- Craft / Work ---
    "stirs a bubbling cauldron with a wooden spoon",
    "forges a sword on a glowing anvil",
    "paints on a large canvas with broad strokes",
    "kneads bread dough on a flour-dusted table",
    "sews fabric on a sewing machine",
    "polishes a gemstone under a magnifying glass",
    "assembles tiny clockwork gears at a workbench",
    "welds metal parts with bright sparks flying",
    "carves a wooden figure with a small knife",
    "mixes potions in glass flasks",
    "types rapidly on a glowing keyboard",
    "reads a large map spread across a table",
    "tunes a violin by turning the pegs",
    "pours molten metal into a mold",
    "arranges flowers in a tall vase",
    # --- Social / Expressive ---
    "waves to someone across the street",
    "dances a jig in the town square",
    "argues with a shopkeeper over prices",
    "pets a small animal on the ground",
    "high-fives another tiny character",
    "sits by a campfire roasting marshmallows",
    "stands on a soapbox giving a speech",
    "plays catch with a friend",
    "arm-wrestles an opponent at a tavern table",
    "hands a wrapped gift to another character",
    # --- Combat / Action ---
    "practices sword swings at a training dummy",
    "fires an arrow at a distant target",
    "blocks an attack with a raised shield",
    "casts a glowing spell from outstretched hands",
    "throws a lasso around a post",
    "dodges rolling boulders",
    "parries a blow and counterattacks",
    "charges forward on horseback",
    # --- Mundane / Cozy ---
    "waters plants with a small watering can",
    "hangs laundry on a clothesline",
    "flips pancakes in a frying pan",
    "pours tea from a kettle into a cup",
    "feeds breadcrumbs to birds on the ground",
    "opens a treasure chest with a rusty key",
    "lights a lantern and holds it up",
    "blows out candles on a birthday cake",
    "eats a large sandwich while sitting on a bench",
    "writes in a journal with a feather quill",
    "shelves books on a tall bookcase using a step ladder",
    "unrolls a scroll and reads it aloud",
]

# =============================================================================
# LOCATIONS — where it happens
# =============================================================================

LOCATIONS = [
    # --- Fantasy ---
    "in a wizard's tower filled with floating books",
    "inside a dragon's treasure cave",
    "in an enchanted forest clearing with glowing mushrooms",
    "on a floating island connected by chain bridges",
    "inside a dwarven mining hall with gem-studded walls",
    "in an elf village built into giant tree trunks",
    "inside a ruined temple overgrown with vines",
    "in a witch's cottage with a smoking chimney",
    "on a pirate ship deck at sea",
    "in a gladiator arena with stone columns",
    "at a fairy ring surrounded by tiny mushroom houses",
    "inside a crystal cave with glowing formations",
    "on a castle battlement overlooking a moat",
    "in an alchemist's basement laboratory",
    "at a magical marketplace with enchanted wares",
    # --- Urban / Modern ---
    "in a busy coffee shop with steaming espresso machines",
    "on a rooftop garden with city skyline behind",
    "inside a retro arcade with glowing cabinet screens",
    "at a street food market with sizzling grills",
    "in a cluttered mechanic's garage",
    "inside a cozy bookstore with floor-to-ceiling shelves",
    "at a basketball court under floodlights",
    "in a recording studio with mixing boards",
    "inside a barbershop with vintage chairs",
    "at a train platform as a locomotive arrives",
    "in a neon-lit ramen shop at night",
    "inside a fire station with a red engine",
    "at a rooftop pool party with string lights",
    "in a greenhouse bursting with tropical plants",
    "inside an art gallery with sculptures on pedestals",
    # --- Sci-Fi ---
    "on a space station observation deck with stars outside",
    "inside a cyberpunk alley with holographic ads",
    "in a robot assembly factory with conveyor belts",
    "at a Mars colony outpost with red dust dunes",
    "inside a starship bridge with blinking consoles",
    "in a neon underground hacker den",
    "at a teleportation hub with swirling energy portals",
    "inside a cryogenic sleep chamber facility",
    "on an asteroid mining platform",
    "in a futuristic hospital with floating med-bots",
    # --- Nature ---
    "beside a mountain waterfall with a rocky pool",
    "in a bamboo forest with shafts of sunlight",
    "at a desert oasis with palm trees and a tent",
    "on a tropical beach with turquoise water",
    "in a snowy pine forest with a frozen lake",
    "at the edge of a volcano with glowing lava below",
    "in a cherry blossom garden with a stone bridge",
    "beside a river with a working water mill",
    "in a sunflower field stretching to the horizon",
    "at a coral reef visible through crystal water",
    # --- Historical ---
    "in a medieval blacksmith's forge with bellows",
    "inside a Viking longhouse with a central fire pit",
    "at a Roman bathhouse with mosaic tile floors",
    "inside a Wild West saloon with swinging doors",
    "in a Japanese tea house with paper screens",
    "at an Egyptian marketplace near the pyramids",
    "inside a Renaissance painter's studio",
    "on the deck of a tall ship during golden hour",
    "in a 1920s speakeasy with jazz instruments on stage",
    "at a medieval jousting tournament field",
    # --- Cozy / Interior ---
    "in a cluttered inventor's workshop with gadgets",
    "inside a bakery with bread cooling on racks",
    "in a pottery studio with clay on the wheel",
    "inside a record store with vinyl bins and posters",
    "in a tiny sushi restaurant with a conveyor belt",
    "inside a candlelit wine cellar with oak barrels",
    "in a woodworking shop with sawdust everywhere",
    "at a campsite with a tent and crackling fire",
    "inside a model train room with tiny landscapes",
    "in a cozy cabin kitchen with a wood-burning stove",
]

# =============================================================================
# LIGHTING — sets mood (optional)
# =============================================================================

LIGHTING = [
    "warm golden-hour lighting",
    "cool blue twilight atmosphere",
    "dramatic spotlight from above",
    "soft pastel colors",
    "rich saturated tones",
    "moody volumetric fog",
    "crisp morning sunlight",
    "cozy warm glow from windows",
    "neon-lit nighttime ambiance",
    "autumn orange and red palette",
    "snowy winter atmosphere",
    "rainy day with wet reflections",
    "sunset casting long shadows",
    "candlelit warm ambiance",
    "overcast diffused daylight",
]

TEMPLATE = (
    "Isometric 3D miniature diorama, static camera. "
    "{character} {action} {location}. "
    "{lighting}. Tilt-shift depth of field. No camera movement."
)


def generate(count: int = 300, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    results = []
    used = set()

    for i in range(count):
        for _ in range(500):
            c = rng.choice(CHARACTERS)
            a = rng.choice(ACTIONS)
            l = rng.choice(LOCATIONS)
            key = (c, a, l)
            if key not in used:
                used.add(key)
                break

        light = rng.choice(LIGHTING)
        prompt = TEMPLATE.format(
            character=c, action=a, location=l, lighting=light,
        )
        results.append({
            "id": i,
            "prompt": prompt,
            "character": c,
            "action": a,
            "location": l,
            "lighting": light,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl", type=str, default=None, help="Also save JSONL")
    args = parser.parse_args()

    prompts = generate(args.count, args.seed)

    # Stats
    from collections import Counter
    print(f"# Generated {len(prompts)} character-action isometric prompts", file=__import__('sys').stderr)
    print(f"# Unique characters: {len(set(p['character'] for p in prompts))}/{len(CHARACTERS)}", file=__import__('sys').stderr)
    print(f"# Unique actions:    {len(set(p['action'] for p in prompts))}/{len(ACTIONS)}", file=__import__('sys').stderr)
    print(f"# Unique locations:  {len(set(p['location'] for p in prompts))}/{len(LOCATIONS)}", file=__import__('sys').stderr)
    print(f"# Combos possible:   {len(CHARACTERS)*len(ACTIONS)*len(LOCATIONS):,}", file=__import__('sys').stderr)

    # Print prompts to stdout
    for p in prompts:
        print(p["prompt"])

    # Optionally save JSONL
    if args.jsonl:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl, "w") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")
        print(f"# Saved JSONL to {args.jsonl}", file=__import__('sys').stderr)


if __name__ == "__main__":
    main()
