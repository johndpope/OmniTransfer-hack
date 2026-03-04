#!/usr/bin/env python3
"""
Isometric 3D Video Prompt Generator for SCD Training Data

Generates diverse isometric scene prompts for Grok API video generation.
Target: 300-500 unique clips for robust SCD LoRA training.

Usage:
    python data/isometric_prompts.py --count 300 --output prompts.jsonl
    python data/isometric_prompts.py --count 50 --category fantasy --output fantasy_prompts.jsonl
    python data/isometric_prompts.py --list-categories
"""

import argparse
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# LOCATIONS — organized by theme
# =============================================================================

LOCATIONS = {
    # --- Urban / Modern ---
    "urban": [
        "a busy city intersection with crosswalks and traffic lights",
        "a rooftop garden on a high-rise building",
        "a downtown coffee shop with outdoor seating",
        "a subway station platform with waiting commuters",
        "a food truck park with picnic tables",
        "a construction site with cranes and scaffolding",
        "a parking garage with multiple levels",
        "a basketball court in an urban park",
        "a laundromat with rows of washing machines",
        "a fire station with an open garage bay",
        "a barber shop interior with vintage chairs",
        "a convenience store with neon signs",
        "a bus stop shelter on a rainy street",
        "a skatepark with ramps and half-pipes",
        "a rooftop swimming pool overlooking the city",
        "a street market with colorful stalls",
        "an electronics repair shop",
        "a taxi stand outside a train station",
        "a flower shop with sidewalk displays",
        "a public library reading room",
    ],
    # --- Nature / Outdoor ---
    "nature": [
        "a mountain cabin beside a frozen lake",
        "a tropical waterfall with a rocky pool",
        "a bamboo forest with a winding path",
        "a desert oasis with palm trees and a tent",
        "a coral reef visible through crystal-clear water",
        "a volcanic island with lava flows reaching the sea",
        "a redwood forest floor with massive tree trunks",
        "a wildflower meadow with a wooden fence",
        "a river canyon with layered rock walls",
        "a glacier cave with blue ice formations",
        "a mangrove swamp with tangled roots",
        "a cherry blossom garden with a stone bridge",
        "a mushroom-covered forest clearing",
        "an alpine lake reflecting snow-capped peaks",
        "a coastal tide pool with sea creatures",
        "a savanna with acacia trees and tall grass",
        "a rice paddy terrace on a hillside",
        "a misty swamp with cypress trees",
        "a seaside cliff with crashing waves below",
        "a hot spring surrounded by snow",
    ],
    # --- Fantasy / Game ---
    "fantasy": [
        "a wizard's tower with floating bookshelves",
        "a dragon's treasure hoard in a cavern",
        "an enchanted forest with glowing mushrooms",
        "a floating island connected by chain bridges",
        "a dwarf mining outpost inside a mountain",
        "an elf treehouse village in ancient oaks",
        "a necromancer's dungeon with green torches",
        "a crystal cave with pulsating gem formations",
        "a pirate cove with a hidden ship dock",
        "a witch's cottage with a bubbling cauldron",
        "a ruined temple overgrown with vines",
        "a frost giant's throne room made of ice",
        "an underwater palace with coral columns",
        "a steampunk airship docking tower",
        "a fairy ring clearing with tiny houses",
        "a volcanic forge with molten metal channels",
        "an astral observatory on a floating platform",
        "a haunted graveyard with crooked tombstones",
        "a gladiator arena with sand and stone walls",
        "a time-frozen battlefield with suspended arrows",
    ],
    # --- Sci-Fi / Futuristic ---
    "scifi": [
        "a space station control room with holographic displays",
        "a Mars colony habitat dome with red dust outside",
        "a cyberpunk noodle bar under neon signs",
        "a robot assembly line in a clean factory",
        "a teleportation hub with glowing portals",
        "an alien marketplace on a desert planet",
        "a zero-gravity laboratory with floating equipment",
        "a mech repair bay with giant robot parts",
        "a terraforming station on a barren moon",
        "a quantum computer core with energy conduits",
        "a cryogenic sleep chamber facility",
        "a hyperloop transit station",
        "a drone delivery sorting center",
        "an android coffee shop serving synthetic humans",
        "a holographic training simulation room",
        "a space elevator ground terminal",
        "a bio-dome greenhouse on an asteroid",
        "a deep-sea research station with viewing ports",
        "a fusion reactor control room",
        "a VR arcade with players in motion chairs",
    ],
    # --- Historical / Period ---
    "historical": [
        "a medieval blacksmith's forge",
        "an ancient Egyptian marketplace near pyramids",
        "a Viking longhouse interior with a fire pit",
        "a Roman bathhouse with mosaic floors",
        "a samurai dojo with wooden training dummies",
        "a Wild West saloon with swinging doors",
        "a Renaissance artist's workshop",
        "a Prohibition-era speakeasy behind a bookshelf",
        "an ancient Greek amphitheater",
        "a Victorian-era steam train station",
        "a colonial-era apothecary shop",
        "a Mayan temple plaza with carved stelae",
        "a medieval castle kitchen with a spit roast",
        "an Ottoman-era bazaar with arched ceilings",
        "a Civil War field hospital tent",
        "a 1920s jazz club with a small stage",
        "a Ming Dynasty pottery workshop",
        "a Polynesian village with thatched huts",
        "an ancient Roman gladiator training ground",
        "a medieval monastery scriptorium",
    ],
    # --- Cozy / Interior ---
    "cozy": [
        "a Japanese ramen shop with a counter and stools",
        "a cluttered inventor's workshop with gadgets everywhere",
        "a bakery kitchen with bread in the oven",
        "a record store with vinyl bins and a turntable",
        "a tattoo parlor with flash sheets on the walls",
        "a plant-filled living room with a reading nook",
        "a camping scene with a tent and campfire",
        "a pottery studio with a spinning wheel",
        "a cozy attic bedroom under wooden beams",
        "a wine cellar with oak barrels and dim lighting",
        "a Christmas-decorated living room with a tree",
        "a sushi restaurant with a conveyor belt",
        "a woodworking shop with sawdust and tools",
        "a greenhouse nursery full of tropical plants",
        "an aquarium maintenance room with fish tanks",
        "a vintage camera shop with shelves of equipment",
        "a cozy tea house with paper lanterns",
        "a model train room with a detailed landscape",
        "a music practice room with instruments on stands",
        "a home brewing setup with copper kettles",
    ],
    # --- Industrial / Infrastructure ---
    "industrial": [
        "a power plant control room with analog gauges",
        "a shipyard dry dock with a vessel under repair",
        "a warehouse with forklifts and stacked pallets",
        "a water treatment facility with settling tanks",
        "a mining operation with conveyor belts",
        "a steel foundry with glowing molten metal",
        "a sawmill with logs on a river",
        "a hydroelectric dam interior",
        "a satellite dish array in a desert field",
        "a wind farm on rolling green hills",
        "a lighthouse keeper's station on a rocky coast",
        "a train marshaling yard with multiple tracks",
        "an oil rig platform in the ocean",
        "a recycling sorting facility",
        "a grain silo and elevator complex",
    ],
}

# =============================================================================
# ACTIONS — things happening in the scene (subtle, looping-friendly)
# =============================================================================

ACTIONS = {
    "ambient": [
        "smoke rises gently from a chimney",
        "leaves drift slowly in the breeze",
        "water flows steadily in a stream",
        "clouds drift across the sky above",
        "dust motes float in shafts of light",
        "snow falls softly on the ground",
        "fireflies blink in the shadows",
        "steam rises from a hot surface",
        "waves lap gently at the shore",
        "rain drips from awnings and gutters",
        "candles flicker on every surface",
        "banners flutter in a gentle wind",
        "bubbles rise from the water below",
        "gears turn and pistons pump rhythmically",
        "lights blink on control panels",
    ],
    "character_movement": [
        "a tiny figure walks along a path",
        "a small character carries supplies across the scene",
        "a miniature worker hammers at an anvil",
        "a tiny chef stirs a large pot",
        "small figures chat in a group",
        "a character sweeps the floor with a broom",
        "a miniature guard patrols back and forth",
        "a figure reads a book on a bench",
        "a small character tends a garden",
        "a figure climbs a ladder to a higher level",
        "a tiny musician plays an instrument",
        "a character feeds animals in a pen",
        "two small figures spar with wooden swords",
        "a figure pushes a cart loaded with goods",
        "a character fishes from a small dock",
    ],
    "vehicle_mechanical": [
        "a small train chugs along a track",
        "a miniature car drives down the road",
        "a tiny boat sails across the water",
        "a windmill turns slowly in the wind",
        "an elevator moves between floors",
        "conveyor belts carry tiny packages",
        "a waterwheel rotates beside a mill",
        "a crane lifts a crate at a construction site",
        "a drawbridge lowers over a moat",
        "a cable car glides along its line",
        "a helicopter lands on a rooftop pad",
        "a mining cart rolls along a track",
    ],
    "nature_dynamic": [
        "a waterfall cascades down rocky cliffs",
        "lava flows slowly down the mountainside",
        "a geyser erupts with steam and spray",
        "aurora borealis shimmers in the sky",
        "tide pools fill and drain with waves",
        "a river rapids churns over boulders",
        "firepit flames dance and crackle",
        "cherry blossoms fall like pink snow",
        "lightning flashes in distant storm clouds",
        "a volcano rumbles with glowing embers",
    ],
    "magical_effects": [
        "magical runes glow and pulse on the walls",
        "a crystal orb radiates shifting colors",
        "enchanted books float off the shelves",
        "a cauldron bubbles with green potion",
        "portal energy swirls in an archway",
        "healing light streams from a character's hands",
        "ghostly wisps drift through the corridors",
        "an alchemist's apparatus distills glowing liquid",
        "spell circles rotate on the floor",
        "elemental energy crackles between two pillars",
    ],
    "daily_life": [
        "customers browse shelves and make purchases",
        "a baker pulls fresh bread from the oven",
        "diners eat and drink at small tables",
        "a librarian reshelves books methodically",
        "children play on playground equipment",
        "a painter works at an easel near the window",
        "a mechanic works under a raised vehicle",
        "students sit at desks taking notes",
        "a bartender mixes and serves drinks",
        "a tailor sews fabric at a workstation",
    ],
}

# =============================================================================
# CHARACTERS / SUBJECTS — who populates the scene
# =============================================================================

CHARACTERS = [
    "tiny detailed figures",
    "miniature workers",
    "small animated characters",
    "detailed figurines",
    "miniature adventurers",
    "tiny robot assistants",
    "small fantasy creatures",
    "miniature townspeople",
    "chibi-style characters",
    "detailed miniature animals",
]

# =============================================================================
# STYLE MODIFIERS — visual quality descriptors
# =============================================================================

STYLE_MODIFIERS = [
    "warm golden-hour lighting",
    "cool blue twilight atmosphere",
    "dramatic overhead spotlight",
    "soft pastel color palette",
    "rich saturated colors",
    "moody volumetric fog",
    "crisp morning sunlight",
    "cozy interior glow from windows",
    "neon-lit nighttime scene",
    "autumn color palette with orange and red",
    "winter scene with fresh snow",
    "rainy atmosphere with wet reflections",
    "sunset casting long shadows",
    "underwater caustic lighting",
    "candlelit warm ambiance",
]

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

# =============================================================================
# LOCATION → ACTION COMPATIBILITY
# Which action types make sense with which location categories
# =============================================================================

COMPATIBLE_ACTIONS: dict[str, list[str]] = {
    "urban":      ["ambient", "character_movement", "vehicle_mechanical", "daily_life"],
    "nature":     ["ambient", "character_movement", "nature_dynamic"],
    "fantasy":    ["ambient", "character_movement", "magical_effects", "nature_dynamic"],
    "scifi":      ["ambient", "character_movement", "vehicle_mechanical", "daily_life"],
    "historical": ["ambient", "character_movement", "daily_life", "vehicle_mechanical"],
    "cozy":       ["ambient", "character_movement", "daily_life"],
    "industrial": ["ambient", "vehicle_mechanical", "character_movement"],
}

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_TEMPLATE = (
    "Isometric 3D miniature diorama, static camera, fixed viewpoint. "
    "{location}. {characters} populate the scene. "
    "{action}. {style}. "
    "Tilt-shift depth of field, highly detailed, "
    "no camera movement."
)

PROMPT_TEMPLATE_SIMPLE = (
    "Isometric 3D view, static camera. "
    "{location}. {action}. "
    "{style}. No camera movement."
)

PROMPT_TEMPLATE_MINIMAL = (
    "Isometric 3D miniature scene. "
    "{location}. {action}. "
    "No camera movement."
)


@dataclass
class PromptEntry:
    """A single generated prompt with metadata."""
    id: int
    prompt: str
    category: str
    location: str
    action_type: str
    action: str
    style: str | None = None
    characters: str | None = None


def generate_prompts(
    count: int = 300,
    categories: list[str] | None = None,
    seed: int = 42,
    template: str = "standard",
) -> list[PromptEntry]:
    """Generate diverse isometric scene prompts.

    Args:
        count: Number of prompts to generate
        categories: Location categories to use (None = all)
        seed: Random seed for reproducibility
        template: "standard", "simple", or "minimal"

    Returns:
        List of PromptEntry objects
    """
    rng = random.Random(seed)

    # Select template
    tmpl = {
        "standard": PROMPT_TEMPLATE,
        "simple": PROMPT_TEMPLATE_SIMPLE,
        "minimal": PROMPT_TEMPLATE_MINIMAL,
    }[template]

    # Filter categories
    if categories:
        locs = {k: v for k, v in LOCATIONS.items() if k in categories}
    else:
        locs = LOCATIONS

    # Build compatible (location, action) pairs per category
    all_pairs: list[tuple[str, str, str, str]] = []  # (cat, loc, act_type, act)
    for cat, loc_list in locs.items():
        compat_types = COMPATIBLE_ACTIONS.get(cat, list(ACTIONS.keys()))
        for loc in loc_list:
            for act_type in compat_types:
                for act in ACTIONS[act_type]:
                    all_pairs.append((cat, loc, act_type, act))

    rng.shuffle(all_pairs)

    # Generate combinations
    prompts = []
    used_combos = set()

    for i in range(count):
        # Pick random compatible pair (avoid repeats)
        for _ in range(200):  # max retries to find unique combo
            cat, location, act_type, action = rng.choice(all_pairs)
            combo_key = (location, action)
            if combo_key not in used_combos:
                used_combos.add(combo_key)
                break

        style = rng.choice(STYLE_MODIFIERS)
        characters = rng.choice(CHARACTERS)

        # Build prompt
        if template == "standard":
            prompt = tmpl.format(
                location=location,
                characters=characters,
                action=action,
                style=style,
            )
        elif template == "simple":
            prompt = tmpl.format(
                location=location,
                action=action,
                style=style,
            )
        else:  # minimal
            prompt = tmpl.format(
                location=location,
                action=action,
            )

        prompts.append(PromptEntry(
            id=i,
            prompt=prompt,
            category=cat,
            location=location,
            action_type=act_type,
            action=action,
            style=style if template != "minimal" else None,
            characters=characters if template == "standard" else None,
        ))

    return prompts


def print_stats(prompts: list[PromptEntry]) -> None:
    """Print distribution statistics."""
    from collections import Counter

    cat_counts = Counter(p.category for p in prompts)
    act_counts = Counter(p.action_type for p in prompts)

    print(f"\n{'='*60}")
    print(f"Generated {len(prompts)} prompts")
    print(f"{'='*60}")

    print(f"\n📍 Location categories:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"  {cat:15s} {count:3d} {bar}")

    print(f"\n🎬 Action types:")
    for act, count in sorted(act_counts.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"  {act:20s} {count:3d} {bar}")

    # Unique locations and actions used
    unique_locs = len(set(p.location for p in prompts))
    unique_acts = len(set(p.action for p in prompts))
    total_locs = sum(len(v) for v in LOCATIONS.values())
    total_acts = sum(len(v) for v in ACTIONS.values())
    print(f"\n📊 Coverage:")
    print(f"  Locations used: {unique_locs}/{total_locs} ({100*unique_locs/total_locs:.0f}%)")
    print(f"  Actions used:   {unique_acts}/{total_acts} ({100*unique_acts/total_acts:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate isometric scene prompts")
    parser.add_argument("--count", type=int, default=300, help="Number of prompts")
    parser.add_argument("--category", type=str, default=None,
                        help="Specific category (urban, nature, fantasy, scifi, historical, cozy, industrial)")
    parser.add_argument("--template", choices=["standard", "simple", "minimal"],
                        default="simple", help="Prompt template verbosity")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file")
    parser.add_argument("--list-categories", action="store_true", help="List available categories")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=0, help="Print N sample prompts")
    args = parser.parse_args()

    if args.list_categories:
        print("Available categories:")
        for cat, locs in LOCATIONS.items():
            print(f"  {cat:15s} — {len(locs)} locations")
        print(f"\nAction types:")
        for act_type, acts in ACTIONS.items():
            print(f"  {act_type:20s} — {len(acts)} actions")
        total_combos = sum(len(v) for v in LOCATIONS.values()) * sum(len(v) for v in ACTIONS.values())
        print(f"\nTotal unique combinations: {total_combos:,}")
        return

    categories = [args.category] if args.category else None
    prompts = generate_prompts(
        count=args.count,
        categories=categories,
        seed=args.seed,
        template=args.template,
    )

    print_stats(prompts)

    if args.preview > 0:
        print(f"\n🔍 Sample prompts ({args.preview}):")
        for p in prompts[:args.preview]:
            print(f"\n  [{p.category}/{p.action_type}]")
            print(f"  {p.prompt}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for p in prompts:
                f.write(json.dumps({
                    "id": p.id,
                    "prompt": p.prompt,
                    "category": p.category,
                    "location": p.location,
                    "action_type": p.action_type,
                    "action": p.action,
                    "style": p.style,
                    "characters": p.characters,
                }) + "\n")
        print(f"\n✅ Saved {len(prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()
