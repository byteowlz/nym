//! Session ID generation for tracking anonymization sessions.
//!
//! Generates human-readable, searchable session IDs in the format:
//! `adjective-noun` (e.g., "calm-frog", "bold-lamp")
//!
//! These IDs are embedded in replacements and stored in key files to enable
//! easy lookup and association between anonymized data and its key file.

use fake::rand::SeedableRng;
use fake::rand::prelude::IndexedRandom;
use fake::rand::rngs::StdRng;

/// Word lists for generating session IDs.
/// Source: https://github.com/byteowlz/schemas/blob/main/wordlist/word_lists.toml
mod words {
    pub const ADJECTIVES: &[&str] = &[
        "able", "acid", "aged", "airy", "akin", "alto", "amok", "anti", "arch", "arid", "arty",
        "auld", "avid", "away", "awol", "awry", "back", "bald", "bare", "base", "bass", "bats",
        "beat", "bent", "best", "beta", "bias", "blue", "bold", "bone", "bony", "boon", "born",
        "boss", "both", "brag", "buff", "bulk", "bush", "bust", "busy", "calm", "camp", "chic",
        "clad", "cold", "cool", "cosy", "cozy", "curt", "cute", "cyan", "daft", "damp", "dank",
        "dark", "deaf", "dear", "deep", "deft", "dire", "dirt", "done", "dour", "down", "drab",
        "dual", "dull", "dyed", "each", "east", "easy", "edgy", "epic", "even", "evil", "eyed",
        "fair", "fake", "fast", "faux", "fell", "fine", "firm", "five", "flat", "flip", "fond",
        "foul", "foxy", "free", "full", "gaga", "game", "gilt", "glad", "glib", "glum", "gold",
        "gone", "good", "gray", "grey", "grim", "hale", "half", "halt", "hard", "hazy", "held",
        "here", "hick", "high", "hind", "holy", "home", "huge", "iced", "icky", "idle", "iffy",
        "inky", "iron", "just", "keen", "kept", "kind", "lacy", "laid", "lame", "lank", "last",
        "late", "lazy", "lean", "left", "less", "lest", "like", "limp", "lite", "live", "loco",
        "lone", "long", "lost", "loud", "lush", "luxe", "made", "main", "male", "many", "mass",
        "maxi", "mean", "meek", "meet", "mere", "midi", "mild", "mini", "mint", "mock", "mono",
        "moot", "more", "most", "much", "must", "mute", "near", "neat", "next", "nice", "nigh",
        "nine", "none", "nosy", "nude", "null", "numb", "nuts", "oily", "okay", "only", "open",
        "oral", "oval", "over", "paid", "pale", "pass", "past", "pent", "pied", "pink", "plus",
        "poor", "port", "posh", "prim", "puff", "punk", "puny", "pure", "racy", "rank", "rare",
        "rash", "real", "rear", "rich", "rife", "ripe", "roan", "rosy", "rude", "rust", "safe",
        "salt", "same", "sane", "sear", "self", "sent", "sewn", "sham", "shed", "shot", "shut",
        "side", "sign", "size", "skew", "skim", "slim", "slow", "smug", "snub", "snug", "soft",
        "sold", "sole", "solo", "some", "sore", "sour", "sown", "spry", "star", "such", "sunk",
        "sure", "tall", "tame", "tart", "taut", "teal", "teen", "then", "thin", "tidy", "tied",
        "tiny", "toed", "tops", "torn", "trig", "trim", "true", "twin", "ugly", "used", "vain",
        "vast", "very", "vile", "void", "warm", "wary", "wavy", "waxy", "weak", "wide", "wild",
        "wily", "wise", "worn", "zany", "zero",
    ];

    pub const NOUNS: &[&str] = &[
        "acer", "aces", "acid", "acne", "acre", "acts", "adds", "afro", "agar", "aged", "ages",
        "ahem", "aide", "aids", "aims", "airs", "alas", "ally", "aloe", "alto", "alum", "amen",
        "amir", "ammo", "amor", "amps", "anil", "ante", "anti", "ants", "apes", "apex", "aqua",
        "arch", "arcs", "area", "ares", "aria", "arms", "army", "arts", "atom", "aunt", "aura",
        "auto", "axes", "axis", "axle", "babe", "baby", "back", "bags", "bail", "bait", "bale",
        "ball", "balm", "band", "bane", "bang", "bank", "bans", "barb", "bard", "bark", "barn",
        "bars", "base", "bash", "bass", "bath", "bats", "bays", "bead", "beak", "beam", "bean",
        "bear", "beat", "beau", "beds", "beef", "beep", "beer", "bees", "beet", "bell", "belt",
        "bend", "bent", "berg", "best", "beta", "bets", "bias", "bids", "bike", "bill", "bind",
        "bins", "bird", "bite", "bits", "blob", "bloc", "blog", "blot", "blow", "blue", "blur",
        "boar", "boat", "body", "boil", "bold", "bolt", "bond", "bone", "bong", "book", "boom",
        "boon", "boot", "bore", "born", "boss", "bots", "bout", "bowl", "bows", "boys", "brag",
        "bran", "bras", "brat", "bray", "brew", "brie", "brig", "brim", "brow", "buck", "buds",
        "buff", "bugs", "bulb", "bulk", "bull", "bump", "bunk", "buns", "buoy", "burn", "burr",
        "bush", "bust", "buys", "buzz", "byte", "cabs", "cafe", "cage", "cake", "calf", "call",
        "calm", "camo", "camp", "cams", "cane", "cans", "cape", "caps", "card", "care", "carp",
        "cars", "cart", "case", "cash", "cast", "cats", "cave", "cell", "cent", "cert", "chap",
        "char", "chat", "chef", "chew", "chic", "chin", "chit", "chop", "cite", "city", "clam",
        "clan", "clap", "claw", "clay", "clip", "clot", "club", "clue", "coal", "coat", "coca",
        "coco", "code", "coil", "coin", "cola", "cold", "colt", "coma", "comb", "come", "comp",
        "cone", "cons", "cool", "coop", "cope", "cops", "copy", "cord", "core", "cork", "corn",
        "corp", "cost", "cosy", "coup", "cove", "cows", "cozy", "crab", "crew", "crib", "crop",
        "crow", "cube", "cubs", "cues", "cuff", "cult", "cups", "curb", "cure", "curl", "cusp",
        "cuts", "cyst", "czar", "dads", "dame", "damp", "dams", "dare", "dark", "dart", "dash",
        "data", "date", "days", "deaf", "deal", "dear", "debt", "deck", "deco", "deed", "deep",
        "deer", "deli", "demo", "dent", "desk", "dial", "dice", "dies", "diet", "digs", "dill",
        "dime", "ding", "dips", "dirt", "disc", "dish", "disk", "diva", "dive", "dock", "docs",
        "does", "dogs", "dole", "doll", "dome", "dong", "dons", "doom", "door", "dope", "dork",
        "dorm", "dose", "dots", "dove", "down", "drab", "drag", "draw", "drip", "drop", "drum",
        "dubs", "duck", "duct", "dude", "duel", "dues", "duet", "duff", "dump", "dune", "dung",
        "dunk", "dusk", "dust", "duty", "dyer", "dyes", "dyke", "ears", "ease", "east", "eats",
        "echo", "eddy", "edge", "eels", "eggs", "egos", "emir", "ends", "envy", "epic", "eras",
        "even", "evil", "exam", "exec", "exes", "exit", "expo", "eyes", "face", "fact", "fade",
        "fair", "fake", "fall", "fame", "fang", "fans", "fare", "farm", "fast", "fate", "fats",
        "fawn", "fear", "feat", "feds", "feed", "feel", "fees", "feet", "fell", "felt", "fern",
        "feud", "fife", "figs", "file", "fill", "film", "find", "fine", "fink", "fins", "fire",
        "firm", "fish", "fist", "fits", "five", "flag", "flak", "flap", "flat", "flaw", "flax",
        "flea", "flex", "flip", "flop", "flow", "flux", "foam", "foes", "foil", "fold", "folk",
        "font", "food", "fool", "foot", "fork", "form", "fort", "foul", "fowl", "frat", "fray",
        "free", "fret", "frog", "fuel", "full", "fund", "funk", "furs", "fury", "fuse", "fuss",
        "fuzz", "gage", "gags", "gain", "gait", "gala", "gale", "gall", "gals", "game", "gang",
        "gaps", "garb", "gasp", "gate", "gays", "gaze", "gear", "geek", "gems", "gent", "germ",
        "gets", "gift", "gigs", "gill", "gilt", "girl", "gist", "give", "glad", "glee", "glow",
        "glue", "goal", "goat", "gods", "goes", "gold", "golf", "gong", "good", "goth", "gout",
        "gown", "grab", "grad", "gran", "gray", "grey", "grid", "grin", "grip", "grit", "grub",
        "gull", "gums", "guns", "guru", "gust", "guts", "guys", "gyms", "hack", "hail", "hair",
        "hale", "half", "hall", "halo", "halt", "hand", "hang", "hank", "hare", "harm", "harp",
        "hash", "hats", "haul", "have", "hawk", "hays", "haze", "head", "heap", "heat", "heed",
        "heel", "heir", "helm", "help", "hemp", "hens", "herb", "herd", "here", "hero", "hide",
        "high", "hike", "hill", "hind", "hint", "hips", "hire", "hiss", "hits", "hive", "hoax",
        "hobo", "hogs", "hold", "hole", "holy", "home", "homo", "hone", "hoof", "hook", "hoop",
        "hops", "horn", "hose", "host", "hour", "howl", "hubs", "hues", "huff", "hugs", "hula",
        "hulk", "hump", "hunk", "hush", "huts", "hymn", "hype", "icon", "idea", "idle", "idol",
        "ills", "imam", "inch", "info", "inks", "inns", "ions", "iron", "itch", "item", "jail",
        "jams", "jars", "jaws", "jays", "jazz", "jeep", "jest", "jets", "jinx", "jive", "jobs",
        "jock", "join", "joke", "jolt", "joys", "judo", "july", "jump", "june", "junk", "jury",
        "kale", "keel", "keen", "keep", "keys", "kick", "kids", "kiln", "kilo", "kind", "king",
        "kink", "kiss", "kite", "kits", "kiwi", "knee", "knit", "knob", "labs", "lace", "lack",
        "lads", "lady", "lags", "lair", "lake", "lama", "lamb", "lame", "lamp", "land", "lane",
        "laps", "lark", "lash", "lass", "last", "lava", "lawn", "laws", "lays", "lead", "leaf",
        "leak", "lean", "leap", "leds", "lees", "left", "lego", "legs", "lens", "lent", "lets",
        "liar", "lice", "lick", "lids", "lied", "lien", "lies", "lieu", "life", "lift", "like",
        "lily", "limb", "lime", "limo", "limp", "line", "ling", "link", "lion", "lips", "lisp",
        "list", "load", "loaf", "loan", "lobe", "loch", "lock", "loft", "logo", "logs", "look",
        "loom", "loop", "loot", "lord", "lore", "loss", "lost", "lots", "love", "lows", "lube",
        "luck", "lull", "lump", "lung", "lure", "lush", "lynx", "mace", "mach", "mack", "macs",
        "mags", "maid", "mail", "main", "make", "male", "mall", "malt", "mama", "mane", "mans",
        "maps", "mare", "mart", "mash", "mask", "mass", "mast", "mate", "math", "mats", "maxi",
        "mayo", "mays", "maze", "meal", "mean", "meat", "meds", "meet", "melt", "meme", "memo",
        "mend", "mens", "menu", "meow", "mere", "mesh", "mess", "mice", "midi", "mile", "milk",
        "mill", "mime", "mind", "mine", "mini", "mink", "mins", "mint", "miss", "mite", "mitt",
        "moan", "moat", "mobs", "mock", "mode", "mods", "mojo", "mold", "mole", "moms", "monk",
        "mono", "mood", "moon", "moor", "moot", "more", "morn", "moss", "moth", "move", "much",
        "muck", "mugs", "mule", "mums", "must", "mute", "myth", "nada", "nail", "name", "naps",
        "nave", "neck", "need", "neon", "nerd", "nest", "nets", "news", "newt", "nice", "nine",
        "node", "nods", "none", "nook", "noon", "norm", "nose", "note", "noun", "nude", "nuke",
        "null", "nuns", "nuts", "oaks", "oars", "oath", "oats", "odds", "odor", "ogre", "oils",
        "okay", "olds", "omen", "ones", "opal", "open", "oral", "outs", "oval", "oven", "over",
        "owls", "pack", "pads", "page", "pain", "pale", "palm", "pals", "pane", "pang", "pans",
        "pant", "park", "part", "pass", "past", "path", "pats", "pave", "pawn", "paws", "pays",
        "peak", "peas", "peat", "peek", "peel", "peep", "peer", "pens", "perk", "perm", "peso",
        "pest", "pets", "pick", "pics", "pier", "pies", "pigs", "pike", "pile", "pill", "pine",
        "ping", "pink", "pins", "pint", "pipe", "pita", "pits", "pity", "plan", "plat", "play",
        "plea", "plot", "plow", "ploy", "plug", "plum", "plus", "pods", "poem", "poet", "poke",
        "pole", "poll", "pond", "pong", "pony", "pool", "poor", "pops", "pore", "pork", "port",
        "pose", "post", "pots", "prep", "prey", "prod", "prof", "prom", "prop", "pros", "pubs",
        "puck", "puff", "pull", "pulp", "puma", "pump", "punk", "puns", "punt", "pups", "push",
        "puts", "putt", "quad", "quay", "quid", "quiz", "race", "rack", "raft", "rage", "rags",
        "raid", "rail", "rain", "rake", "ramp", "rams", "rand", "rank", "rant", "raps", "rash",
        "rate", "rats", "rave", "rays", "read", "real", "rear", "reds", "reed", "reef", "reel",
        "refs", "rein", "rent", "reps", "rest", "ribs", "rice", "rich", "ride", "riff", "rift",
        "rigs", "rims", "ring", "rink", "riot", "rips", "rise", "risk", "rite", "road", "roar",
        "robe", "rock", "rods", "role", "roll", "roof", "rook", "room", "root", "rope", "rout",
        "rows", "rubs", "ruff", "rugs", "ruin", "rule", "rump", "rune", "rung", "runs", "ruse",
        "rust", "sack", "safe", "saga", "sail", "sake", "sale", "salt", "same", "sand", "sang",
        "sari", "sash", "save", "says", "scam", "scan", "scar", "seal", "seam", "seas", "seat",
        "secs", "sect", "seed", "seek", "seer", "sees", "self", "sell", "semi", "sens", "sent",
        "sept", "sera", "sets", "shag", "sham", "shed", "shin", "ship", "shoe", "shop", "shot",
        "show", "side", "sigh", "sign", "silk", "sill", "silo", "sine", "sink", "sins", "sire",
        "site", "size", "skid", "skim", "skin", "skip", "skis", "skit", "slab", "slag", "slam",
        "slap", "sled", "slew", "slip", "slit", "slot", "slug", "slum", "slur", "smog", "snag",
        "snap", "snow", "snug", "soak", "soap", "soar", "sobs", "sock", "soda", "sofa", "soil",
        "sole", "solo", "soma", "song", "sons", "soot", "sore", "sort", "soul", "soup", "sour",
        "sous", "spam", "spar", "spat", "spec", "spin", "spot", "spur", "stab", "stag", "star",
        "stay", "stem", "step", "stew", "stir", "stop", "stub", "stud", "subs", "suit", "sumo",
        "sums", "sung", "suns", "surf", "swan", "swap", "swat", "sway", "swim", "tabs", "tack",
        "taco", "tact", "tags", "tail", "take", "tale", "talk", "tall", "tang", "tank", "tape",
        "taps", "tart", "task", "taxi", "teal", "team", "tear", "teas", "tech", "teen", "tees",
        "tell", "temp", "tens", "tent", "term", "test", "text", "thaw", "then", "thou", "thus",
        "tick", "tide", "tidy", "tier", "ties", "tiff", "tile", "till", "tilt", "time", "ting",
        "tins", "tint", "tips", "tire", "toad", "toes", "tofu", "toil", "toll", "tomb", "tome",
        "tone", "tons", "tool", "toon", "toot", "tops", "tort", "toss", "tote", "tots", "tour",
        "tout", "town", "toys", "tram", "trap", "tray", "tree", "trek", "trim", "trio", "trip",
        "trot", "true", "tsar", "tube", "tubs", "tuck", "tuna", "tune", "turf", "turn", "tutu",
        "twig", "twin", "type", "typo", "tyre", "unit", "urge", "user", "uses", "vale", "vane",
        "vans", "vase", "veal", "veil", "vein", "vent", "verb", "vest", "veto", "vets", "vial",
        "vibe", "vice", "view", "vine", "viva", "void", "volt", "vote", "vows", "wage", "wait",
        "wake", "walk", "wall", "wand", "want", "warp", "wars", "wash", "wasp", "wave", "ways",
        "webs", "week", "weir", "weld", "whey", "whim", "whip", "whit", "whos", "wick", "wife",
        "wifi", "wigs", "wild", "will", "wind", "wine", "wing", "wink", "wipe", "wire", "wise",
        "wish", "wits", "woes", "womb", "wont", "woof", "wool", "word", "work", "worm", "wrap",
        "wren", "writ", "yank", "yard", "yarn", "yawn", "year", "yelp", "yeti", "yoke", "yolk",
        "zeal", "zero", "zest", "zeta", "zinc", "zone", "zoom", "zoos",
    ];
}

/// A session identifier for tracking anonymization sessions.
#[derive(Debug, Clone)]
pub struct Session {
    /// The adjective-noun identifier (e.g., "calm-frog")
    pub id: String,
    /// Original source filename (if available)
    pub source: Option<String>,
}

impl Session {
    /// Generate a new random session ID.
    pub fn new(source: Option<&str>) -> Self {
        let mut rng = StdRng::from_os_rng();
        let id = generate_id(&mut rng);
        Self {
            id,
            source: source.map(String::from),
        }
    }

    /// Generate a session with a specific seed (for deterministic testing).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Public API - used by consumers")
    )]
    pub fn with_seed(seed: u64, source: Option<&str>) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let id = generate_id(&mut rng);
        Self {
            id,
            source: source.map(String::from),
        }
    }

    /// Create a session with a user-specified tag.
    pub fn with_tag(tag: &str, source: Option<&str>) -> Self {
        Self {
            id: tag.to_string(),
            source: source.map(String::from),
        }
    }

    /// Get the full session reference including source filename.
    /// Format: "adjective-noun" or "adjective-noun (filename.txt)"
    pub fn full_reference(&self) -> String {
        match &self.source {
            Some(src) => format!("{} ({})", self.id, src),
            None => self.id.clone(),
        }
    }

    /// Generate a JSON header line for a key file.
    #[cfg(feature = "streaming")]
    pub fn to_key_file_header(&self) -> String {
        serde_json::json!({
            "version": "1",
            "session": self.id,
            "source": self.source,
            "created": chrono::Utc::now().to_rfc3339(),
        })
        .to_string()
    }
}

/// Generate an adjective-noun identifier.
fn generate_id<R: fake::rand::Rng>(rng: &mut R) -> String {
    let adj = words::ADJECTIVES.choose(rng).unwrap_or(&"calm");
    let noun = words::NOUNS.choose(rng).unwrap_or(&"lamp");
    format!("{adj}-{noun}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_session() {
        let session = Session::new(Some("test.txt"));
        assert!(session.id.contains('-'));
        assert_eq!(session.source, Some("test.txt".to_string()));
    }

    #[test]
    fn test_seeded_session_is_deterministic() {
        let s1 = Session::with_seed(42, None);
        let s2 = Session::with_seed(42, None);
        assert_eq!(s1.id, s2.id);
    }

    #[test]
    fn test_different_seeds_different_ids() {
        let s1 = Session::with_seed(1, None);
        let s2 = Session::with_seed(2, None);
        assert_ne!(s1.id, s2.id);
    }

    #[test]
    fn test_custom_tag() {
        let session = Session::with_tag("audit-2024", Some("data.json"));
        assert_eq!(session.id, "audit-2024");
        assert_eq!(session.full_reference(), "audit-2024 (data.json)");
    }

    #[test]
    fn test_full_reference() {
        let session = Session::with_seed(42, Some("input.txt"));
        let reference = session.full_reference();
        assert!(reference.contains("input.txt"));
        assert!(reference.contains('-'));
    }
}
