"""
attacks.py
----------
The Attack / Robustness Engine. Each attack takes watermarked text and tries to
wash the watermark out while keeping the meaning.

    ATTACKS = {
        "Synonym Replacement", "Paraphrasing", "Back-Translation", "LLM Rewriting",
        "Text Shortening", "Random Word Deletion", "Sentence Restructuring",
    }

Every attack has the signature  attack(text, use_model=False) -> str.

use_model=False  : fast, offline, rule-based fallbacks (no downloads). The
                   Paraphrasing / Back-Translation / LLM-Rewriting fallbacks are
                   SIMULATIONS built from synonym swaps, phrase rewrites, word-order
                   changes and deletions; they are labelled as such in the README.
use_model=True   : real NLP models via Hugging Face (needs internet + torch):
                   a T5 paraphraser, MarianMT en->de->en, and flan-t5 rewriting.
                   If a model cannot be loaded the attack falls back to the rules.

Honest-evaluation note: the attacker's synonym table (ATTACK_SYNONYMS) is
written independently of the defender's synonym clusters in lexicon.py, so
words the defender's clusters do not cover are *not* protected.
"""

import random
import re
import warnings
from typing import Callable, Dict, List

# ---------------------------------------------------------------------------
# Attacker's synonym table:  word -> candidate replacements
# ---------------------------------------------------------------------------
_RAW = """
people: folks, persons, individuals | many: numerous, plenty of | good: decent, solid, fine | text: wording, prose, passage
small: tiny, little, modest | simple: easy, plain, straightforward | building: constructing, assembling, creating
computer: machine, pc, device | learn: pick up, absorb, study | data: figures, records, information | large: big, huge, sizeable
machine: device, apparatus, engine | write: compose, draft, pen | world: globe, planet, earth | help: assist, aid, support
brings: delivers, offers, carries | work: labour, effort, operate | words: terms, vocabulary, expressions
language: speech, tongue, wording | problems: issues, troubles, difficulties | helps: assists, aids, supports
better: superior, improved, finer | learning: studying, education, training | improves: enhances, boosts, upgrades
model: system, template, design | programs: software, applications, routines | tell: inform, say, notify | almost: nearly, virtually
travel: journey, tour, voyage | important: significant, vital, crucial | strong: sturdy, powerful, robust, solid
regular: steady, routine, frequent | walk: stroll, stride, wander | weather: climate, conditions, forecast
food: meals, cuisine, nourishment | often: frequently, regularly, commonly | problem: issue, difficulty, trouble
money: cash, funds, income | signal: sign, cue, marker | artificial: synthetic, man-made, manufactured
intelligence: intellect, smarts, reasoning | systems: setups, frameworks, structures | solve: resolve, fix, crack
patterns: designs, trends, structures | think: believe, reckon, suppose | examples: samples, cases, instances
slowly: gradually, steadily, leisurely | makes: creates, produces, forms | kind: sort, type, variety | connects: links, joins, ties
cells: units, units, nodes | uses: employs, utilises, applies | find: locate, discover, spot | information: details, facts, knowledge
today: nowadays, currently, now | phone: handset, mobile, cellphone | around: about, roughly, nearby | across: over, throughout, through
passwords: passcodes, secrets, keys | begins: starts, opens, commences | fresh: crisp, new, clean | rest: relax, pause, recover
heat: warmth, hotness, high temperature | grows: expands, increases, rises | everyone: all people, everybody | looks: seeks, appears, searches
cold: chilly, icy, cool | water: liquid, fluid, h2o | rain: showers, rainfall, drizzle | days: periods, times, spells
energy: power, vigour, vitality | grow: expand, develop, increase | provide: supply, offer, give | home: house, residence, dwelling
meal: dish, dinner, repast | friends: companions, pals, mates | keeps: retains, maintains, holds | sleep: slumber, rest, nap
short: brief, quick, concise | reduce: lower, cut, decrease | stress: strain, pressure, tension | enough: sufficient, adequate, ample
wind: breeze, gust, air | countries: nations, states, lands | farms: ranches, holdings, plantations | jobs: positions, careers, posts
life: existence, living, being | difficult: hard, tough, tricky | students: pupils, learners, scholars | questions: queries, enquiries, problems
best: finest, top, greatest | practice: rehearse, train, drill | mistakes: errors, slips, blunders | online: on the web, on the internet
allow: permit, let, enable | study: examine, research, review | cities: towns, metropolises, urban centres | city: town, metropolis, municipality
living: residing, existing, dwelling | build: construct, assemble, erect | listen: hear, attend, pay attention | carefully: attentively, cautiously, closely
budget: spending plan, financial plan, allowance | reading: perusing, studying, scanning | person: individual, human, someone
years: decades, seasons, periods | memories: recollections, remembrances, reminiscences | clear: obvious, plain, evident
valuable: precious, useful, worthwhile | times: occasions, moments, periods | science: research, knowledge, discipline
decisions: choices, judgements, resolutions | boring: dull, tedious, monotonous | modern: contemporary, current, up-to-date
amounts: quantities, volumes, sums | instead: rather, alternatively | following: obeying, tracking, pursuing | fixed: set, rigid, constant
rules: regulations, guidelines, principles | beginner: novice, newcomer, learner | teaching: instructing, educating, coaching
showing: displaying, revealing, presenting | experience: practice, exposure, know-how | thousands: many thousands, countless
numbers: figures, digits, values | predictions: forecasts, estimates, projections | popular: well-liked, common, favoured
copies: imitates, mimics, duplicates | brain: mind, intellect | deep: profound, extensive | layers: levels, tiers, strata
complex: complicated, intricate, elaborate | images: pictures, photos, visuals | sound: audio, noise, tone
process: handle, treat, work through | stores: keeps, saves, holds | memory: recall, storage | perform: carry out, execute, do
calculations: computations, sums, arithmetic | quickly: swiftly, rapidly, fast | languages: tongues, dialects | device: gadget, appliance, unit
contains: holds, includes, carries | internet: web, net, online world | share: distribute, pass on, exchange
instantly: immediately, at once, right away | websites: web pages, sites | network: web, system, grid | security: safety, protection
steal: take, pinch, swipe | private: personal, confidential, secret | updates: upgrades, revisions, patches | safe: secure, protected
warm: mild, balmy, cosy | light: brightness, glow, illumination | spreading: stretching, extending, expanding
birds: fowl, songbirds | sing: chirp, warble, croon | trees: timber, woods | park: garden, green, common | feels: seems, senses
grass: lawn, turf, greenery | night: evening, darkness, dusk | children: kids, youngsters, youth | play: frolic, romp, have fun
near: close to, next to, beside | river: stream, waterway, creek | parents: mothers and fathers, guardians
shade: shadow, shelter, cover | afternoon: midday, daytime | quiet: silent, peaceful, calm | place: spot, location, site
changes: shifts, alters, varies | seasons: times of year, periods | gentle: soft, mild, light | flowers: blooms, blossoms
summer: warm season, hot season | long: lengthy, extended, prolonged | cool: chilly, fresh, brisk | golden: gilded, amber, yellow
leaves: foliage, fronds | winter: cold season, wintertime | snow: snowfall, flurries | farmers: growers, cultivators
watch: observe, monitor, view | closely: carefully, intently | decides: determines, settles, chooses | land: ground, soil, terrain
produce: yield, generate, create | healthy: fit, well, wholesome | gives: provides, grants, offers | body: physique, frame
needs: requires, demands, wants | vegetables: greens, veggies | fruit: produce, berries | strength: power, might, vigour
cooking: preparing food, baking, cuisine | cheaper: less costly, more affordable | buying: purchasing, acquiring, getting
fast: quick, speedy, rapid | meals: dishes, dinners | shared: split, divided, joint | feel: sense, experience, perceive
happy: glad, pleased, content | calm: relaxed, serene, tranquil | exercise: workout, training, activity | heart: core, cardiac organ
doctors: physicians, medics, clinicians | advise: recommend, counsel, suggest | adults: grown-ups, mature people | move: shift, go, travel
minutes: moments, brief periods | drinking: sipping, consuming, having | getting: obtaining, receiving, gaining
health: wellbeing, fitness, condition | climate: weather pattern, environment | change: shift, alteration, modification
serious: grave, severe, major | whole: entire, complete, full | burning: combusting, igniting, scorching | adds: contributes, includes
planet: world, globe | rising: climbing, increasing, ascending | temperatures: heat levels, degrees | cause: trigger, produce, lead to
stronger: more powerful, tougher, sturdier | storms: tempests, gales, squalls | longer: lengthier, more extended
melting: thawing, dissolving | clean: pure, spotless, fresh | damage: harm, injury, destruction | solar: sun-powered, sun
education: schooling, teaching, learning | opens: unlocks, starts | teacher: instructor, tutor, educator | explains: clarifies, describes
ideas: thoughts, notions, concepts | encourages: motivates, urges, supports | afraid: scared, fearful, nervous
making: creating, forming, producing | libraries: reading rooms, archives | courses: classes, programmes | anyone: anybody, any person
subject: topic, field, theme | crowded: packed, busy, congested | places: spots, locations, sites | millions: countless people, masses
live: reside, dwell, exist | busy: hectic, bustling, occupied | roads: streets, routes, highways | tall: high, lofty, towering
public: communal, civic, shared | trains: railways, locomotives | function: operate, work, run | traffic: congestion, vehicles
pollution: contamination, smog, dirt | common: usual, frequent, widespread | lanes: paths, tracks | parks: gardens, greens
offers: provides, presents, gives | cultural: artistic, social, traditional | events: occasions, happenings, gatherings
expensive: costly, pricey, dear | noisy: loud, rowdy, clamorous | space: outer space, cosmos, room | exploration: discovery, investigation
humans: people, mankind, men and women | understand: grasp, comprehend, follow | universe: cosmos, creation
"""

ATTACK_SYNONYMS: Dict[str, List[str]] = {}
for _line in _RAW.strip().splitlines():
    for _entry in _line.split("|"):
        if ":" in _entry:
            _k, _v = _entry.split(":", 1)
            ATTACK_SYNONYMS[_k.strip().lower()] = [s.strip() for s in _v.split(",") if s.strip()]

# phrase-level rewrites used by the rule-based paraphraser
_PHRASES = [
    (r"\bin order to\b", "to"), (r"\ba lot of\b", "plenty of"), (r"\bis able to\b", "can"),
    (r"\bsuch as\b", "like"), (r"\bfor example\b", "for instance"), (r"\bhowever\b", "but"),
    (r"\bbecause\b", "since"), (r"\bmany people\b", "lots of people"), (r"\bvery\b", "quite"),
    (r"\bimportant\b", "significant"), (r"\balmost every\b", "nearly every"),
    (r"\bit can be hard to\b", "it is difficult to"), (r"\bthere is\b", "there exists"),
    (r"\bcan help\b", "is able to help"), (r"\bhelps?\b(?= people)", "assists"),
    (r"\bas well as\b", "along with"), (r"\bmost important\b", "key"),
    (r"\bat the same time\b", "simultaneously"), (r"\bof the\b", "of the"),
]
_DROPPABLE = {"also", "very", "really", "just", "quite", "still", "even", "often", "sometimes"}
_FUNCTION_DROP = {"the", "a", "an", "of"}

_WORD = re.compile(r"[A-Za-z']+")


def _rng(text: str, salt: int = 0) -> random.Random:
    return random.Random((sum(map(ord, text[:200])) * 7919 + salt) & 0xFFFFFFFF)


def _match_case(src: str, dst: str) -> str:
    if src.isupper() and len(src) > 1:
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def _lookup(word: str):
    w = word.lower()
    if w in ATTACK_SYNONYMS:
        return ATTACK_SYNONYMS[w]
    return None


def _wordnet_synonym(word: str, rnd: random.Random):
    """Use WordNet when nltk + its data are installed (optional)."""
    try:
        from nltk.corpus import wordnet
        lemmas = {l.name().replace("_", " ") for s in wordnet.synsets(word) for l in s.lemmas()}
        lemmas.discard(word.lower())
        lemmas = [l for l in lemmas if " " not in l]
        return rnd.choice(sorted(lemmas)) if lemmas else None
    except Exception:
        return None


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# 1. Synonym replacement
# ---------------------------------------------------------------------------
def synonym_replace_attack(text: str, fraction: float = 0.3, use_model: bool = False) -> str:
    """Replace about `fraction` of the replaceable words with synonyms."""
    rnd = _rng(text, 1)

    def repl(m):
        w = m.group(0)
        if len(w) < 4 or rnd.random() > fraction:
            return w
        cands = _lookup(w)
        if cands:
            return _match_case(w, rnd.choice(cands))
        if use_model:
            s = _wordnet_synonym(w, rnd)
            if s:
                return _match_case(w, s)
        return w

    return _WORD.sub(repl, text)


# ---------------------------------------------------------------------------
# helpers shared by the rule-based fallbacks
# ---------------------------------------------------------------------------
def _apply_phrases(text: str, rnd: random.Random, p: float = 0.8) -> str:
    for pat, rep in _PHRASES:
        if rnd.random() < p:
            text = re.sub(pat, lambda m: _match_case(m.group(0), rep), text, flags=re.I)
    return text


def _swap_clauses(sentence: str, rnd: random.Random) -> str:
    """'A, and B.' -> 'B, and A.' style restructuring for simple coordinated sentences."""
    m = re.match(r"^(.*?[^,]),\s+(and|but|so)\s+(.*?)([.!?]?)$", sentence)
    if m and rnd.random() < 0.7:
        a, conj, b, end = m.groups()
        b = b[:1].upper() + b[1:]
        a = a[:1].lower() + a[1:]
        return f"{b}, {conj} {a}{end or '.'}"
    return sentence


def _drop_fillers(text: str, rnd: random.Random, p: float) -> str:
    def repl(m):
        return "" if m.group(0).lower() in _DROPPABLE and rnd.random() < p else m.group(0)
    return re.sub(r"\s{2,}", " ", _WORD.sub(repl, text)).strip()


def _swap_adjacent_words(text: str, rnd: random.Random, p: float) -> str:
    toks = text.split(" ")
    i = 0
    while i < len(toks) - 1:
        a, b = toks[i], toks[i + 1]
        if a.isalpha() and b.isalpha() and rnd.random() < p:
            toks[i], toks[i + 1] = b, a
            i += 2
        else:
            i += 1
    return " ".join(toks)


def _drop_function_words(text: str, rnd: random.Random, p: float) -> str:
    def repl(m):
        return "" if m.group(0).lower() in _FUNCTION_DROP and rnd.random() < p else m.group(0)
    return re.sub(r"\s{2,}", " ", _WORD.sub(repl, text)).strip()


def _fix_sentence_case(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


# ---------------------------------------------------------------------------
# optional real-model attacks (lazy, cached)
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}


def _hf_pipeline(task: str, model: str):
    key = (task, model)
    if key not in _MODEL_CACHE:
        from transformers import pipeline
        _MODEL_CACHE[key] = pipeline(task, model=model)
    return _MODEL_CACHE[key]


def _model_paraphrase(text: str) -> str:
    pipe = _hf_pipeline("text2text-generation", "Vamsi/T5_Paraphrase_Paws")
    out = []
    for s in _split_sentences(text):
        r = pipe(f"paraphrase: {s} </s>", max_length=96, num_beams=4, do_sample=False)
        out.append(r[0]["generated_text"])
    return " ".join(out)


def _model_back_translate(text: str) -> str:
    fwd = _hf_pipeline("translation", "Helsinki-NLP/opus-mt-en-de")
    back = _hf_pipeline("translation", "Helsinki-NLP/opus-mt-de-en")
    out = []
    for s in _split_sentences(text):
        de = fwd(s, max_length=200)[0]["translation_text"]
        out.append(back(de, max_length=200)[0]["translation_text"])
    return " ".join(out)


def _model_rewrite(text: str) -> str:
    pipe = _hf_pipeline("text2text-generation", "google/flan-t5-base")
    out = []
    for s in _split_sentences(text):
        r = pipe(f"Rewrite this sentence in different words: {s}", max_length=128, do_sample=False, num_beams=4)
        out.append(r[0]["generated_text"])
    return " ".join(out)


def _try_model(fn: Callable[[str], str], text: str, fallback: Callable[[str], str]) -> str:
    try:
        return fn(text)
    except Exception as exc:    # no torch / no internet / model missing
        warnings.warn(f"Model-based attack unavailable ({exc!r}); using the offline fallback.", RuntimeWarning)
        return fallback(text)


# ---------------------------------------------------------------------------
# 2-4. Paraphrasing, back-translation, LLM rewriting
# ---------------------------------------------------------------------------
def _rule_paraphrase(text: str) -> str:
    rnd = _rng(text, 2)
    out = _apply_phrases(text, rnd)
    out = " ".join(_swap_clauses(s, rnd) for s in _split_sentences(out))
    out = synonym_replace_attack(out, fraction=0.25)
    out = _drop_fillers(out, rnd, 0.5)
    return _fix_sentence_case(out)


def paraphrase_attack(text: str, use_model: bool = False) -> str:
    if use_model:
        return _try_model(_model_paraphrase, text, _rule_paraphrase)
    return _rule_paraphrase(text)


def _rule_back_translate(text: str) -> str:
    """SIMULATED round-trip translation: lossy synonym/word-order/function-word noise."""
    rnd = _rng(text, 3)
    out = _apply_phrases(text, rnd, 0.6)
    out = synonym_replace_attack(out, fraction=0.35)
    out = _swap_adjacent_words(out, rnd, 0.06)
    out = _drop_function_words(out, rnd, 0.12)
    return _fix_sentence_case(out)


def back_translation_attack(text: str, use_model: bool = False) -> str:
    if use_model:
        return _try_model(_model_back_translate, text, _rule_back_translate)
    return _rule_back_translate(text)


def _rule_rewrite(text: str) -> str:
    """SIMULATED LLM rewrite: strongest rule-based mix (synonyms, clause swaps, reordering, trimming)."""
    rnd = _rng(text, 4)
    sents = _split_sentences(_apply_phrases(text, rnd, 0.9))
    sents = [_swap_clauses(s, rnd) for s in sents]
    if len(sents) > 3 and rnd.random() < 0.8:        # move one sentence to a new place
        s = sents.pop(rnd.randrange(len(sents)))
        sents.insert(rnd.randrange(len(sents) + 1), s)
    out = synonym_replace_attack(" ".join(sents), fraction=0.5)
    out = _drop_fillers(out, rnd, 0.8)
    out = _swap_adjacent_words(out, rnd, 0.04)
    return _fix_sentence_case(out)


def llm_rewrite_attack(text: str, use_model: bool = False) -> str:
    if use_model:
        return _try_model(_model_rewrite, text, _rule_rewrite)
    return _rule_rewrite(text)


# ---------------------------------------------------------------------------
# 5-7. structural attacks from the project synopsis
# ---------------------------------------------------------------------------
def text_shortening_attack(text: str, use_model: bool = False) -> str:
    """Drop roughly 30% of the sentences (keeps at least one)."""
    rnd = _rng(text, 5)
    sents = _split_sentences(text)
    if len(sents) <= 1:
        words = text.split()
        return " ".join(words[: max(5, int(len(words) * 0.7))])
    keep = [s for s in sents if rnd.random() > 0.3] or [sents[0]]
    return " ".join(keep)


def random_deletion_attack(text: str, use_model: bool = False, fraction: float = 0.12) -> str:
    """Delete about 12% of the words at random."""
    rnd = _rng(text, 6)
    words = text.split()
    kept = [w for w in words if rnd.random() > fraction] or words[:1]
    return " ".join(kept)


def sentence_restructure_attack(text: str, use_model: bool = False) -> str:
    """Shuffle sentence order and swap coordinated clauses."""
    rnd = _rng(text, 7)
    sents = [_swap_clauses(s, rnd) for s in _split_sentences(text)]
    rnd.shuffle(sents)
    return " ".join(sents)


ATTACKS: Dict[str, Callable[..., str]] = {
    "Synonym Replacement": synonym_replace_attack,
    "Paraphrasing": paraphrase_attack,
    "Back-Translation": back_translation_attack,
    "LLM Rewriting": llm_rewrite_attack,
    "Text Shortening": text_shortening_attack,
    "Random Word Deletion": random_deletion_attack,
    "Sentence Restructuring": sentence_restructure_attack,
}
