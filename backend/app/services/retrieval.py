"""Candidate retrieval: Chroma semantic query → SQL hard filters → scored shortlist (spec §6.2).

Semantic search finds places a keyword query would miss ("my 6-year-old loves animals" → zoos and
aquariums). SQL then applies the constraints that must never be approximated — age limits, price,
emirate, opening hours.

With embeddings unavailable (no OPENAI_API_KEY, or the call failing) this degrades to
deterministic keyword scoring over tags, category and description. Retrieval gets less clever;
it never stops working.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Place
from . import vectors
from .planner import TRIP_RADIUS_KM, PartyProfile, PlaceCandidate, haversine_km
from .tracing import traced

log = logging.getLogger(__name__)

SHORTLIST_SIZE = 80
SEMANTIC_POOL = 200
# Places held back for affordability rather than relevance. Similarity ranking is blind to price,
# so on a large catalog the free beaches and cheap cafes stop making the shortlist at all — and
# the planner then "gives up" on a tight budget when substitutions were available the whole time.
CHEAP_RESERVE_DINING = 4
CHEAP_RESERVE_OTHER = 10

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have i in is it its of on or our that the their
    them they this to was we were what when where which who will with you your my me like loves
    love want would need""".split()
)


def to_candidate(place: Place, similarity: float = 0.0) -> PlaceCandidate:
    """Adapt a SQLAlchemy Place into the frozen dataclass the pure planner consumes."""
    return PlaceCandidate(
        id=place.id,
        name=place.name,
        category=place.category,
        emirate=place.emirate,
        lat=place.lat,
        lng=place.lng,
        price_adult=place.price_adult,
        price_child=place.price_child,
        price_bands=tuple(place.price_bands) if place.price_bands else None,
        min_age=place.min_age,
        open_time=place.open_time,
        close_time=place.close_time,
        avg_duration_min=place.avg_duration_min,
        tags=tuple(place.tags or ()),
        indoor=bool(place.indoor),
        booking_required=bool(place.booking_required),
        closed_months=tuple(place.closed_months or ()),
        kid_score=place.kid_score,
        teen_score=place.teen_score,
        romance_score=place.romance_score,
        similarity=similarity,
    )


@traced("retrieval.semantic", run_type="retriever")
def semantic_similarities(query: str, limit: int = SEMANTIC_POOL) -> dict[int, float]:
    """{place_id: similarity} from Chroma. Empty dict means 'fall back to keywords'."""
    if not query.strip():
        return {}
    try:
        embedding = vectors.embed([query])[0]
    except vectors.EmbeddingUnavailable as exc:
        log.info("semantic retrieval unavailable (%s); using keyword scoring", exc)
        return {}

    try:
        collection = vectors.get_collection(vectors.PLACES_COLLECTION)
        result = collection.query(query_embeddings=[embedding], n_results=limit)
    except Exception:  # noqa: BLE001
        log.exception("Chroma query failed; using keyword scoring")
        return {}

    ids = (result.get("ids") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    similarities: dict[int, float] = {}
    for index, raw_id in enumerate(ids):
        distance = float(distances[index]) if index < len(distances) else 1.0
        # Cosine distance → similarity, clamped so a far match cannot score negative.
        similarities[int(raw_id)] = max(0.0, 1.0 - distance)
    return similarities


def keyword_similarities(query: str, places: Sequence[Place]) -> dict[int, float]:
    """Deterministic fallback: fraction of meaningful query words a place's text matches."""
    tokens = {
        word
        for word in re.findall(r"[a-z]+", query.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }
    if not tokens:
        return {}

    similarities: dict[int, float] = {}
    for place in places:
        haystack = " ".join(
            [place.name, place.category, place.description or "", " ".join(place.tags or ())]
        ).lower()
        hits = sum(1 for token in tokens if token in haystack)
        if hits:
            similarities[place.id] = min(1.0, hits / len(tokens))
    return similarities


@traced("retrieval.candidates", run_type="retriever")
def retrieve_candidates(
    db: Session,
    profile: PartyProfile,
    query: str = "",
    *,
    limit: int = SHORTLIST_SIZE,
    max_price_per_adult: float | None = None,
    emirates: Sequence[str] | None = None,
    origin: tuple[float, float] | None = None,
    radius_km: float = TRIP_RADIUS_KM,
) -> list[PlaceCandidate]:
    """Semantic pool → hard filters → shortlist, ordered by similarity then breadth.

    Geography is applied BEFORE shortlisting, not after. The planner also enforces a trip radius,
    but by then the shortlist has already been chosen — so on a large catalog its slots get spent
    on places two hours away and the day is left without a reachable lunch. Filtering here means
    every shortlisted candidate is somewhere the trip could actually go.

    The shortlist also keeps a spread of categories: handing the planner eighty aquariums because
    the query mentioned animals would starve it of meals and budget relief.
    """
    statement = select(Place).where(Place.min_age <= profile.youngest_age)
    if emirates:
        statement = statement.where(Place.emirate.in_(list(emirates)))
    if max_price_per_adult is not None:
        statement = statement.where(Place.price_adult <= max_price_per_adult)

    places = list(db.scalars(statement))
    if origin is not None:
        reachable = [
            place
            for place in places
            if haversine_km(origin[0], origin[1], place.lat, place.lng) <= radius_km
        ]
        # Only honour the radius if it leaves a workable catalog; a remote start location should
        # produce a longer drive, not an empty plan.
        if len(reachable) >= limit:
            places = reachable
    if not places:
        return []

    similarities = semantic_similarities(query) if query.strip() else {}
    if not similarities:
        similarities = keyword_similarities(query, places)

    candidates = [to_candidate(place, similarities.get(place.id, 0.0)) for place in places]

    # Keep the whole catalog when it is small enough to hand over wholesale.
    if len(candidates) <= limit:
        return candidates

    ranked = sorted(candidates, key=lambda c: c.similarity, reverse=True)
    shortlist: list[PlaceCandidate] = []
    per_category: dict[str, int] = {}
    cap = max(3, limit // 6)

    for candidate in ranked:
        used = per_category.get(candidate.category, 0)
        if used >= cap:
            continue
        shortlist.append(candidate)
        per_category[candidate.category] = used + 1
        if len(shortlist) >= limit:
            break

    # Top up from whatever is left if the per-category cap left us short.
    if len(shortlist) < limit:
        chosen = {c.id for c in shortlist}
        shortlist.extend(c for c in ranked if c.id not in chosen)

    shortlist = shortlist[:limit]
    return _with_budget_relief(shortlist, candidates)


def _with_budget_relief(shortlist: list[PlaceCandidate], all_candidates: list[PlaceCandidate]):
    """Guarantee the cheapest options survive shortlisting.

    The catalog is built so that every category has budget entries, precisely so a tight cap is
    met by substitution rather than failure (spec §10). That property is only useful if those
    entries actually reach the planner, so the cheapest few are added back regardless of how they
    ranked on similarity.
    """
    from .planner import DINING_CATEGORIES

    chosen = {c.id for c in shortlist}

    def cheapest(pool, count):
        return sorted(pool, key=lambda c: (c.price_adult, c.price_child))[:count]

    dining = [c for c in all_candidates if c.category in DINING_CATEGORIES]
    other = [c for c in all_candidates if c.category not in DINING_CATEGORIES]

    for candidate in cheapest(dining, CHEAP_RESERVE_DINING) + cheapest(other, CHEAP_RESERVE_OTHER):
        if candidate.id not in chosen:
            shortlist.append(candidate)
            chosen.add(candidate.id)

    return shortlist


def query_for(profile: PartyProfile, event_title: str = "", notes: str = "") -> str:
    """Build the retrieval query from the party itself, so it works with no chat at all."""
    parts: list[str] = []
    if event_title:
        parts.append(event_title)
    if notes:
        parts.append(notes)

    if profile.children_ages:
        ages = ", ".join(str(age) for age in profile.children_ages)
        parts.append(f"family trip with children aged {ages}")
        if any(age < 8 for age in profile.children_ages):
            parts.append("young children, parks, zoos, aquariums, gentle indoor activities")
        if any(13 <= age <= 17 for age in profile.children_ages):
            parts.append("teenagers, waterparks, theme parks, adventure activities")
    elif profile.event_type == "anniversary":
        parts.append("romantic evening for two, fine dining, sunset views, adults only")
    else:
        parts.append("adults exploring, dining and sightseeing")

    return ". ".join(parts)
