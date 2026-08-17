/** react-leaflet map with MapTiler raster tiles.
 *
 *  The design's basemap is a hand-drawn cream/teal illustration. Real tiles cannot reproduce the
 *  hand lettering, so the tile layer is pushed toward that palette with a CSS filter (see
 *  `.map .leaflet-tile-pane` in styles.css) and everything drawn on top — pins, routes, the
 *  estimated-leg chip, the popover card, the route badge — matches the mock exactly.
 *
 *  Without VITE_MAPTILER_KEY it falls back to OpenStreetMap tiles under the same filter, so the
 *  map degrades rather than breaking.
 */

import { useEffect, useMemo } from 'react'
import { MapContainer, Marker, Polyline, Popup, TileLayer, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useStore } from '../state/store'
import type { Day, Slot } from '../types'
import { Thumb } from './Thumb'

const MAPTILER_KEY = import.meta.env.VITE_MAPTILER_KEY as string | undefined

const TILES = MAPTILER_KEY
  ? {
      url: `https://api.maptiler.com/maps/dataviz-light/{z}/{x}/{y}.png?key=${MAPTILER_KEY}`,
      attribution:
        '&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }
  : {
      url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    }

const UAE_CENTER: [number, number] = [24.9, 55.0]

function pinIcon(index: number, active: boolean): L.DivIcon {
  return L.divIcon({
    className: '',
    html: `<div class="pin${active ? ' pin--active' : ''}"><div class="pin__body"><span>${
      index + 1
    }</span></div></div>`,
    iconSize: [34, 40],
    iconAnchor: [17, 38],
    popupAnchor: [0, -34],
  })
}

/** Keep the viewport on the active day's stops as the day or plan changes. */
function FitBounds({ day }: { day: Day | null }) {
  const map = useMap()

  useEffect(() => {
    if (!day?.slots.length) {
      map.setView(UAE_CENTER, 8)
      return
    }
    const bounds = L.latLngBounds(day.slots.map((slot) => [slot.place.lat, slot.place.lng]))
    map.fitBounds(bounds, { padding: [56, 56], maxZoom: 13, animate: true })
  }, [day, map])

  return null
}

/** Pan to a slot when it is selected in the strip (spec §9: click pans and zooms). */
function PanToSelected({ slots }: { slots: Slot[] }) {
  const map = useMap()
  const selectedSlotId = useStore((s) => s.selectedSlotId)

  useEffect(() => {
    const slot = slots.find((entry) => entry.id === selectedSlotId)
    if (slot) map.panTo([slot.place.lat, slot.place.lng], { animate: true })
  }, [selectedSlotId, slots, map])

  return null
}

function midpoint(path: [number, number][]): [number, number] {
  return path[Math.floor(path.length / 2)] ?? path[0]
}

/** The "~50 min (estimated)" chip, pinned to the middle of a dashed leg as in the design. */
function estimatedChipIcon(minutes: number): L.DivIcon {
  return L.divIcon({
    className: '',
    html:
      `<div class="estimated-chip"><strong>~${minutes} min</strong><span>(estimated)</span></div>`,
    iconSize: [140, 26],
    iconAnchor: [70, 13],
  })
}

export function MapView() {
  const itinerary = useStore((s) => s.itinerary)
  const selectedDay = useStore((s) => s.selectedDay)
  const hoveredSlotId = useStore((s) => s.hoveredSlotId)
  const selectedSlotId = useStore((s) => s.selectedSlotId)
  const setHoveredSlot = useStore((s) => s.setHoveredSlot)
  const setSelectedSlot = useStore((s) => s.setSelectedSlot)

  const day = itinerary?.days[selectedDay] ?? null
  const slots = day?.slots ?? []

  const routes = useMemo(() => {
    if (!day) return []
    const byId = new Map(slots.map((slot) => [slot.id, slot]))
    return day.segments
      .map((segment) => {
        const to = segment.to_slot_id ? byId.get(segment.to_slot_id) : undefined
        if (!to) return null

        // Prefer the provider's real geometry; a haversine estimate ships a straight line, which
        // is exactly what the dashed style is telling the user.
        const from = segment.from_slot_id ? byId.get(segment.from_slot_id) : undefined
        const path: [number, number][] =
          segment.geometry_json && segment.geometry_json.length > 1
            ? segment.geometry_json
            : from
              ? [
                  [from.place.lat, from.place.lng],
                  [to.place.lat, to.place.lng],
                ]
              : []

        return path.length > 1 ? { segment, path } : null
      })
      .filter((entry): entry is { segment: Day['segments'][number]; path: [number, number][] } =>
        Boolean(entry),
      )
  }, [day, slots])

  const estimatedLegs = routes.filter((route) => route.segment.estimated)
  // The mock shows a single chip on the route. One per leg turns into confetti as soon as a day
  // has several estimates, so label the longest and count the rest in the badge.
  const longestEstimated = estimatedLegs.reduce<(typeof estimatedLegs)[number] | null>(
    (longest, route) =>
      !longest || route.segment.duration_min > longest.segment.duration_min ? route : longest,
    null,
  )

  return (
    <div className="map">
      <MapContainer center={UAE_CENTER} zoom={8} scrollWheelZoom zoomControl={false}>
        <TileLayer url={TILES.url} attribution={TILES.attribution} />
        <FitBounds day={day} />
        <PanToSelected slots={slots} />

        {routes.map(({ segment, path }) => (
          <Polyline
            key={segment.id}
            positions={path}
            pathOptions={{
              color: '#0F6B66',
              weight: 3.4,
              opacity: segment.estimated ? 0.85 : 1,
              dashArray: segment.estimated ? '9 8' : undefined,
              lineCap: 'round',
            }}
          />
        ))}

        {longestEstimated && (
          <Marker
            key={`estimated-${longestEstimated.segment.id}`}
            position={midpoint(longestEstimated.path)}
            icon={estimatedChipIcon(longestEstimated.segment.duration_min)}
            interactive={false}
            keyboard={false}
          />
        )}

        {slots.map((slot, index) => (
          <Marker
            key={slot.id}
            position={[slot.place.lat, slot.place.lng]}
            icon={pinIcon(index, slot.id === hoveredSlotId || slot.id === selectedSlotId)}
            eventHandlers={{
              mouseover: () => setHoveredSlot(slot.id),
              mouseout: () => setHoveredSlot(null),
              click: () => setSelectedSlot(slot.id),
            }}
          >
            <Popup autoPan={false}>
              <div className="popover-card">
                <Thumb
                  src={slot.place.image_url}
                  category={slot.place.category}
                  alt={slot.place.name}
                  size={92}
                  radius={0}
                  className="popover-card__thumb"
                />
                <div className="popover-card__body">
                  <div className="popover-card__eyebrow">
                    <span className="popover-card__stop">Stop {index + 1}</span>
                    <span className="popover-card__area">{slot.place.emirate}</span>
                  </div>
                  <div className="popover-card__name">{slot.place.name}</div>
                  <div className="popover-card__meta">
                    <span className="popover-card__time">
                      {slot.start_time}–{slot.end_time}
                    </span>
                    <span className="cost-chip cost-chip--adult">
                      {itinerary?.currency} {Math.round(slot.cost_breakdown.total)}
                    </span>
                  </div>
                </div>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>

      <div className="map__overlay">
        {day && slots.length > 0 && (
          <div className="map__badge">
            <span className="dot" />
            Day {selectedDay + 1} route · {slots.length} stop{slots.length === 1 ? '' : 's'} ·{' '}
            {Math.floor(day.driving_total_min / 60)} h {day.driving_total_min % 60} m driving
          </div>
        )}
        {estimatedLegs.length > 0 && (
          <div className="map__badge" title="No live route available for these legs">
            <span className="dot" style={{ background: 'var(--ink-42)' }} />
            {estimatedLegs.length} estimated leg{estimatedLegs.length === 1 ? '' : 's'}
          </div>
        )}
      </div>
    </div>
  )
}
