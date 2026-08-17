/** The real-image counterpart of the mock's <image-slot>.
 *
 *  Renders place.image_url over the sand placeholder, and on any load failure falls back to the
 *  placeholder plus a category label. A card can therefore never show a broken image (spec §9),
 *  whether image_url is null, wrong, or the network is down.
 */

import { useEffect, useState } from 'react'

const CATEGORY_LABELS: Record<string, string> = {
  park: 'Park',
  waterpark: 'Waterpark',
  theme_park: 'Theme park',
  museum: 'Culture',
  aquarium: 'Wildlife',
  beach: 'Beach',
  adventure: 'Adventure',
  casual_dining: 'Dining',
  fine_dining: 'Fine dining',
  mall: 'Shopping',
  show: 'Show',
  cruise: 'Cruise',
}

export function categoryLabel(category: string): string {
  return CATEGORY_LABELS[category] ?? category.replace(/_/g, ' ')
}

interface Props {
  src: string | null
  category: string
  alt: string
  size: number
  radius?: number
  className?: string
}

export function Thumb({ src, category, alt, size, radius = 13, className }: Props) {
  const [failed, setFailed] = useState(false)

  // A new place in the same card slot deserves a fresh attempt at its image.
  useEffect(() => setFailed(false), [src])

  const showImage = Boolean(src) && !failed

  return (
    <div
      className={`thumb${className ? ` ${className}` : ''}`}
      style={{ width: size, height: size, borderRadius: radius }}
      aria-hidden={showImage ? undefined : 'true'}
    >
      {showImage ? (
        <img src={src!} alt={alt} onError={() => setFailed(true)} loading="lazy" />
      ) : (
        <span className="thumb__label">{categoryLabel(category)}</span>
      )}
    </div>
  )
}
