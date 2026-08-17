/** Inline SVGs, traced from the design mock. Kept as one module so stroke weights and viewBoxes
 *  stay consistent rather than drifting per component. */

interface IconProps {
  size?: number
  color?: string
  className?: string
}

const stroke = (color?: string) => color ?? 'currentColor'

export function Sparkle({ size = 14, color = '#F7F1E7' }: IconProps) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path
        d="M7 1l1.9 3.1L12.5 5 10 7.6l.6 3.7L7 9.6 3.4 11.3 4 7.6 1.5 5l3.6-.9L7 1z"
        fill={color}
      />
    </svg>
  )
}

export function Star({ size = 16, color = '#F2B34A' }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="1.9"
      strokeLinecap="round"
      style={{ marginTop: 1, flex: 'none' }}
      aria-hidden="true"
    >
      <path d="M12 3l2.2 5.6L20 9.5l-4 4.1.9 5.9L12 16.8 7.1 19.5 8 13.6l-4-4.1 5.8-.9L12 3z" />
    </svg>
  )
}

export function Plus({ size = 16, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="2.2"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M12 5v14M5 12h14" />
    </svg>
  )
}

// An eight-tooth cog, generated geometrically rather than drawn by hand so the teeth sit on an
// even pitch. The mock's settings glyph was a circle with radiating spokes, which reads as a
// brightness control rather than settings.
const COG_PATH =
  'M18.84 10.50L21.23 10.87A9.30 9.30 0 0 1 21.23 13.13L18.84 13.50A7.00 7.00 0 0 1 17.90 15.77' +
  'L19.33 17.73A9.30 9.30 0 0 1 17.73 19.33L15.77 17.90A7.00 7.00 0 0 1 13.50 18.84L13.13 21.23' +
  'A9.30 9.30 0 0 1 10.87 21.23L10.50 18.84A7.00 7.00 0 0 1 8.23 17.90L6.27 19.33' +
  'A9.30 9.30 0 0 1 4.67 17.73L6.10 15.77A7.00 7.00 0 0 1 5.16 13.50L2.77 13.13' +
  'A9.30 9.30 0 0 1 2.77 10.87L5.16 10.50A7.00 7.00 0 0 1 6.10 8.23L4.67 6.27' +
  'A9.30 9.30 0 0 1 6.27 4.67L8.23 6.10A7.00 7.00 0 0 1 10.50 5.16L10.87 2.77' +
  'A9.30 9.30 0 0 1 13.13 2.77L13.50 5.16A7.00 7.00 0 0 1 15.77 6.10L17.73 4.67' +
  'A9.30 9.30 0 0 1 19.33 6.27L17.90 8.23A7.00 7.00 0 0 1 18.84 10.50Z'

export function Gear({ size = 16, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="1.6"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={COG_PATH} />
      <circle cx="12" cy="12" r="3.1" />
    </svg>
  )
}

export function Lines({ size = 15, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="1.9"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M5 7h14M5 12h14M5 17h9" />
    </svg>
  )
}

export function Chevron({ size = 13, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="2.2"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M7 10l5 5 5-5" />
    </svg>
  )
}

export function Arrow({ size = 15, color = '#F7F1E7' }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 12h13M12 6l6 6-6 6" />
    </svg>
  )
}

export function Car({ size = 15, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="1.7"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M4 16v-3l2-5h12l2 5v3M4 16h16M7 16v2M17 16v2" />
    </svg>
  )
}

export function Swap({ size = 14, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 8h13l-3-3M20 16H7l3 3" />
    </svg>
  )
}

export function Clock({ size = 14, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="1.8"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="8" />
      <path d="M12 8v4.4l3 1.8" />
    </svg>
  )
}

export function Trash({ size = 14, color = '#B4552E' }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="1.8"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M6 7h12M9 7V5h6v2M8 7l1 12h6l1-12" />
    </svg>
  )
}

export function Check({ size = 13, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 12.5l4.5 4.5L19 7.5" />
    </svg>
  )
}

export function Close({ size = 15, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="2"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  )
}

export function Calendar({ size = 15, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="1.8"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <rect x="3.5" y="5" width="17" height="15" rx="3" />
      <path d="M3.5 10h17M8 3.5v3M16 3.5v3" />
    </svg>
  )
}

export function People({ size = 15, color }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke(color)}
      strokeWidth="1.8"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <circle cx="9" cy="8" r="3.2" />
      <path d="M3.5 19.5c0-3 2.5-5 5.5-5s5.5 2 5.5 5" />
      <path d="M16 6.2a3 3 0 010 5.6M17.5 19.5c0-2.2-.8-3.9-2-5" />
    </svg>
  )
}
