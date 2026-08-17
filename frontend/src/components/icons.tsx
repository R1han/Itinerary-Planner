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

export function Gear({ size = 16, color }: IconProps) {
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
      <circle cx="12" cy="12" r="3.2" />
      <path d="M12 3.5v2M12 18.5v2M3.5 12h2M18.5 12h2M6 6l1.4 1.4M16.6 16.6L18 18M18 6l-1.4 1.4M7.4 16.6L6 18" />
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
