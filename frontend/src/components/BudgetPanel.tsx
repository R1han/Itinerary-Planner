/** The pinned budget bar. Every figure here comes from the server — the client never computes a
 *  total, so what the user sees is always what the planner enforced (spec §6.5). */

import { useStore } from '../state/store'

const SEGMENTS = [
  { key: 'activities', label: 'Activities', colour: 'var(--teal)' },
  { key: 'food', label: 'Food', colour: 'var(--amber)' },
  { key: 'travel', label: 'Travel', colour: 'var(--coral)' },
] as const

export function BudgetPanel() {
  const itinerary = useStore((s) => s.itinerary)
  const selectedDay = useStore((s) => s.selectedDay)
  const setSelectedDay = useStore((s) => s.setSelectedDay)

  if (!itinerary) return null
  const { budget, currency } = itinerary

  // Widths are shares of the cap, not of the total, so the bar visibly fills toward the cap.
  const denominator = Math.max(budget.cap, budget.total, 1)
  const peakDay = Math.max(...budget.per_day, 1)

  return (
    <footer className="budget" aria-label="Budget">
      <div className="budget__block">
        <span className="eyebrow">Trip total</span>
        <div className="budget__total">
          <b>
            {currency} {Math.round(budget.total).toLocaleString()}
          </b>
          <span className="budget__cap">/ {Math.round(budget.cap).toLocaleString()}</span>
        </div>
      </div>

      <div className="budget__middle">
        <div
          className="budget__bar"
          role="img"
          aria-label={`Activities ${Math.round(budget.categories.activities)}, food ${Math.round(
            budget.categories.food,
          )}, travel ${Math.round(budget.categories.travel)} of ${Math.round(budget.cap)}`}
        >
          {SEGMENTS.map((segment) => (
            <div
              key={segment.key}
              style={{
                width: `${(budget.categories[segment.key] / denominator) * 100}%`,
                background: segment.colour,
              }}
            />
          ))}
        </div>

        <div className="budget__legend">
          {SEGMENTS.map((segment) => (
            <span className="legend" key={segment.key}>
              <span className="legend__swatch" style={{ background: segment.colour }} />
              {segment.label} <strong>{Math.round(budget.categories[segment.key]).toLocaleString()}</strong>
            </span>
          ))}
        </div>
      </div>

      <div className="budget__days">
        {budget.per_day.map((amount, index) => (
          <button
            key={index}
            className={`daybar${index === selectedDay ? ' daybar--active' : ''}`}
            onClick={() => setSelectedDay(index)}
            title={`Day ${index + 1}: ${currency} ${Math.round(amount).toLocaleString()}`}
            aria-label={`Show day ${index + 1}, ${currency} ${Math.round(amount)}`}
          >
            <span className="daybar__track">
              <span
                className="daybar__fill"
                style={{ height: `${Math.max(4, (amount / peakDay) * 100)}%` }}
              />
            </span>
            <span className="daybar__label">D{index + 1}</span>
          </button>
        ))}
      </div>

      <div className="divider-v" style={{ height: 44 }} />

      <div className="budget__remaining">
        <span className="eyebrow">{budget.over_budget ? 'Over budget' : 'Remaining'}</span>
        <b className={budget.over_budget ? 'over' : undefined}>
          {currency} {Math.abs(Math.round(budget.remaining)).toLocaleString()}{' '}
          {budget.over_budget ? 'over' : 'left'}
        </b>
      </div>
    </footer>
  )
}
