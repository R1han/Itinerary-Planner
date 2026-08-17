/** The 62px thread rail: new plan, one avatar per plan with its unread dot, and all-plans. */

import { useStore } from '../state/store'
import { Lines, Plus } from './icons'

function initial(title: string): string {
  const trimmed = title.trim()
  return trimmed ? trimmed[0].toUpperCase() : 'R'
}

export function ThreadRail() {
  const conversations = useStore((s) => s.conversations)
  const conversationId = useStore((s) => s.conversationId)
  const selectConversation = useStore((s) => s.selectConversation)
  const startConversation = useStore((s) => s.startConversation)
  const panel = useStore((s) => s.panel)
  const setPanel = useStore((s) => s.setPanel)

  return (
    <nav className="rail" aria-label="Plans">
      <button className="rail__new" onClick={startConversation} title="New plan" aria-label="New plan">
        <Plus color="#F7F1E7" />
      </button>

      <div className="rail__rule" />

      <div className="rail__list">
        {conversations.map((conversation) => {
          const active = conversation.id === conversationId
          return (
            <button
              key={conversation.id}
              className={`rail__item${active ? ' rail__item--active' : ''}`}
              title={conversation.title}
              aria-label={`${conversation.title}${conversation.unread ? ' — unread updates' : ''}`}
              aria-current={active ? 'true' : undefined}
              onClick={() => selectConversation(conversation.id)}
            >
              <span>{initial(conversation.title)}</span>
              {conversation.unread && !active && <span className="rail__unread" />}
              {conversation.unread && active && <span className="rail__unread" />}
            </button>
          )
        })}
      </div>

      <div className="spacer" />

      <button
        className="rail__all"
        title="All plans and events"
        aria-label="All plans and events"
        aria-pressed={panel === 'events'}
        onClick={() => setPanel(panel === 'events' ? null : 'events')}
      >
        <Lines color="rgba(31,42,42,.55)" />
      </button>
    </nav>
  )
}
