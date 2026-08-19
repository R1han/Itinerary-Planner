import { Fragment, useEffect, useRef, useState } from 'react'
import { api, streamChat } from '../api/client'
import { useStore } from '../state/store'
import type { ChatMessage, StreamEvent } from '../types'
import { Arrow, Check, Chevron, Gear, Sparkle } from './icons'
import { Markdown } from './Markdown'

const EMIRATES = [
  'Abu Dhabi',
  'Dubai',
  'Sharjah',
  'Ajman',
  'Umm Al Quwain',
  'Ras Al Khaimah',
  'Fujairah',
]

/** The mock renders each day theme as a coloured chip; the palette cycles per day. */
function DayChips() {
  const itinerary = useStore((s) => s.itinerary)
  if (!itinerary) return null
  return (
    <div className="day-chips">
      {itinerary.days.map((day) => (
        <span key={day.day_index} className={`day-chip day-chip--${day.day_index % 5}`}>
          Day {day.day_index + 1} · {day.theme}
        </span>
      ))}
    </div>
  )
}

function Bubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="msg msg--user">
        <div className="bubble bubble--user">
          <p>{message.content}</p>
        </div>
      </div>
    )
  }
  return (
    <div className="msg">
      <div className="msg__avatar">
        <Sparkle size={13} />
      </div>
      <div className="bubble bubble--assistant">
        <Markdown>{message.content}</Markdown>
      </div>
    </div>
  )
}

/** What the assistant is doing, and with what inputs — one row per tool call.
 *
 *  Tool calls are otherwise invisible: the assistant goes quiet for several seconds while a plan
 *  is built, and this is the difference between "thinking" and "stuck". Labels and outcomes come
 *  from the server, which already knows what the arguments mean. */
function ToolTrace() {
  const activity = useStore((s) => s.toolActivity)
  if (!activity.length) return null

  return (
    <div className="trace" role="status" aria-label="Assistant activity">
      {activity.map((entry) => (
        <div
          key={entry.id}
          className={`trace__row${entry.outcome ? ' trace__row--done' : ''}${
            entry.failed ? ' trace__row--failed' : ''
          }`}
        >
          <span className="trace__icon">
            {entry.outcome ? <Check size={12} /> : <Gear size={12} />}
          </span>
          <span className="trace__label">{entry.label}</span>
          {entry.detail && <span className="trace__detail">{entry.detail}</span>}
          {entry.outcome && <span className="trace__outcome">{entry.outcome}</span>}
        </div>
      ))}
    </div>
  )
}

function StreamingBubble({ text }: { text: string }) {
  return (
    <div className="msg">
      <div className="msg__avatar">
        <Sparkle size={13} />
      </div>
      <div className="bubble bubble--assistant">
        {/* Partial markdown mid-stream renders as literal text until its syntax closes, which is
            the least surprising thing it can do while tokens are still arriving. */}
        <Markdown>{text}</Markdown>
        <span className="caret" />
      </div>
    </div>
  )
}

/** The numbered intake checklist. Rendered from the server's missing_fields, so it shows exactly
 *  what is still outstanding rather than a fixed list. */
function IntakeChecklist({ fields }: { fields: string[] }) {
  const labels: Record<string, string> = {
    adults: 'How many adults?',
    children_ages: 'Kids, and their ages?',
    budget: 'Total budget in AED?',
    dates: 'Which dates?',
    start_location: 'Where are you starting from?',
  }
  return (
    <div className="msg">
      <div className="msg__avatar">
        <Sparkle size={13} />
      </div>
      <div className="bubble bubble--assistant">
        <p>{fields.length === 1 ? 'One more thing and I can draft it:' : 'A few quick things and I&apos;ll draft it:'}</p>
        <div className="intake">
          {fields.map((field, index) => (
            <div className="intake__row" key={field}>
              <span className="intake__num">{index + 1}</span>
              {labels[field] ?? field.replace(/_/g, ' ')}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function SuggestionChips() {
  const itinerary = useStore((s) => s.itinerary)
  const setItinerary = useStore((s) => s.setItinerary)
  const setError = useStore((s) => s.setError)
  const [busy, setBusy] = useState<string | null>(null)

  if (!itinerary?.suggestions.length) return null

  const run = async (id: string, action: () => Promise<void>) => {
    setBusy(id)
    try {
      await action()
    } catch (error) {
      setError(error instanceof Error ? error.message : 'That did not work.')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="suggestions">
      {itinerary.suggestions.map((suggestion) => (
        <button
          key={suggestion.id}
          className="suggestion"
          disabled={busy !== null}
          onClick={() =>
            run(suggestion.id, async () => {
              const updated =
                suggestion.action === 'cheaper_day'
                  ? await api.cheaperDay(itinerary.id, suggestion.day_index ?? 0)
                  : await api.prayerBreaks(itinerary.id)
              setItinerary(updated)
            })
          }
        >
          {busy === suggestion.id ? 'Working…' : suggestion.label}
        </button>
      ))}
    </div>
  )
}

export function ChatPanel() {
  const conversations = useStore((s) => s.conversations)
  const conversationId = useStore((s) => s.conversationId)
  const selectConversation = useStore((s) => s.selectConversation)
  const refreshConversations = useStore((s) => s.refreshConversations)
  const messages = useStore((s) => s.messages)
  const appendMessage = useStore((s) => s.appendMessage)
  const streaming = useStore((s) => s.streaming)
  const setStreaming = useStore((s) => s.setStreaming)
  const streamedText = useStore((s) => s.streamedText)
  const setStreamedText = useStore((s) => s.setStreamedText)
  const startToolActivity = useStore((s) => s.startToolActivity)
  const finishToolActivity = useStore((s) => s.finishToolActivity)
  const clearToolActivity = useStore((s) => s.clearToolActivity)
  const notice = useStore((s) => s.notice)
  const setNotice = useStore((s) => s.setNotice)
  const loadItinerary = useStore((s) => s.loadItinerary)
  const applyBudget = useStore((s) => s.applyBudget)
  const itinerary = useStore((s) => s.itinerary)
  const setSheetOpen = useStore((s) => s.setSheetOpen)
  const sheetOpen = useStore((s) => s.sheetOpen)

  const [draft, setDraft] = useState('')
  const [emirate, setEmirate] = useState('')
  const [intakeFields, setIntakeFields] = useState<string[]>([])
  const bodyRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef<HTMLTextAreaElement>(null)

  const active = conversations.find((c) => c.id === conversationId)
  const others = conversations.filter((c) => c.id !== conversationId)
  const unreadCount = others.filter((c) => c.unread).length
  // The trace belongs to the turn it describes, so it renders under the last thing the user said
  // — above the reply, whether that reply is still streaming or already finished.
  const lastUserIndex = messages.map((m) => m.role).lastIndexOf('user')

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages.length, streamedText])

  // The composer grows with its content: reset to one line, then take the height the text needs.
  // CSS max-height caps it, after which the textarea scrolls.
  useEffect(() => {
    const field = draftRef.current
    if (!field) return
    field.style.height = 'auto'
    field.style.height = `${field.scrollHeight}px`
  }, [draft])

  const send = async () => {
    const text = draft.trim()
    if (!text || streaming) return

    setDraft('')
    setNotice(null)
    setIntakeFields([])
    clearToolActivity()
    appendMessage({
      id: Date.now(),
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
    })
    setStreaming(true)
    setStreamedText(() => '')

    // The backend has no dedicated field for this — it's extracted from the message text by the
    // chat model's own tool-call reasoning, same as any other place/date detail the user types.
    const outgoing = emirate ? `[Starting emirate: ${emirate}] ${text}` : text

    let assistant = ''
    let thread = conversationId

    try {
      await streamChat(outgoing, conversationId, (event: StreamEvent) => {
        switch (event.type) {
          case 'conversation':
            thread = event.data.conversation_id
            break
          case 'token':
            assistant += event.data
            setStreamedText((previous) => previous + event.data)
            break
          case 'tool':
            startToolActivity(event.data)
            break
          case 'tool_done':
            finishToolActivity(event.data.id, event.data.outcome, event.data.failed)
            break
          case 'itinerary_updated':
            // A plan exists, so nothing is outstanding. The assistant can fill a gap itself
            // mid-turn — asked for three adults, it saves the family and retries — and without
            // this the checklist keeps asking a question that has already been answered.
            setIntakeFields([])
            void loadItinerary(event.data.itinerary_id)
            break
          case 'budget_updated':
            applyBudget(event.data)
            break
          case 'intake_required':
            setIntakeFields(event.data.missing_fields)
            break
          case 'error':
            setNotice(event.data.message)
            break
          case 'done':
            break
        }
      })
    } catch (error) {
      assistant =
        assistant || (error instanceof Error ? error.message : 'Something went wrong.')
    } finally {
      setStreaming(false)
      setStreamedText(() => '')
      appendMessage({
        id: Date.now() + 1,
        role: 'assistant',
        content: assistant,
        created_at: new Date().toISOString(),
      })
      await refreshConversations()
      if (thread && thread !== conversationId) await selectConversation(thread)
    }
  }

  return (
    <div className="chat">
      <button
        className="sheet-handle"
        onClick={() => setSheetOpen(!sheetOpen)}
        aria-expanded={sheetOpen}
      >
        {sheetOpen ? 'Hide chat' : 'Ask Rihla'}
        <Chevron size={14} />
      </button>

      <div className="chat__header">
        <div className="chat__title">
          <div className="chat__title-row">
            <strong>{active?.title ?? 'New plan'}</strong>
            <Chevron color="rgba(31,42,42,.45)" />
          </div>
          <span className="chat__subtitle">
            {others.length
              ? `${others.length} more plan${others.length === 1 ? '' : 's'}${
                  unreadCount ? ` · ${unreadCount} with unread updates` : ''
                }`
              : 'Your first plan'}
          </span>
        </div>

        <div className="spacer" />

        <div className="quick-switch">
          {others.slice(0, 2).map((conversation) => (
            <button
              key={conversation.id}
              onClick={() => selectConversation(conversation.id)}
              title={conversation.title}
            >
              {conversation.title}
            </button>
          ))}
        </div>
      </div>

      <div className="chat__body" ref={bodyRef}>
        <div className="chat__divider">
          {new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'short' })}
        </div>

        {messages.length === 0 && !streaming && (
          <div className="msg">
            <div className="msg__avatar">
              <Sparkle size={13} />
            </div>
            <div className="bubble bubble--assistant">
              <p>
                I plan trips around what&apos;s coming up. Ask me{' '}
                <strong>&ldquo;what events are upcoming?&rdquo;</strong> and I&apos;ll offer to plan
                one.
              </p>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <Fragment key={message.id}>
            <Bubble message={message} />
            {index === lastUserIndex && <ToolTrace />}
          </Fragment>
        ))}

        {streaming && <StreamingBubble text={streamedText} />}

        {intakeFields.length > 0 && <IntakeChecklist fields={intakeFields} />}

        {notice && <div className="notice-text">{notice}</div>}

        {/*
          This used to be an assistant-styled bubble reading "Here's an N-day plan, budget ..."
          with the totals under it, rendered from client state after every turn a plan existed.
          Nobody said it. It appeared over a reply that had just explained the change did NOT go
          through, and it is the same bug the server now checks for — a change claimed in chat
          that nothing performed — reproduced where no server-side check can see it.

          The chips stay: they are navigation, not a claim about what happened. What went is the
          prose and the avatar that made derived numbers look like something the assistant said.
        */}
        {itinerary && !streaming && (
          <div className="plan-affordances" aria-label="Plan shortcuts">
            <DayChips />
            <SuggestionChips />
          </div>
        )}
      </div>

      <div className="composer">
        <select
          className="composer__emirate"
          value={emirate}
          onChange={(event) => setEmirate(event.target.value)}
          aria-label="Starting emirate"
        >
          <option value="">Starting emirate (optional)</option>
          {EMIRATES.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
        <div className="composer__box">
          <textarea
            ref={draftRef}
            rows={1}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends; shift+enter is a newline, which the browser inserts for us.
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault()
                void send()
              }
            }}
            placeholder="Ask Rihla to adjust the plan…"
            aria-label="Message Rihla"
            disabled={streaming}
          />
          <button
            className="composer__send"
            onClick={() => void send()}
            disabled={streaming || !draft.trim()}
            aria-label="Send"
          >
            <Arrow />
          </button>
        </div>
      </div>
    </div>
  )
}
