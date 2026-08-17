import { useEffect, useRef, useState } from 'react'
import { api, streamChat } from '../api/client'
import { useStore } from '../state/store'
import type { ChatMessage, StreamEvent } from '../types'
import { Arrow, Chevron, Plus, Sparkle } from './icons'

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
        <p style={{ whiteSpace: 'pre-wrap' }}>{message.content}</p>
      </div>
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
        <p style={{ whiteSpace: 'pre-wrap' }}>
          {text}
          <span className="caret" />
        </p>
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
  const notice = useStore((s) => s.notice)
  const setNotice = useStore((s) => s.setNotice)
  const loadItinerary = useStore((s) => s.loadItinerary)
  const applyBudget = useStore((s) => s.applyBudget)
  const itinerary = useStore((s) => s.itinerary)
  const setSheetOpen = useStore((s) => s.setSheetOpen)
  const sheetOpen = useStore((s) => s.sheetOpen)

  const [draft, setDraft] = useState('')
  const [intakeFields, setIntakeFields] = useState<string[]>([])
  const bodyRef = useRef<HTMLDivElement>(null)

  const active = conversations.find((c) => c.id === conversationId)
  const others = conversations.filter((c) => c.id !== conversationId)
  const unreadCount = others.filter((c) => c.unread).length

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages.length, streamedText])

  const send = async () => {
    const text = draft.trim()
    if (!text || streaming) return

    setDraft('')
    setNotice(null)
    setIntakeFields([])
    appendMessage({
      id: Date.now(),
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
    })
    setStreaming(true)
    setStreamedText(() => '')

    let assistant = ''
    let thread = conversationId

    try {
      await streamChat(text, conversationId, (event: StreamEvent) => {
        switch (event.type) {
          case 'conversation':
            thread = event.data.conversation_id
            break
          case 'token':
            assistant += event.data
            setStreamedText((previous) => previous + event.data)
            break
          case 'itinerary_updated':
            void loadItinerary(event.data.itinerary_id)
            break
          case 'budget_updated':
            applyBudget(event.data)
            break
          case 'notice':
            setNotice(event.data.message)
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

        {messages.map((message) => (
          <Bubble key={message.id} message={message} />
        ))}

        {streaming && <StreamingBubble text={streamedText} />}

        {intakeFields.length > 0 && <IntakeChecklist fields={intakeFields} />}

        {notice && <div className="notice-text">{notice}</div>}

        {itinerary && !streaming && (
          <>
            <div className="msg">
              <div className="msg__avatar">
                <Sparkle size={13} />
              </div>
              <div className="bubble bubble--assistant">
                <p>
                  Here&apos;s a {itinerary.days.length}-day plan, budget{' '}
                  <strong>
                    {itinerary.currency} {itinerary.budget.cap.toLocaleString()}
                  </strong>
                  .
                </p>
                <DayChips />
                <p className="summary-note">
                  Estimated total is{' '}
                  <strong>
                    {itinerary.currency} {Math.round(itinerary.budget.total).toLocaleString()}
                  </strong>{' '}
                  —{' '}
                  {itinerary.budget.over_budget
                    ? `${itinerary.currency} ${Math.abs(
                        Math.round(itinerary.budget.remaining),
                      ).toLocaleString()} over.`
                    : `${itinerary.currency} ${Math.round(
                        itinerary.budget.remaining,
                      ).toLocaleString()} under.`}
                </p>
              </div>
            </div>
            <SuggestionChips />
          </>
        )}
      </div>

      <div className="composer">
        <div className="composer__box">
          <Plus size={18} color="rgba(31,42,42,.4)" />
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
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
