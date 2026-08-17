/** Login and registration. Not in the design mock — built from the same tokens. */

import { useState } from 'react'
import { api } from '../api/client'
import { Sparkle } from '../components/icons'
import { useStore } from '../state/store'

const DEMO_ACCOUNTS = [
  { email: 'demo1@rihla.app', password: 'demo123', label: 'Family of four', hint: 'kids aged 7 and 13' },
  { email: 'demo2@rihla.app', password: 'demo123', label: 'Couple', hint: 'adults only, romantic' },
]

interface Props {
  mode: 'login' | 'register'
  onSwitch: (mode: 'login' | 'register') => void
}

export function AuthPage({ mode, onSwitch }: Props) {
  const signIn = useStore((s) => s.signIn)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  const submit = async (credentials?: { email: string; password: string }) => {
    const identity = credentials ?? { email, password }
    setBusy(true)
    setProblem(null)
    try {
      const result =
        mode === 'register' && !credentials
          ? await api.register(identity.email, identity.password, name.trim() || 'Traveller')
          : await api.login(identity.email, identity.password)
      await signIn(result.access_token, result.user)
    } catch (error) {
      setProblem(error instanceof Error ? error.message : 'That did not work.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="auth">
      <form
        className="auth__card"
        onSubmit={(event) => {
          event.preventDefault()
          void submit()
        }}
      >
        <div className="auth__brand">
          <div className="brand__mark">
            <Sparkle />
          </div>
          <span className="brand__name">Rihla</span>
        </div>

        <h1>{mode === 'login' ? 'Welcome back' : 'Create your account'}</h1>
        <p className="auth__sub">
          Plan complete UAE itineraries around the things already on your calendar.
        </p>

        {mode === 'register' && (
          <div className="field">
            <label htmlFor="name">Your name</label>
            <input id="name" value={name} onChange={(event) => setName(event.target.value)} />
          </div>
        )}

        <div className="field">
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </div>

        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            required
            minLength={6}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </div>

        {problem && <div className="error-text">{problem}</div>}

        <button className="auth__submit" type="submit" disabled={busy}>
          {busy ? 'One moment…' : mode === 'login' ? 'Sign in' : 'Create account'}
        </button>

        <div className="auth__alt">
          {mode === 'login' ? (
            <>
              No account?{' '}
              <button type="button" onClick={() => onSwitch('register')}>
                <u>Create one</u>
              </button>
            </>
          ) : (
            <>
              Already registered?{' '}
              <button type="button" onClick={() => onSwitch('login')}>
                <u>Sign in</u>
              </button>
            </>
          )}
        </div>

        <div className="auth__demo">
          <span className="auth__demo-label">Demo accounts</span>
          {DEMO_ACCOUNTS.map((account) => (
            <button
              key={account.email}
              type="button"
              disabled={busy}
              onClick={() => void submit({ email: account.email, password: account.password })}
            >
              <strong>{account.label}</strong> <span>— {account.hint}</span>
            </button>
          ))}
        </div>
      </form>
    </div>
  )
}
