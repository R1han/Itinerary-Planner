import { useEffect, useState } from 'react'
import { setUnauthorizedHandler } from './api/client'
import { AppShell } from './components/AppShell'
import { AuthPage } from './pages/AuthPage'
import { useStore } from './state/store'

export default function App() {
  const user = useStore((s) => s.user)
  const booting = useStore((s) => s.booting)
  const boot = useStore((s) => s.boot)
  const signOut = useStore((s) => s.signOut)
  const [mode, setMode] = useState<'login' | 'register'>('login')

  useEffect(() => {
    // A 401 anywhere drops straight back to the auth screen — the route guard is this one check.
    setUnauthorizedHandler(signOut)
    void boot()
  }, [boot, signOut])

  if (booting) {
    return (
      <div className="auth">
        <p style={{ color: 'var(--ink-5)' }}>Loading Rihla…</p>
      </div>
    )
  }

  return user ? <AppShell /> : <AuthPage mode={mode} onSwitch={setMode} />
}
