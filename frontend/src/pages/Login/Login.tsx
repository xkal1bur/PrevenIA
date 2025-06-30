
import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import ParticlesComponent from '../../components/particles'
import './Login.css'

const Login: React.FC = () => {
  const [email, setEmail] = useState<string>('')
  const [password, setPassword] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)

  const navigate = useNavigate()

  const validateForm = (): boolean => {
    if (!email || !password) {
      setError('Email and password are required')
      return false
    }
    setError('')
    return true
  }

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!validateForm()) return

    setLoading(true)
    setError('')

    try {
      const response = await fetch('http://52.1.220.84:8000/login/doctor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ correo: email, password: password }),
      })

      setLoading(false)

      if (response.ok) {
        const data: { access_token: string; token_type: string } = await response.json()
        localStorage.setItem('token', data.access_token)
        navigate('/home')
      } else {
        const errorData = await response.json()
        setError(errorData.detail || 'Authentication failed!')
      }
    } catch (err) {
      setLoading(false)
      setError('An error occurred. Please try again later.')
      console.error(err)
    }
  }

  return (
    <div className="login-container">
      <ParticlesComponent id="particles" />
      <form onSubmit={handleSubmit} className="login-form">
        <img src="/images/Logo.png" alt="Logo" className="login-logo" />
        <div className="input-field">
          <label className="input-label">Email:</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="input-box"
            placeholder="doctor@example.com"
            required
          />
        </div>
        <div className="input-field">
          <label className="input-label">Password:</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="input-box"
            placeholder="••••••••"
            required
          />
        </div>
        <button type="submit" disabled={loading} className="login-button">
          {loading ? 'Logging in...' : 'Login'}
        </button>
        {error && <p className="error-message">{error}</p>}
      </form>
    </div>
  )
}

export default Login