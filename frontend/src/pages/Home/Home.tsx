import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import Sidebar from '../../components/Sidebar'
import Topbar from '../../components/Topbar'
import MonthlyNewChart from '../../components/MonthlyNewChart'
import DoctorCalendar from '../../components/DoctorCalendar'

import './Home.css'

interface Paciente {
  id: number
  dni: string
  nombres: string
  apellidos: string
  edad: number
  celular: string
  correo: string
  foto?: string | null
  doctor_id: number
  created_at: string
}

interface ActiveStats {
  total: number
  active: number
  inactive: number
}

interface BucketUsage {
  used_gb: number
  total_gb: number
}

const Home: React.FC = () => {
  const navigate = useNavigate()
  const [doctorId, setDoctorId] = useState<number | null>(null)
  const [doctorName, setDoctorName] = useState<string>('')
  const [clinicName, setClinicName] = useState<string>('')
  const [loading, setLoading] = useState(true)
  const [pacientes, setPacientes] = useState<Paciente[]>([])
  const [errorPacientes, setErrorPacientes] = useState<string | null>(null)
  const [activeStats, setActiveStats] = useState<ActiveStats | null>(null)
  const [bucketUsage, setBucketUsage] = useState<BucketUsage | null>(null)

  const handleLogout = () => {
    localStorage.removeItem('token')
    navigate('/')
  }

  // 1) obtener perfil del doctor
  useEffect(() => {
    const fetchDoctorProfile = async () => {
      const token = localStorage.getItem('token')
      if (!token) {
        navigate('/')
        return
      }
      try {
        const r = await axios.get('http://localhost:8000/doctors/me', {
          headers: { Authorization: `Bearer ${token}` }
        })
        setDoctorId(r.data.id)
        setDoctorName(r.data.nombre)
        setClinicName(r.data.clinic_name)
        setLoading(false)
      } catch {
        localStorage.removeItem('token')
        navigate('/')
      }
    }
    fetchDoctorProfile()
  }, [navigate])

  // 2) obtener pacientes, stats de actividad y uso de bucket
  useEffect(() => {
    if (doctorId === null) return
    const token = localStorage.getItem('token')
    axios
      .get<Paciente[]>(`http://localhost:8000/pacientes/doctor/${doctorId}`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      .then(r => setPacientes(r.data))
      .catch(() => setErrorPacientes('No se pudo cargar la lista'))
    axios
      .get<ActiveStats>(`http://localhost:8000/pacientes/stats/active_inactive?days=30`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      .then(r => setActiveStats(r.data))
      .catch(() => setActiveStats(null))
    axios
      .get<BucketUsage>(`http://localhost:8000/stats/bucket_usage`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      .then(r => setBucketUsage(r.data))
      .catch(() => setBucketUsage(null))
  }, [doctorId])

  if (loading) {
    return (
      <div className="home-loading">
        <p>Cargando datos del doctor…</p>
      </div>
    )
  }

  const total = pacientes.length

  // nuevos en últimos 30 días
  const nuevos30 = (() => {
    const ahora = new Date()
    const hace30 = new Date(ahora.getTime() - 30 * 24 * 60 * 60 * 1000)
    return pacientes.filter(p => new Date(p.created_at) >= hace30).length
  })()

  // estadísticas de edades
  const estadEdades = (() => {
    if (!total) return null
    const edades = pacientes.map(p => p.edad)
    const min = Math.min(...edades)
    const max = Math.max(...edades)
    const prom = Math.round((edades.reduce((a, b) => a + b, 0) / edades.length) * 10) / 10
    const rangos = {
      '0-17': edades.filter(e => e <= 17).length,
      '18-30': edades.filter(e => e >= 18 && e <= 30).length,
      '31-50': edades.filter(e => e >= 31 && e <= 50).length,
      '51-70': edades.filter(e => e >= 51 && e <= 70).length,
      '71+': edades.filter(e => e >= 71).length
    }
    return { min, max, prom, rangos }
  })()

  return (
    <div className="home-container">
      <Sidebar onLogout={handleLogout} />

      <div className="home-content">
        <Topbar clinicName={clinicName} doctorName={doctorName} />
        <div className="dash">
          {/* ─── Cabecera ─── */}
          <div className="dashboard-top">
            <h2 className="dashboard-title">Dashboard - Dr. {doctorName}</h2>
            <h3 className="ultimos-title">Últimos Pacientes Registrados</h3>
          </div>

          {errorPacientes && (
            <div className="error-message">{errorPacientes}</div>
          )}

          <div className="statistics-container">
            {/* Total */}
            <div className="stat-card">
              <div className="stat-header"><h3>Total de Pacientes</h3></div>
              <div className="stat-content">
                <div className="stat-number">{total}</div>
                <div className="stat-description">
                  {total === 1 ? 'Paciente Registrado' : 'Pacientes Registrados'}
                </div>
              </div>
            </div>

            {/* Nuevos */}
            <div className="stat-card">
              <div className="stat-header"><h3>Nuevos (30 días)</h3></div>
              <div className="stat-content">
                <div className="stat-number">+{nuevos30}</div>
                <div className="stat-description">Pacientes Nuevos</div>
              </div>
            </div>

            {/* Actividad */}
            {activeStats && total > 0 && (
              <div className="stat-card">
                <div className="stat-header"><h3>Actividad (30 días)</h3></div>
                <div className="stat-content">
                  <div className="age-stat">
                    <span className="age-label">Activos:</span>
                    <span className="age-value">
                      {Math.round((activeStats.active / total) * 100)}%
                    </span>
                  </div>
                  <div className="age-stat">
                    <span className="age-label">Inactivos:</span>
                    <span className="age-value">
                      {Math.round((activeStats.inactive / total) * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Edades */}
            {estadEdades && (
              <div className="stat-card">
                <div className="stat-header"><h3>Edades</h3></div>
                <div className="stat-content">
                  <div className="age-summary">
                    <div className="age-stat">
                      <span className="age-label">Promedio:</span>
                      <span className="age-value">{estadEdades.prom} años</span>
                    </div>
                    <div className="age-stat">
                      <span className="age-label">Rango:</span>
                      <span className="age-value">
                        {estadEdades.min} - {estadEdades.max} años
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Uso de Bucket */}
            {bucketUsage && (
              <div className="stat-card">
                <div className="stat-header"><h3>Uso de Bucket</h3></div>
                <div className="stat-content">
                  <div className="stat-number">
                    {bucketUsage.used_gb} GB
                  </div>
                  <div className="stat-description">Almacenamiento usado</div>
                </div>
              </div>
            )}

            {/* Últimos 3 pacientes */}
            {[0, 1, 2].map((_, idx) => {
              const p = pacientes[pacientes.length - 1 - idx]
              return (
                <div className="stat-card" key={idx}>
                  {p ? (
                    <div className="stat-content">
                      <p className="paciente-nombre-mini">
                        {p.nombres} {p.apellidos}
                      </p>
                      <p className="paciente-codigo-mini">DNI: {p.dni}</p>
                      <p className="paciente-edad-mini">Edad: {p.edad} años</p>
                      <p className="paciente-celular-mini">Cel: {p.celular}</p>
                    </div>
                  ) : (
                    <div className="stat-content empty">— vacío —</div>
                  )}
                </div>
              )
            })}

            {/* Distribución Edades */}
            {estadEdades && (
              <div className="stat-card age-distribution">
                <div className="stat-header"><h3>Distribución Edades</h3></div>
                <div className="stat-content">
                  <div className="age-ranges">
                    {Object.entries(estadEdades.rangos).map(([r, c]) => (
                      <div key={r} className="age-range-item">
                        <div className="age-range-label">{r} años</div>
                        <div className="age-range-bar">
                          <div
                            className="age-range-fill"
                            style={{ width: `${(c / total) * 100}%` }}
                          />
                        </div>
                        <div className="age-range-count">{c}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Gráfico mensual */}
          <MonthlyNewChart />

          {/* Calendario de citas */}
          <DoctorCalendar />
        </div>
      </div>
    </div>
  )
}

export default Home
