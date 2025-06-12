// src/pages/Home/Home.tsx

import React, { useState, useEffect } from 'react'
import Sidebar from '../../components/Sidebar'
import Topbar from '../../components/Topbar'
import './Home.css'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

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
}

const Home: React.FC = () => {
  const navigate = useNavigate()
  const [doctorId, setDoctorId] = useState<number | null>(null)
  const [doctorName, setDoctorName] = useState<string>('')
  const [clinicName, setClinicName] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)
  const [pacientes, setPacientes] = useState<Paciente[]>([])
  const [errorPacientes, setErrorPacientes] = useState<string | null>(null)

  const handleLogout = () => {
    localStorage.removeItem('token')
    navigate('/')
  }

  // Obtener datos del doctor
  useEffect(() => {
    const token = localStorage.getItem('token')
    if (!token) {
      navigate('/')
      return
    }

    axios
      .get('http://localhost:8000/doctors/me', {
        headers: { Authorization: `Bearer ${token}` },
      })
      .then((resp) => {
        setDoctorId(resp.data.id)
        setDoctorName(resp.data.nombre)
        setClinicName(resp.data.clinic_name)
        setLoading(false)
      })
      .catch((err) => {
        console.error('Error al obtener perfil del doctor:', err)
        localStorage.removeItem('token')
        navigate('/')
      })
  }, [navigate])

  // Obtener pacientes cuando tengamos el doctorId
  useEffect(() => {
    if (doctorId === null) return

    const token = localStorage.getItem('token')
    if (!token) return

    axios
      .get<Paciente[]>(`http://localhost:8000/pacientes/doctor/${doctorId}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      .then((response) => {
        setPacientes(response.data)
      })
      .catch((err) => {
        console.error('Error al obtener pacientes:', err)
        setErrorPacientes('No se pudo cargar la lista de pacientes.')
      })
  }, [doctorId])

  // Calcular estadísticas de edades
  const calcularEstadisticasEdades = () => {
    if (pacientes.length === 0) return null

    const edades = pacientes.map(p => p.edad)
    const edadMinima = Math.min(...edades)
    const edadMaxima = Math.max(...edades)
    const edadPromedio = edades.reduce((sum, edad) => sum + edad, 0) / edades.length

    // Agrupar por rangos de edad
    const rangos = {
      '0-17': edades.filter(edad => edad >= 0 && edad <= 17).length,
      '18-30': edades.filter(edad => edad >= 18 && edad <= 30).length,
      '31-50': edades.filter(edad => edad >= 31 && edad <= 50).length,
      '51-70': edades.filter(edad => edad >= 51 && edad <= 70).length,
      '71+': edades.filter(edad => edad >= 71).length,
    }

    return {
      edadMinima,
      edadMaxima,
      edadPromedio: Math.round(edadPromedio * 10) / 10,
      rangos
    }
  }

  const estadisticasEdades = calcularEstadisticasEdades()

  if (loading) {
    return (
      <div className="home-loading">
        <p>Cargando datos del doctor…</p>
      </div>
    )
  }

  return (
    <div className="home-container">
      <Sidebar onLogout={handleLogout} />

      <div className="home-content">
        <Topbar clinicName={clinicName} doctorName={doctorName} />

        <main className="home-main">
          <div className="dashboard-header">
            <h2>Dashboard - Dr. {doctorName}</h2>
          </div>

          {errorPacientes && (
            <div className="error-message">
              {errorPacientes}
            </div>
          )}

          <div className="statistics-container">
            {/* Estadística de cantidad de pacientes */}
            <div className="stat-card">
              <div className="stat-header">
                <h3>Total de Pacientes</h3>
              </div>
              <div className="stat-content">
                <div className="stat-number">{pacientes.length}</div>
                <div className="stat-description">
                  {pacientes.length === 1 ? 'paciente registrado' : 'pacientes registrados'}
                </div>
              </div>
            </div>

            {/* Estadísticas de edades */}
            {estadisticasEdades && (
              <div className="stat-card">
                <div className="stat-header">
                  <h3>Edades de Pacientes</h3>
                </div>
                <div className="stat-content">
                  <div className="age-summary">
                    <div className="age-stat">
                      <span className="age-label">Promedio:</span>
                      <span className="age-value">{estadisticasEdades.edadPromedio} años</span>
                    </div>
                    <div className="age-stat">
                      <span className="age-label">Rango:</span>
                      <span className="age-value">{estadisticasEdades.edadMinima} - {estadisticasEdades.edadMaxima} años</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Distribución por rangos de edad */}
            {estadisticasEdades && (
              <div className="stat-card age-distribution">
                <div className="stat-header">
                  <h3>Distribución por Edades</h3>
                </div>
                <div className="stat-content">
                  <div className="age-ranges">
                    {Object.entries(estadisticasEdades.rangos).map(([rango, cantidad]) => (
                      <div key={rango} className="age-range-item">
                        <div className="age-range-label">{rango} años</div>
                        <div className="age-range-bar">
                          <div 
                            className="age-range-fill"
                            style={{ 
                              width: `${pacientes.length > 0 ? (cantidad / pacientes.length) * 100 : 0}%` 
                            }}
                          ></div>
                        </div>
                        <div className="age-range-count">{cantidad}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {pacientes.length === 0 && !errorPacientes && (
              <div className="no-data-message">
                <p>No hay pacientes registrados aún.</p>
                <p>Los gráficos aparecerán cuando agregues pacientes.</p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default Home
