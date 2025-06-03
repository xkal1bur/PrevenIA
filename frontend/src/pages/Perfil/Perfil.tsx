// src/pages/Perfil/Perfil.tsx

import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import Sidebar from '../../components/Sidebar'
import Topbar from '../../components/Topbar'
import { FiUploadCloud } from 'react-icons/fi'
import './Perfil.css'

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

const Perfil: React.FC = () => {
  const { dni } = useParams<{ dni: string }>()
  const navigate = useNavigate()

  const [doctorName, setDoctorName] = useState<string>('')
  const [clinicName, setClinicName] = useState<string>('')

  const [paciente, setPaciente] = useState<Paciente | null>(null)
  const [error, setError] = useState<string>('')

  const [loadingPaciente, setLoadingPaciente] = useState<boolean>(true)

  const handleLogout = (): void => {
    localStorage.removeItem('token')
    navigate('/')
  }

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (!token) {
      navigate('/')
      return
    }

    axios
      .get('http://localhost:8000/doctors/me', {
        headers: { Authorization: `Bearer ${token}` }
      })
      .then((resp) => {
        setDoctorName(resp.data.nombre)
        setClinicName(resp.data.clinic_name)
      })
      .catch((err) => {
        console.error('Error al obtener perfil del doctor:', err)
        localStorage.removeItem('token')
        navigate('/')
      })

    if (dni) {
      axios
        .get<Paciente>(`http://localhost:8000/pacientes/dni/${dni}`, {
          headers: { Authorization: `Bearer ${token}` }
        })
        .then((resp) => {
          setPaciente(resp.data)
          setLoadingPaciente(false)
        })
        .catch((err) => {
          console.error('Error al obtener datos del paciente:', err)
          setError('No se pudo cargar la información del paciente.')
          setLoadingPaciente(false)
        })
    } else {
      setError('DNI inválido en la URL.')
      setLoadingPaciente(false)
    }
  }, [dni, navigate])

  if (loadingPaciente) {
    return (
      <div className="home-loading">
        <p>Cargando datos del paciente…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="home-loading">
        <p style={{ color: 'red' }}>{error}</p>
      </div>
    )
  }

  if (!paciente) {
    return (
      <div className="home-loading">
        <p>No se encontró el paciente.</p>
      </div>
    )
  }

  return (
    <div className="perfil-container">
      <Sidebar onLogout={handleLogout} />

      <div className="perfil-content">
        <Topbar clinicName={clinicName} doctorName={doctorName} />

        <main className="perfil-main">
          <h2 className="perfil-title">Perfil del Paciente</h2>

          <div className="perfil-grid">
            <div className="perfil-left">
              <div className="perfil-card foto-card">
                <img
                  src={
                    paciente.foto
                      ? `http://localhost:8000/static/pacientes/${paciente.foto}`
                      : '/images/placeholder-person.png'
                  }
                  alt={`${paciente.nombres} ${paciente.apellidos}`}
                  className="perfil-photo"
                />
              </div>

              <div className="perfil-card datos-card">
                <h3 className="perfil-nombre">
                  {paciente.nombres} {paciente.apellidos}
                </h3>
                <p>DNI: {paciente.dni}</p>
                <p>Edad: {paciente.edad} años</p>
                <p>Celular: {paciente.celular}</p>
                <p>Correo: {paciente.correo}</p>
              </div>
            </div>

            <div className="perfil-right">
              <div className="perfil-card upload-card">
                <div className="upload-box">
                  <FiUploadCloud size={48} />
                </div>
                <button className="upload-button">Cargar Archivo</button>
              </div>

              <div className="perfil-card files-card">
                <div className="files-box">
                  <p>No hay archivos cargados.</p>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

export default Perfil
