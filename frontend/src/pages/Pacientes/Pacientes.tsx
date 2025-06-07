// src/pages/Pacientes/Pacientes.tsx

import React, { useState, useEffect } from 'react'
import type { ChangeEvent, FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { FiPlus, FiX } from 'react-icons/fi'

import Sidebar from '../../components/Sidebar'
import Topbar from '../../components/Topbar'
import './Pacientes.css'

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

interface FormDataState {
  dni: string
  nombres: string
  apellidos: string
  edad: string
  celular: string
  correo: string
  foto: File | null
}

const Pacientes: React.FC = () => {
  const navigate = useNavigate()

  const [doctorId, setDoctorId] = useState<number | null>(null)
  const [doctorName, setDoctorName] = useState<string>('')
  const [clinicName, setClinicName] = useState<string>('')

  const [pacientes, setPacientes] = useState<Paciente[]>([])
  const [errorPacientes, setErrorPacientes] = useState<string | null>(null)

  const [showModal, setShowModal] = useState<boolean>(false)
  const [formData, setFormData] = useState<FormDataState>({
    dni: '',
    nombres: '',
    apellidos: '',
    edad: '',
    celular: '',
    correo: '',
    foto: null,
  })
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  /* Función para cerrar sesión */
  const handleLogout = (): void => {
    localStorage.removeItem('token')
    navigate('/')
  }

  /* 1) Al montar, validar token y obtener datos del doctor */
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
      })
      .catch((err) => {
        console.error('Error al obtener perfil del doctor:', err)
        localStorage.removeItem('token')
        navigate('/')
      })
  }, [navigate])

  /* 2) Cuando tengamos doctorId, obtener lista de pacientes de ese doctor */
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

  /* Manejador de cambios en el formulario de creación */
  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value, files } = e.target
    if (name === 'foto' && files) {
      setFormData((prev) => ({ ...prev, foto: files[0] }))
    } else {
      setFormData((prev) => ({ ...prev, [name]: value }))
    }
  }

  /* Enviar formulario para crear nuevo paciente */
  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setIsSubmitting(true)
    setSubmitError(null)

    try {
      if (doctorId === null) {
        setSubmitError(
          'Aún no se ha cargado tu perfil. Recarga la página e inténtalo de nuevo.'
        )
        setIsSubmitting(false)
        return
      }

      const token = localStorage.getItem('token')
      const data = new FormData()
      data.append('dni', formData.dni)
      data.append('nombres', formData.nombres)
      data.append('apellidos', formData.apellidos)
      data.append('edad', formData.edad)
      data.append('celular', formData.celular)
      data.append('correo', formData.correo)
      if (formData.foto) {
        data.append('foto', formData.foto)
      }
      data.append('doctor_id', doctorId.toString())

      const respCreate = await axios.post<Paciente>(
        'http://localhost:8000/register/paciente',
        data,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${token}`,
          },
        }
      )

      const nuevoPaciente = respCreate.data
      setPacientes((prev) => [...prev, nuevoPaciente])

      setShowModal(false)
      setFormData({
        dni: '',
        nombres: '',
        apellidos: '',
        edad: '',
        celular: '',
        correo: '',
        foto: null,
      })
    } catch (err) {
      console.error('Error al crear paciente:', err)
      setSubmitError('Error al crear paciente. Verifica los datos.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="home-container">
      {/* Sidebar con botón de cierre de sesión */}
      <Sidebar onLogout={handleLogout} />

      <div className="home-content">
        {/* Topbar con datos de clínica y doctor */}
        <Topbar clinicName={clinicName} doctorName={doctorName} />

        <main className="home-main">
          <div className="pacientes-header">
            <h2>Pacientes de {doctorName}</h2>
            <button
              className="btn-add-paciente"
              onClick={() => setShowModal(true)}
            >
              <FiPlus size={18} /> Agregar
            </button>
          </div>

          {errorPacientes && (
            <p style={{ color: 'red', marginBottom: '1rem' }}>
              {errorPacientes}
            </p>
          )}

          {pacientes.length === 0 ? (
            <p className="no-pacientes-text">
              No tienes pacientes registrados aún.
            </p>
          ) : (
            <div className="pacientes-grid">
              {pacientes.map((p) => (
                <div key={p.id} className="paciente-card">
                  <div 
                    className="paciente-content"
                    onClick={() => navigate(`/perfil/${p.dni}`)}
                  >
                    <div className="paciente-photo">
                      {p.foto ? (
                        <img
                          src={`http://localhost:8000/static/pacientes/${p.foto}`}
                          alt={`${p.nombres} ${p.apellidos}`}
                        />
                      ) : (
                        <div className="no-photo-placeholder" />
                      )}
                    </div>
                    <div className="paciente-info">
                      <h3 className="paciente-nombre">
                        {p.nombres} {p.apellidos}
                      </h3>
                      <p className="paciente-codigo">DNI: {p.dni}</p>
                      <p className="paciente-edad">Edad: {p.edad} años</p>
                      <p className="paciente-celular">Celular: {p.celular}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>

      {/* Modal para agregar nuevo paciente */}
      {showModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <button
              className="modal-close-btn"
              onClick={() => {
                setShowModal(false)
                setSubmitError(null)
              }}
            >
              <FiX size={24} />
            </button>
            <h3>Agregar Nuevo Paciente</h3>
            <form className="paciente-form" onSubmit={handleSubmit}>
              <label>
                DNI:
                <input
                  type="text"
                  name="dni"
                  value={formData.dni}
                  onChange={handleInputChange}
                  required
                />
              </label>
              <label>
                Nombres:
                <input
                  type="text"
                  name="nombres"
                  value={formData.nombres}
                  onChange={handleInputChange}
                  required
                />
              </label>
              <label>
                Apellidos:
                <input
                  type="text"
                  name="apellidos"
                  value={formData.apellidos}
                  onChange={handleInputChange}
                  required
                />
              </label>
              <label>
                Edad:
                <input
                  type="number"
                  name="edad"
                  value={formData.edad}
                  onChange={handleInputChange}
                  required
                  min="0"
                />
              </label>
              <label>
                Celular:
                <input
                  type="text"
                  name="celular"
                  value={formData.celular}
                  onChange={handleInputChange}
                  required
                />
              </label>
              <label>
                Correo:
                <input
                  type="email"
                  name="correo"
                  value={formData.correo}
                  onChange={handleInputChange}
                  required
                />
              </label>
              <label>
                Foto:
                <input
                  type="file"
                  name="foto"
                  accept="image/*"
                  onChange={handleInputChange}
                />
              </label>

              {submitError && (
                <p className="submit-error-text">{submitError}</p>
              )}

              <button
                type="submit"
                className="btn-submit-paciente"
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Guardando...' : 'Guardar'}
              </button>
            </form>
          </div>
        </div>
      )}

    </div>
  )
}

export default Pacientes
