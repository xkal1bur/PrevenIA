import React, { useState, useEffect } from 'react'
import type { ChangeEvent, FormEvent, DragEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { FiPlus, FiX, FiUploadCloud } from 'react-icons/fi'
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
  const [highlightDrag, setHighlightDrag] = useState(false)

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

  /* ----------  handlers drop-zone foto  ---------- */
  const handleDragOver = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setHighlightDrag(true)
  }

  const handleDragLeave = () => setHighlightDrag(false)

  const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setHighlightDrag(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      setFormData(prev => ({ ...prev, foto: file }))
    }
  }

  /* ----------  cambio input genérico con filtrado ---------- */
  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value, files } = e.target

    if (name === 'foto' && files) {
      setFormData(prev => ({ ...prev, foto: files[0] }))
      return
    }

    let sanitized = value

    switch (name) {
      case 'dni':
      case 'celular':
        // sólo dígitos
        sanitized = value.replace(/\D/g, '')
        break
      case 'nombres':
      case 'apellidos':
        // sólo letras y espacios (incluye acentos y ñ)
        sanitized = value.replace(/[^A-Za-zÁÉÍÓÚáéíóúÑñ ]/g, '')
        break
      case 'edad':
        // números sólo
        sanitized = value.replace(/\D/g, '')
        break
      default:
        break
    }

    setFormData(prev => ({ ...prev, [name]: sanitized }))
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

            <h3 className="modal-title">Agregar Nuevo Paciente</h3>

            <form className="paciente-form" onSubmit={handleSubmit}>
              <div className="form-field">
                <label htmlFor="dni">DNI</label>
                <input
                  id="dni"
                  name="dni"
                  value={formData.dni}
                  onChange={handleInputChange}
                  inputMode="numeric"
                  pattern="\d*"
                  maxLength={8}
                  required
                />
              </div>

              <div className="form-field">
                <label htmlFor="nombres">Nombres</label>
                <input
                  id="nombres"
                  name="nombres"
                  value={formData.nombres}
                  onChange={handleInputChange}
                  inputMode="text"
                  pattern="[A-Za-zÁÉÍÓÚáéíóúÑñ ]+"
                  required
                />
              </div>

              <div className="form-field">
                <label htmlFor="apellidos">Apellidos</label>
                <input
                  id="apellidos"
                  name="apellidos"
                  value={formData.apellidos}
                  onChange={handleInputChange}
                  inputMode="text"
                  pattern="[A-Za-zÁÉÍÓÚáéíóúÑñ ]+"
                  required
                />
              </div>

              <div className="form-field">
                <label htmlFor="edad">Edad</label>
                <input
                  id="edad"
                  name="edad"
                  type="number"
                  value={formData.edad}
                  onChange={handleInputChange}
                  min={0}
                  required
                />
              </div>

              <div className="form-field">
                <label htmlFor="celular">Celular</label>
                <input
                  id="celular"
                  name="celular"
                  value={formData.celular}
                  onChange={handleInputChange}
                  inputMode="numeric"
                  pattern="\d*"
                  maxLength={9}
                  required
                />
              </div>

              <div className="form-field">
                <label htmlFor="correo">Correo</label>
                <input
                  id="correo"
                  name="correo"
                  type="email"
                  value={formData.correo}
                  onChange={handleInputChange}
                  required
                />
              </div>

              <div className="form-field full-width">
                <label
                  htmlFor="foto-input"
                  className={`drop-zone ${highlightDrag ? 'drag-over' : ''}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  {formData.foto ? (
                    <div className="drop-zone-filename">
                      📁 {formData.foto.name}
                    </div>
                  ) : (
                    <div className="drop-zone-placeholder">
                      <FiUploadCloud size={40} />
                      <p>Click o arrastra tu foto aquí</p>
                    </div>
                  )}
                  <input
                    id="foto-input"
                    type="file"
                    name="foto"
                    accept="image/*"
                    onChange={handleInputChange}
                    className="file-hidden"
                  />
                </label>
              </div>

              {submitError && <p className="submit-error-text">{submitError}</p>}

              <button type="submit" className="btn-submit-paciente" disabled={isSubmitting}>
                {isSubmitting ? 'Guardando…' : 'Guardar'}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default Pacientes
