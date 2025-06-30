import React, { useState, useEffect} from 'react'
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
  created_at: string   // ISO date string
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

  // Cerrar sesión
  const handleLogout = (): void => {
    localStorage.removeItem('token')
    navigate('/')
  }

  // Al montar: validar token y obtener doctor
  useEffect(() => {
    const token = localStorage.getItem('token')
    if (!token) {
      navigate('/')
      return
    }
    axios.get('http://52.1.220.84:8000/doctors/me', {
      headers: { Authorization: `Bearer ${token}` },
    })
    .then(resp => {
      setDoctorId(resp.data.id)
      setDoctorName(resp.data.nombre)
      setClinicName(resp.data.clinic_name)
    })
    .catch(err => {
      console.error('Error al obtener perfil del doctor:', err)
      localStorage.removeItem('token')
      navigate('/')
    })
  }, [navigate])

  // Obtener pacientes cuando doctorId esté listo
  useEffect(() => {
    if (doctorId === null) return
    const token = localStorage.getItem('token')
    if (!token) return

    axios.get<Paciente[]>(`http://52.1.220.84:8000/pacientes/doctor/${doctorId}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    .then(res => setPacientes(res.data))
    .catch(err => {
      console.error('Error al obtener pacientes:', err)
      setErrorPacientes('No se pudo cargar la lista de pacientes.')
    })
  }, [doctorId])

  // Handle form submission
  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setIsSubmitting(true)
    setSubmitError(null)

    if (doctorId === null) {
      setSubmitError('Perfil no cargado. Recarga e intenta de nuevo.')
      setIsSubmitting(false)
      return
    }

    try {
      const token = localStorage.getItem('token')
      const data = new FormData()
      data.append('dni', formData.dni)
      data.append('nombres', formData.nombres)
      data.append('apellidos', formData.apellidos)
      data.append('edad', formData.edad)
      data.append('celular', formData.celular)
      data.append('correo', formData.correo)
      if (formData.foto) data.append('foto', formData.foto)
      data.append('doctor_id', doctorId.toString())

      const resp = await axios.post<Paciente>(
        'http://52.1.220.84:8000/register/paciente',
        data,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${token}`,
          },
        }
      )
      setPacientes(prev => [...prev, resp.data])
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
      console.error('Error creando paciente:', err)
      setSubmitError('Error al crear paciente. Verifica los datos.')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Drag & drop handlers
  const handleDragOver = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setHighlightDrag(true)
  }
  const handleDragLeave = () => setHighlightDrag(false)
  const handleDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault()
    setHighlightDrag(false)
    if (e.dataTransfer.files[0]) {
      setFormData(prev => ({ ...prev, foto: e.dataTransfer.files[0] }))
    }
  }

  // Input change handler
  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value, files } = e.target
    if (name === 'foto' && files) {
      setFormData(prev => ({ ...prev, foto: files[0] }))
      return
    }
    let v = value
    if (name === 'dni' || name === 'celular' || name === 'edad') {
      v = value.replace(/\D/g, '')
    }
    if (name === 'nombres' || name === 'apellidos') {
      v = value.replace(/[^A-Za-zÁÉÍÓÚáéíóúÑñ ]/g, '')
    }
    setFormData(prev => ({ ...prev, [name]: v }))
  }

  return (
    <div className="home-container">
      <Sidebar onLogout={handleLogout} />
      <div className="home-content">
        <Topbar clinicName={clinicName} doctorName={doctorName} />
        <main className="home-main">
          <div className="pacientes-header">
            <h2>Pacientes de {doctorName}</h2>
            <button className="btn-add-paciente" onClick={() => setShowModal(true)}>
              <FiPlus size={18} /> Agregar
            </button>
          </div>

          {errorPacientes && (
            <p className="error-message">{errorPacientes}</p>
          )}

          {pacientes.length === 0 ? (
            <p className="no-pacientes-text">
              No tienes pacientes registrados aún.
            </p>
          ) : (
            <div className="pacientes-grid">
              {pacientes.map(p => (
                <div key={p.id} className="paciente-card">
                  <div
                    className="paciente-content"
                    onClick={() => navigate(`/perfil/${p.dni}`)}
                  >
                    <div className="paciente-photo">
                      {p.foto ? (
                        <img
                          src={`http://52.1.220.84:8000/static/pacientes/${p.foto}`}
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
                      <p className="paciente-fecha">
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>

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
              <button
                type="submit"
                className="btn-submit-paciente"
                disabled={isSubmitting}
              >
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