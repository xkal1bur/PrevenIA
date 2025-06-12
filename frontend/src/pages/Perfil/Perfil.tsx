/* ===========================================================================
 *  Perfil.tsx — versión “solo sequence2” (sin mutaciones)
 *    • Cuando el usuario hace clic en un FASTA:
 *        - Extraemos la secuencia → patientSeq
 *        - Enviamos SOLO sequence2 al <DNAViewer>
 *    • sequence1 no se pasa (DNAViewer mantiene su valor por defecto).
 *    • Incluye: subir, descargar, eliminar archivos, agenda de notas, modal ML.
 * ========================================================================== */
import React, { useEffect, useState, type ChangeEvent } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import { FiUploadCloud, FiActivity } from 'react-icons/fi'
import Sidebar from '../../components/Sidebar'
import Topbar from '../../components/Topbar'
import PredictionsPanel from '../../components/PredictionsPanel'
import DNAViewer from '../../components/DNAViewer'
import './Perfil.css'

/* ---------- Tipos de dato ---------- */
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
interface S3File { filename: string; key: string; lastModified: string }
interface Note   { id: number; title: string; content: string; timestamp: string }

/* ---------- Componente ---------- */
const Perfil: React.FC = () => {
  const { dni }  = useParams<{ dni: string }>()
  const navigate = useNavigate()

  /* ---------- State global ---------- */
  const [doctorName, setDoctorName]   = useState('')
  const [clinicName, setClinicName]   = useState('')
  const [paciente,   setPaciente]     = useState<Paciente | null>(null)
  const [loadingPaciente, setLoadingPaciente] = useState(true)
  const [error, setError]             = useState('')

  /* FASTA y visualizador */
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadMessage, setUploadMessage] = useState('')
  const [patientSeq, setPatientSeq]   = useState('')   // única secuencia que enviamos
  const [sequenceLoaded, setSequenceLoaded] = useState(false)

  /* Modal ML */
  const [showPredictions, setShowPredictions] = useState(false)

  /* S3 */
  const [files, setFiles] = useState<S3File[]>([])

  /* Notas */
  const [notes, setNotes] = useState<Note[]>([])
  const [newTitle, setNewTitle] = useState('')
  const [newContent, setNewContent] = useState('')

  /* ---------- Helpers ---------- */
  const tokenHeader = () => {
    const t = localStorage.getItem('token')
    return t ? { Authorization: `Bearer ${t}` } : {}
  }

  const formatDate = (iso: string) => {
    const d = new Date(iso)
    return (
      d.toLocaleDateString('es-PE', { day: '2-digit', month: '2-digit', year: 'numeric' }) +
      ' ' +
      d.toLocaleTimeString('es-PE', { hour: '2-digit', minute: '2-digit', hour12: true })
    )
  }

  /** Extrae y limpia la cadena de un FASTA (descarta cabeceras y caracteres inválidos) */
  const extractSequenceFromFasta = (text: string) =>
    text
      .split('\n')
      .filter(l => !l.startsWith('>') && l.trim())
      .map(l => l.trim().toUpperCase().replace(/[^ATCG]/g, ''))
      .join('')

  /* ---------- Efectos ---------- */

  /* 1) Cargar perfil del doctor y datos del paciente */
  useEffect(() => {
    const hdr = tokenHeader()
    if (!hdr.Authorization) {
      navigate('/')
      return
    }

    axios
      .get('http://localhost:8000/doctors/me', { headers: hdr })
      .then(res => {
        setDoctorName(res.data.nombre)
        setClinicName(res.data.clinic_name)
      })
      .catch(() => {
        localStorage.removeItem('token')
        navigate('/')
      })

    if (!dni) {
      setError('DNI inválido.')
      setLoadingPaciente(false)
      return
    }

    axios
      .get<Paciente>(`http://localhost:8000/pacientes/dni/${dni}`, { headers: hdr })
      .then(res => {
        setPaciente(res.data)
        setLoadingPaciente(false)
      })
      .catch(() => {
        setError('No se pudo cargar el paciente.')
        setLoadingPaciente(false)
      })
  }, [dni, navigate])

  /* 2) Listar archivos S3 */
  useEffect(() => {
    if (!dni) return
    const hdr = tokenHeader()
    axios
      .get<{ files: S3File[] }>(`http://localhost:8000/pacientes/${dni}/files`, { headers: hdr })
      .then(res =>
        setFiles(
          res.data.files.map(f => ({
            ...f,
            lastModified: formatDate(f.lastModified)
          }))
        )
      )
      .catch(console.error)
  }, [dni])

  /* 3) Listar notas */
  useEffect(() => {
    if (!dni) return
    const hdr = tokenHeader()
    axios
      .get<Note[]>(`http://localhost:8000/pacientes/${dni}/notes`, { headers: hdr })
      .then(res =>
        setNotes(
          res.data.map(n => ({
            ...n,
            timestamp: formatDate(n.timestamp)
          }))
        )
      )
      .catch(console.error)
  }, [dni])

  /* ---------- Handlers ---------- */

  /* Selección de FASTA */
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setUploadMessage('')
    setSelectedFile(e.target.files?.[0] ?? null)
  }

  /* Subir FASTA */
  const handleUpload = () => {
    if (!selectedFile) {
      setUploadMessage('Selecciona un archivo')
      return
    }
    const ext = selectedFile.name.split('.').pop()?.toLowerCase() || ''
    if (!['fasta', 'fa', 'fna'].includes(ext)) {
      setUploadMessage('Formato inválido')
      return
    }

    const hdr = tokenHeader()
    const fd = new FormData()
    fd.append('fasta_file', selectedFile)

    axios
      .post(`http://localhost:8000/pacientes/${dni}/upload_fasta`, fd, { headers: hdr })
      .then(() =>
        axios.get<{ files: S3File[] }>(`http://localhost:8000/pacientes/${dni}/files`, {
          headers: hdr
        })
      )
      .then(res => {
        setFiles(
          res.data.files.map(f => ({
            ...f,
            lastModified: formatDate(f.lastModified)
          }))
        )
        setUploadMessage('✅ Archivo subido')
      })
      .catch(() => setUploadMessage('Error al subir'))
  }

  /* Descargar FASTA */
  const handleDownload = (fn: string) => {
    const hdr = tokenHeader()
    axios
      .get(`http://localhost:8000/pacientes/${dni}/files/${encodeURIComponent(fn)}`, {
        headers: hdr,
        responseType: 'blob'
      })
      .then(res => {
        const url = URL.createObjectURL(res.data)
        const a = document.createElement('a')
        a.href = url
        a.download = fn
        a.click()
        URL.revokeObjectURL(url)
      })
      .catch(console.error)
  }

  /* Eliminar FASTA */
  const handleDelete = (fn: string) => {
    if (!window.confirm(`¿Eliminar ${fn}?`)) return
    const hdr = tokenHeader()
    axios
      .delete(`http://localhost:8000/pacientes/${dni}/files/${encodeURIComponent(fn)}`, {
        headers: hdr
      })
      .then(() => setFiles(files.filter(f => f.filename !== fn)))
      .catch(() => alert('No se pudo eliminar'))
  }

  /* Cargar FASTA para visualización */
  const loadFromFile = async (fn: string) => {
    try {
      const hdr = tokenHeader()
      const res = await axios.get(
        `http://localhost:8000/pacientes/${dni}/files/${encodeURIComponent(fn)}`,
        { headers: hdr, responseType: 'blob' }
      )
      const text = await new Response(res.data).text()
      const original = extractSequenceFromFasta(text)
      setPatientSeq(original)
      setSequenceLoaded(true)
    } catch (e) {
      console.error(e)
      alert('No se pudo cargar para visualización')
    }
  }

  /* Añadir nota */
  const addNote = () => {
    if (!newTitle.trim() || !newContent.trim()) return
    const hdr = tokenHeader()
    axios
      .post<Note>(
        `http://localhost:8000/pacientes/${dni}/notes`,
        { title: newTitle, content: newContent },
        { headers: hdr }
      )
      .then(res =>
        setNotes([
          { ...res.data, timestamp: formatDate(res.data.timestamp) },
          ...notes
        ])
      )
      .then(() => {
        setNewTitle('')
        setNewContent('')
      })
      .catch(console.error)
  }

  /* ---------- Early returns ---------- */
  if (loadingPaciente)
    return (
      <div className="home-loading">
        <p>Cargando…</p>
      </div>
    )

  if (error || !paciente)
    return (
      <div className="home-loading">
        <p style={{ color: 'red' }}>{error}</p>
      </div>
    )

  /* ---------- Render ---------- */
  return (
    <div className="perfil-container">
      {/* --- Lateral --- */}
      <Sidebar onLogout={() => { localStorage.removeItem('token'); navigate('/') }} />

      {/* --- Contenido principal --- */}
      <div className="perfil-content">
        <Topbar clinicName={clinicName} doctorName={doctorName} />

        <main className="perfil-main">
          <h2 className="perfil-title">Perfil del Paciente</h2>

          {/* ----- Grid Datos + Archivos ----- */}
          <div className="perfil-grid">
            {/* IZQUIERDA: Foto + datos */}
            <div className="perfil-left">
              <div className="perfil-card foto-card">
                <img
                  src={
                    paciente.foto
                      ? `http://localhost:8000/static/pacientes/${paciente.foto}`
                      : '/images/placeholder-person.png'
                  }
                  alt="foto paciente"
                  className="perfil-photo"
                />
              </div>

              <div className="perfil-card datos-card">
                <h3 className="perfil-nombre">
                  {paciente.nombres} {paciente.apellidos}
                </h3>
                <p className="perfil-dato">DNI: {paciente.dni}</p>
                <p className="perfil-dato">Edad: {paciente.edad} años</p>
                <p className="perfil-dato">Celular: {paciente.celular}</p>
                <p className="perfil-dato">Correo: {paciente.correo}</p>

                <button
                  className="btn-predictions"
                  onClick={() => setShowPredictions(true)}
                >
                  <FiActivity size={16} /> Predicciones ML
                </button>
              </div>
            </div>

            {/* DERECHA: Upload + lista de archivos */}
            <div className="perfil-right">
              {/* Subir FASTA */}
              <div className="perfil-card upload-card">
                <label htmlFor="fasta-upload" className="upload-box">
                  <FiUploadCloud size={48} />
                  <p>Click o arrastra tu FASTA</p>
                  <input
                    id="fasta-upload"
                    type="file"
                    accept=".fasta,.fa,.fna"
                    onChange={handleFileChange}
                    className="file-input"
                  />
                </label>
                {uploadMessage && (
                  <p className="upload-message">{uploadMessage}</p>
                )}
                <button onClick={handleUpload} className="upload-button">
                  Procesar Secuencia
                </button>
              </div>

              {/* Lista Archivos */}
              <div className="perfil-card files-card">
                <div className="files-box">
                  {files.length === 0 ? (
                    <p>No hay archivos cargados.</p>
                  ) : (
                    <ul className="files-list">
                      {files.map(f => (
                        <li key={f.key} className="file-item">
                          <span
                            className="file-name"
                            onClick={() => loadFromFile(f.filename)}
                          >
                            {f.filename}
                          </span>
                          <span className="file-date">{f.lastModified}</span>

                          <button
                            className="file-download-btn"
                            onClick={() => handleDownload(f.filename)}
                          >
                            Descargar
                          </button>

                          <button
                            className="file-delete-btn"
                            onClick={() => handleDelete(f.filename)}
                          >
                            Eliminar
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* ----- Agenda de notas ----- */}
          <div className="perfil-card notes-card">
            <h4 className="notes-header">Agenda de Apuntes</h4>

            <div className="notes-container">
              {/* Formulario nueva nota */}
              <div className="notes-form">
                <input
                  type="text"
                  placeholder="Título de la nota"
                  value={newTitle}
                  onChange={e => setNewTitle(e.target.value)}
                  className="notes-input"
                />
                <textarea
                  placeholder="Escribe tu apunte aquí..."
                  value={newContent}
                  onChange={e => setNewContent(e.target.value)}
                  className="notes-textarea"
                />
                <button className="notes-add-btn" onClick={addNote}>
                  Agregar Nota
                </button>
              </div>

              {/* Lista notas */}
              <ul className="notes-list">
                {notes.map((n, i) => (
                  <li key={i} className="note-item">
                    <div className="note-timestamp">{n.timestamp}</div>
                    <div className="note-title">{n.title}</div>
                    <div className="note-content">{n.content}</div>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* ----- DNAViewer ----- */}
          {sequenceLoaded ? (
            <DNAViewer
              sequence2={patientSeq} /* enviamos solo la secuencia del paciente */
              title2={`Paciente ${paciente.nombres}`}
            />
          ) : (
            <div className="perfil-card dna-placeholder">
              <div className="dna-placeholder-text">
                <p>Sube un FASTA para visualizar la secuencia</p>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* ----- Modal predicciones ML ----- */}
      {showPredictions && (
        <div className="modal-overlay">
          <div className="modal-content predictions-modal">
            <button
              className="modal-close-btn"
              onClick={() => setShowPredictions(false)}
            >
              ✕
            </button>
            <PredictionsPanel
              patientDni={paciente.dni}
              patientName={`${paciente.nombres} ${paciente.apellidos}`}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default Perfil
