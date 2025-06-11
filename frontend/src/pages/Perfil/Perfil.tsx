import React, { useEffect, useState} from 'react'
import type { ChangeEvent } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import Sidebar from '../../components/Sidebar'
import Topbar from '../../components/Topbar'
import PredictionsPanel from '../../components/PredictionsPanel'
import DNAViewer from '../../components/DNAViewer'
import { FiUploadCloud, FiActivity } from 'react-icons/fi'
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

  // --- NUEVO ESTADO para el archivo FASTA y para el feedback de carga ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadMessage, setUploadMessage] = useState<string>('')

  // Estado para las secuencias de ADN
  const [referenceSequence, setReferenceSequence] = useState<string>('')
  const [patientSequence, setPatientSequence] = useState<string>('')
  const [sequenceLoaded, setSequenceLoaded] = useState<boolean>(false)

  // Estado para el panel de predicciones
  const [showPredictions, setShowPredictions] = useState<boolean>(false)

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

  // Función para extraer secuencia de un archivo FASTA
  const extractSequenceFromFasta = (fileContent: string): string => {
    const lines = fileContent.split('\n')
    let sequence = ''
    
    for (const line of lines) {
      const trimmedLine = line.trim()
      // Ignorar líneas de encabezado (que empiezan con >)
      if (!trimmedLine.startsWith('>') && trimmedLine.length > 0) {
        sequence += trimmedLine.toUpperCase().replace(/[^ATCG]/g, '') // Solo bases válidas
      }
    }
    
    return sequence
  }

  // Función optimizada para generar mutaciones y deleciones simuladas en secuencias largas
  const generatePatientSequence = (referenceSeq: string): string => {
    if (referenceSeq.length === 0) return ''
    
    const bases = ['A', 'T', 'C', 'G']
    const mutationRate = 0.02 // 2% de mutaciones
    const deletionRate = 0.005 // 0.5% de deleciones
    const sequenceArray = referenceSeq.split('')
    
    // Optimización: pre-calcular posiciones de mutación y deleción para secuencias largas
    const totalMutations = Math.floor(referenceSeq.length * mutationRate)
    const totalDeletions = Math.floor(referenceSeq.length * deletionRate)
    
    // Generar posiciones aleatorias únicas para mutaciones
    const mutationPositions = new Set<number>()
    while (mutationPositions.size < totalMutations) {
      mutationPositions.add(Math.floor(Math.random() * referenceSeq.length))
    }
    
    // Generar posiciones aleatorias únicas para deleciones (que no coincidan con mutaciones)
    const deletionPositions = new Set<number>()
    while (deletionPositions.size < totalDeletions) {
      const position = Math.floor(Math.random() * referenceSeq.length)
      if (!mutationPositions.has(position)) {
        deletionPositions.add(position)
      }
    }
    
    // Aplicar deleciones primero
    deletionPositions.forEach(position => {
      sequenceArray[position] = '-'
    })
    
    // Aplicar mutaciones en las posiciones seleccionadas
    mutationPositions.forEach(position => {
      const currentBase = sequenceArray[position]
      if (currentBase !== '-') { // Solo mutar si no es una deleción
        const availableBases = bases.filter(base => base !== currentBase)
        sequenceArray[position] = availableBases[Math.floor(Math.random() * availableBases.length)]
      }
    })
    
    return sequenceArray.join('')
  }

  // --- Handler para cuando el usuario seleccione un archivo ---
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setUploadMessage('') // limpiar mensajes previos
    setSequenceLoaded(false) // resetear estado de secuencias
    setReferenceSequence('')
    setPatientSequence('')
    
    const file = e.target.files?.[0] || null
    if (file) {
      // Validar extensión localmente (opcional)
      const ext = file.name.split('.').pop()?.toLowerCase()
      if (!['fasta', 'fa', 'fna'].includes(ext || '')) {
        setUploadMessage('Formato inválido. Debe ser .fasta, .fa o .fna')
        setSelectedFile(null)
        return
      }
      setSelectedFile(file)
    } else {
      setSelectedFile(null)
    }
  }

  // --- Handler para hacer la subida a tu backend y luego a S3 ---
  const handleUpload = async () => {
    setUploadMessage('')
    if (!selectedFile) {
      setUploadMessage('Selecciona un archivo primero')
      return
    }
    try {
      const token = localStorage.getItem('token')
      if (!token) throw new Error('Token no encontrado')

      // Leer el contenido del archivo FASTA localmente para extraer la secuencia
      const fileContent = await selectedFile.text()
      const extractedSequence = extractSequenceFromFasta(fileContent)
      
      if (extractedSequence.length === 0) {
        setUploadMessage('Error: No se encontró una secuencia válida en el archivo FASTA')
        return
      }

      // Generar secuencia del paciente con mutaciones simuladas
      const patientSeq = generatePatientSequence(extractedSequence)

      // Preparamos FormData
      const formData = new FormData()
      formData.append('fasta_file', selectedFile)

      // Llamamos a nuestro endpoint FastAPI
      await axios.post(
        `http://localhost:8000/pacientes/${dni}/upload_fasta`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
            Authorization: `Bearer ${token}`
          }
        }
      )

      // Guardar las secuencias en el estado
      setReferenceSequence(extractedSequence)
      setPatientSequence(patientSeq)
      setSequenceLoaded(true)

      const isLarge = extractedSequence.length > 2000
      setUploadMessage(
        `Archivo subido correctamente. Secuencia de ${extractedSequence.length.toLocaleString()} bases procesada.${
          isLarge ? ' ⚡ Usando visualización optimizada para mejor rendimiento.' : ''
        }`
      )
    } catch (err: unknown) {
      console.error(err)
      const errorMessage = axios.isAxiosError(err) && err.response?.data?.detail 
        ? err.response.data.detail 
        : 'Error al subir el archivo'
      setUploadMessage(errorMessage)
    }
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
                
                <button
                  className="btn-predictions"
                  onClick={() => setShowPredictions(true)}
                  style={{ marginTop: '1rem', padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                >
                  <FiActivity size={16} /> Predicciones ML
                </button>
              </div>
            </div>

            <div className="perfil-right">
              <div className="perfil-card upload-card">
                <div className="upload-box">
                  <FiUploadCloud size={48} />
                </div>
                <input
                  type="file"
                  accept=".fasta,.fa,.fna"
                  onChange={handleFileChange}
                  className="file-input"
                />
                <button onClick={handleUpload} className="upload-button">
                  Cargar Archivo FASTA
                </button>
                {uploadMessage && (
                  <p style={{ marginTop: '8px', color: 'green' }}>
                    {uploadMessage}
                  </p>
                )}
              </div>

              <div className="perfil-card files-card">
                <div className="files-box">
                  <p>No hay archivos cargados.</p>
                </div>
              </div>
          </div>
            </div>
              {sequenceLoaded ? (
                <DNAViewer
                  sequence1={referenceSequence}
                  sequence2={patientSequence}
                  title1="Secuencia de Referencia (Archivo FASTA)"
                  title2={`Secuencia del Paciente ${paciente.nombres} ${paciente.apellidos}`}
                />
              ) : (
                <div className="perfil-card dna-placeholder">
                  <div style={{ padding: '2rem', textAlign: 'center', color: '#6c757d' }}>
                    <p>📧 Sube un archivo FASTA para visualizar las secuencias de ADN</p>
                    <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                      El visualizador mostrará la secuencia de referencia y generará una secuencia del paciente con mutaciones simuladas.
                    </p>
                  </div>
                </div>
              )}
        </main>
      </div>

      {/* Modal para mostrar predicciones */}
      {showPredictions && (
        <div className="modal-overlay">
          <div className="modal-content predictions-modal">
            <button
              className="modal-close-btn"
              onClick={() => setShowPredictions(false)}
              style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'none', border: 'none', cursor: 'pointer' }}
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