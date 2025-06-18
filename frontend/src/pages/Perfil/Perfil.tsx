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
interface S3File { filename: string; key: string; lastModified: string; isFolder?: boolean; chunkCount?: number; type?: string }
interface Note   { id: number; title: string; content: string; timestamp: string }
interface PatientChunksInfo { 
  chunks_available: boolean; 
  chunk_count: number; 
  first_match_position: number; 
  chunk_length: number; 
  total_length: number; 
  chunks_folder: string; 
  chunk_type?: string;
  blast_results_key?: string;
  reference_chunk_number?: number;
  reference_filename?: string;
  // Nueva información de navegación de BLAST
  navigation_info?: {
    match_start_position: number;
    match_end_position: number;
    recommended_chunk: number;                    // Chunk del PACIENTE donde está el match
    recommended_reference_chunk?: number;         // Chunk de REFERENCIA correspondiente
    position_in_chunk: number;
  };
}

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
  const [patientFiles, setPatientFiles] = useState<string[]>([])   // archivos del paciente
  const [sequenceLoaded, setSequenceLoaded] = useState(false)
  const [loadingPatientFiles, setLoadingPatientFiles] = useState(false)
  const [patientFilesStatus, setPatientFilesStatus] = useState('')
  const [patientChunksInfo, setPatientChunksInfo] = useState<PatientChunksInfo | null>(null)
  
  /* Alineamiento */
  const [showAlignmentModal, setShowAlignmentModal] = useState(false)
  const [availableFiles, setAvailableFiles] = useState<S3File[]>([])
  const [selectedFileForAlignment, setSelectedFileForAlignment] = useState<string>('')
  const [alignmentLoading, setAlignmentLoading] = useState(false)
  const [alignmentMessage, setAlignmentMessage] = useState('')

  /* Procesamiento de Embeddings */
  const [showEmbeddingModal, setShowEmbeddingModal] = useState(false)
  const [selectedFileForEmbedding, setSelectedFileForEmbedding] = useState<string>('')
  const [embeddingLoading, setEmbeddingLoading] = useState(false)
  const [embeddingMessage, setEmbeddingMessage] = useState('')

  // Auto-clear messages after delay
  useEffect(() => {
    if (uploadMessage && !uploadMessage.includes('⏳')) {
      const timer = setTimeout(() => {
        // Add fade-out class first
        const messageEl = document.querySelector('.upload-message')
        if (messageEl) {
          messageEl.classList.add('fade-out')
          setTimeout(() => setUploadMessage(''), 500) // Wait for animation
        } else {
          setUploadMessage('')
        }
      }, 4000) // 4 segundos + 0.5s de animación
      return () => clearTimeout(timer)
    }
  }, [uploadMessage])

  useEffect(() => {
    if (patientFilesStatus && !patientFilesStatus.includes('Buscando')) {
      const timer = setTimeout(() => {
        // Add fade-out class first
        const statusEl = document.querySelector('.status-message')
        if (statusEl) {
          statusEl.classList.add('fade-out')
          setTimeout(() => setPatientFilesStatus(''), 500) // Wait for animation
        } else {
          setPatientFilesStatus('')
        }
      }, 6000) // 6 segundos + 0.5s de animación
      return () => clearTimeout(timer)
    }
  }, [patientFilesStatus])

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
      .then(res => {
        const filesWithDate = res.data.files.map(f => ({
            ...f,
            lastModified: formatDate(f.lastModified)
          }))
        setFiles(filesWithDate)
        
        // Buscar automáticamente archivos alineados si están disponibles
        const hasAlignedChunks = res.data.files.some(f => f.isFolder && f.filename === 'aligned_chunks/' && f.type === 'aligned_sequences')
        
        if (hasAlignedChunks) {
          // Cargar archivos alineados si están disponibles
          const alignedFileNames = []
          for (let i = 1; i <= 1000; i++) {
            alignedFileNames.push(`${dni}/aligned_chunks/aligned_part_${i.toString().padStart(4, '0')}.fasta`)
          }
          setPatientFiles(alignedFileNames)
          setSequenceLoaded(true)
          setPatientFilesStatus(`✅ 1000 archivos alineados del paciente encontrados automáticamente`)
        }
      })
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

    setUploadMessage('⏳ Subiendo archivo...')

    axios
      .post(`http://localhost:8000/pacientes/${dni}/upload_fasta`, fd, { headers: hdr })
      .then(() =>
        axios.get<{ files: S3File[] }>(`http://localhost:8000/pacientes/${dni}/files`, {
          headers: hdr
        })
      )
      .then(res => {
        const filesWithDate = res.data.files.map(f => ({
            ...f,
            lastModified: formatDate(f.lastModified)
          }))
        setFiles(filesWithDate)
        setUploadMessage('✅ Archivo subido correctamente')
        
        // No buscar archivos automáticamente después de subir - solo recargar la lista
        
        // Limpiar el input de archivo
        setSelectedFile(null)
        const fileInput = document.getElementById('fasta-upload') as HTMLInputElement
        if (fileInput) fileInput.value = ''
      })
      .catch(() => setUploadMessage('❌ Error al subir archivo'))
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
      .then(() => {
        const updatedFiles = files.filter(f => f.filename !== fn)
        setFiles(updatedFiles)
        
        // Verificar si quedan archivos alineados después de eliminar
        const hasAlignedChunks = updatedFiles.some(f => f.isFolder && f.filename === 'aligned_chunks/' && f.type === 'aligned_sequences')
        
        if (hasAlignedChunks) {
          // Mantener archivos alineados si están disponibles
          const alignedFileNames = []
          for (let i = 1; i <= 100; i++) {
            alignedFileNames.push(`${dni}/aligned_chunks/aligned_part_${i.toString().padStart(3, '0')}.fasta`)
          }
          setPatientFiles(alignedFileNames)
          setSequenceLoaded(true)
          setPatientFilesStatus(`✅ 100 archivos alineados del paciente disponibles`)
        } else {
          // Si no hay archivos alineados, limpiar la visualización
          setPatientFiles([])
          setSequenceLoaded(false)
          setPatientFilesStatus('⚠️ No hay secuencias alineadas disponibles para visualizar')
        }
      })
      .catch(() => alert('No se pudo eliminar'))
  }

  /* Eliminar Carpeta */
  const handleDeleteFolder = (folderName: string) => {
    const folderDisplayName = folderName.replace('/', '')
    if (!window.confirm(`¿Eliminar la carpeta "${folderDisplayName}" y todo su contenido?\n\nEsta acción no se puede deshacer.`)) return
    
    const hdr = tokenHeader()
    const folderNameForAPI = folderName.endsWith('/') ? folderName.slice(0, -1) : folderName
    
    axios
      .delete(`http://localhost:8000/pacientes/${dni}/folders/${encodeURIComponent(folderNameForAPI)}`, {
        headers: hdr
      })
      .then(() => {
        const updatedFiles = files.filter(f => f.filename !== folderName)
        setFiles(updatedFiles)
        
        // Si se eliminó la carpeta de archivos alineados, limpiar la visualización
        if (folderName === 'aligned_chunks/' || folderName === 'patient_chunks/') {
          const hasAlignedChunks = updatedFiles.some(f => f.isFolder && f.filename === 'aligned_chunks/' && f.type === 'aligned_sequences')
          const hasPatientChunks = updatedFiles.some(f => f.isFolder && f.filename === 'patient_chunks/')
          
          if (hasAlignedChunks) {
            // Mantener archivos alineados si están disponibles
            const alignedFileNames = []
            for (let i = 1; i <= 1000; i++) {
              alignedFileNames.push(`${dni}/aligned_chunks/aligned_part_${i.toString().padStart(4, '0')}.fasta`)
            }
            setPatientFiles(alignedFileNames)
            setSequenceLoaded(true)
            setPatientFilesStatus(`✅ Archivos alineados del paciente disponibles`)
          } else if (hasPatientChunks) {
            // Mantener archivos de paciente si están disponibles
            const patientFileNames = []
            for (let i = 1; i <= 1000; i++) {
              patientFileNames.push(`${dni}/patient_chunks/patient_part_${i.toString().padStart(4, '0')}.fasta`)
            }
            setPatientFiles(patientFileNames)
            setSequenceLoaded(true)
            setPatientFilesStatus(`✅ Archivos del paciente disponibles`)
          } else {
            // Si no hay archivos, limpiar la visualización
            setPatientFiles([])
            setSequenceLoaded(false)
            setPatientFilesStatus('⚠️ No hay secuencias disponibles para visualizar')
            setPatientChunksInfo(null)
          }
        }
        
        alert(`Carpeta "${folderDisplayName}" eliminada correctamente`)
      })
      .catch((error) => {
        console.error('Error eliminando carpeta:', error)
        alert('No se pudo eliminar la carpeta')
      })
  }

  /* Cargar archivos del paciente para visualización */
  const loadPatientFiles = async () => {
    try {
      setLoadingPatientFiles(true)
      setPatientFilesStatus('Buscando archivos del paciente...')
      
      const hdr = tokenHeader()
      
      // Primero intentar obtener información de chunks alineados (prioridad)
      try {
        const alignedChunksInfoRes = await axios.get<PatientChunksInfo>(`http://localhost:8000/pacientes/${dni}/aligned_chunks/info`, { headers: hdr })
        setPatientChunksInfo(alignedChunksInfoRes.data)
        
        // Generar lista de archivos alineados
        const alignedFileNames = []
        for (let i = 1; i <= alignedChunksInfoRes.data.chunk_count; i++) {
          alignedFileNames.push(`${dni}/aligned_chunks/aligned_part_${i.toString().padStart(4, '0')}.fasta`)
        }
        
        setPatientFiles(alignedFileNames)
        setSequenceLoaded(true)
        setPatientFilesStatus(`✅ ${alignedChunksInfoRes.data.chunk_count} partes alineadas del paciente cargadas correctamente`)
        
        // Auto-scroll hacia la visualización después de un pequeño delay
        setTimeout(() => {
          const dnaViewerElement = document.querySelector('.perfil-card:last-child')
          if (dnaViewerElement) {
            dnaViewerElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }
        }, 500)
        
        return // Salir exitosamente si se encontraron chunks alineados
        
      } catch {
        // Si no hay chunks alineados, intentar chunks normales del paciente
        try {
          const chunksInfoRes = await axios.get<PatientChunksInfo>(`http://localhost:8000/pacientes/${dni}/patient_chunks/info`, { headers: hdr })
          setPatientChunksInfo(chunksInfoRes.data)
          
          // Generar lista de archivos basada en la información de chunks
          const patientFileNames = []
          for (let i = 1; i <= Math.min(chunksInfoRes.data.chunk_count, 1000); i++) {
            patientFileNames.push(`${dni}/patient_chunks/patient_part_${i.toString().padStart(4, '0')}.fasta`)
          }
          
          setPatientFiles(patientFileNames)
          setSequenceLoaded(true)
          setPatientFilesStatus(`✅ ${chunksInfoRes.data.chunk_count} partes del paciente cargadas correctamente`)
          
          // Auto-scroll hacia la visualización después de un pequeño delay
          setTimeout(() => {
            const dnaViewerElement = document.querySelector('.perfil-card:last-child')
            if (dnaViewerElement) {
              dnaViewerElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
            }
          }, 500)
          
        } catch {
          // Si no hay chunks, intentar el método anterior con archivos S3
          const res = await axios.get<{ files: S3File[] }>(`http://localhost:8000/pacientes/${dni}/files`, { headers: hdr })
          
                  // Verificar si hay archivos alineados disponibles
        const hasAlignedChunks = res.data.files.some(f => f.isFolder && f.filename === 'aligned_chunks/' && f.type === 'aligned_sequences')
        
        if (hasAlignedChunks) {
          // Cargar archivos alineados si están disponibles
          const alignedFileNames = []
          for (let i = 1; i <= 1000; i++) {
            alignedFileNames.push(`${dni}/aligned_chunks/aligned_part_${i.toString().padStart(4, '0')}.fasta`)
          }
            
            setPatientFiles(alignedFileNames)
            setSequenceLoaded(true)
            setPatientFilesStatus('✅ 1000 partes alineadas del paciente cargadas correctamente')
            
            // Auto-scroll hacia la visualización
            setTimeout(() => {
              const dnaViewerElement = document.querySelector('.perfil-card:last-child')
              if (dnaViewerElement) {
                dnaViewerElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
              }
            }, 500)
          } else {
            setPatientFilesStatus('⚠️ No hay secuencias alineadas disponibles. Usa el botón "Alineamiento" para procesar un archivo FASTA.')
          }
        }
      }
    } catch (e) {
      console.error(e)
      setPatientFilesStatus('❌ Error al cargar archivos del paciente')
    } finally {
      setLoadingPatientFiles(false)
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

  /* Abrir modal de alineamiento */
  const openAlignmentModal = () => {
    // Filtrar archivos FASTA que no sean carpetas (incluir todos los archivos .fasta subidos)
    const filesForAlignment = files.filter(f => 
      !f.isFolder && 
      (f.filename.toLowerCase().endsWith('.fasta') || 
       f.filename.toLowerCase().endsWith('.fa') || 
       f.filename.toLowerCase().endsWith('.fna')) &&
      !f.filename.includes('patient_part_') &&
      !f.filename.includes('aligned_part_')
    )
    setAvailableFiles(filesForAlignment)
    setShowAlignmentModal(true)
    setAlignmentMessage('')
  }

  /* Ejecutar alineamiento */
  const executeAlignment = async () => {
    if (!selectedFileForAlignment) {
      setAlignmentMessage('⚠️ Selecciona un archivo para alinear')
      return
    }

    setAlignmentLoading(true)
    setAlignmentMessage('⏳ Alineando secuencia con cr13...')

    try {
      const hdr = tokenHeader()
      const formData = new FormData()
      formData.append('filename', selectedFileForAlignment)

      await axios.post(
        `http://localhost:8000/pacientes/${dni}/align_with_cr13`,
        formData,
        { headers: hdr }
      )

      setAlignmentMessage('✅ Alineamiento completado exitosamente')
      
      // Recargar la lista de archivos
      const filesResponse = await axios.get<{ files: S3File[] }>(
        `http://localhost:8000/pacientes/${dni}/files`,
        { headers: hdr }
      )
      
      const filesWithDate = filesResponse.data.files.map(f => ({
        ...f,
        lastModified: formatDate(f.lastModified)
      }))
      setFiles(filesWithDate)

      // Verificar si hay archivos alineados y cargarlos automáticamente
      const hasAlignedChunks = filesResponse.data.files.some(f => f.isFolder && f.filename === 'aligned_chunks/')
      
      if (hasAlignedChunks) {
        try {
          // Obtener información de los chunks alineados
          const alignedChunksInfoRes = await axios.get<PatientChunksInfo>(`http://localhost:8000/pacientes/${dni}/aligned_chunks/info`, { headers: hdr })
          setPatientChunksInfo(alignedChunksInfoRes.data)
          
          // Generar lista de archivos alineados
          const alignedFileNames = []
          for (let i = 1; i <= Math.min(alignedChunksInfoRes.data.chunk_count, 1000); i++) {
            alignedFileNames.push(`${dni}/aligned_chunks/aligned_part_${i.toString().padStart(4, '0')}.fasta`)
          }
          
          setPatientFiles(alignedFileNames)
          setSequenceLoaded(true)
          
          // Mensaje mejorado con información de navegación
          let statusMessage = `✅ Secuencias alineadas cargadas correctamente`
          if (alignedChunksInfoRes.data.navigation_info) {
            statusMessage += ` - Navegando automáticamente al chunk ${alignedChunksInfoRes.data.navigation_info.recommended_chunk} (contiene el alineamiento)`
          }
          setPatientFilesStatus(statusMessage)
          
          // Auto-scroll hacia la visualización después de cerrar el modal
          setTimeout(() => {
            const dnaViewerElement = document.querySelector('.perfil-card:last-child')
            if (dnaViewerElement) {
              dnaViewerElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
            }
          }, 2500) // Dar tiempo para que se cierre el modal primero
          
        } catch (error) {
          console.error('Error obteniendo información de chunks alineados:', error)
          // Fallback: cargar archivos sin información adicional
          const alignedFileNames = []
          for (let i = 1; i <= 1000; i++) {
            alignedFileNames.push(`${dni}/aligned_chunks/aligned_part_${i.toString().padStart(4, '0')}.fasta`)
          }
          setPatientFiles(alignedFileNames)
          setSequenceLoaded(true)
          setPatientFilesStatus(`✅ Secuencias alineadas cargadas correctamente`)
        }
      }

      // Cerrar modal después de un delay
      setTimeout(() => {
        setShowAlignmentModal(false)
        setSelectedFileForAlignment('')
      }, 2000)
      
    } catch (error) {
      console.error('Error en alineamiento:', error)
      setAlignmentMessage('❌ Error al realizar el alineamiento')
    } finally {
      setAlignmentLoading(false)
    }
  }

  /* Abrir modal de procesamiento de embeddings */
  const openEmbeddingModal = () => {
    // Filtrar archivos FASTA que no sean carpetas (incluir todos los archivos .fasta subidos)
    const filesForEmbedding = files.filter(f => 
      !f.isFolder && 
      (f.filename.toLowerCase().endsWith('.fasta') || 
       f.filename.toLowerCase().endsWith('.fa') || 
       f.filename.toLowerCase().endsWith('.fna')) &&
      !f.filename.includes('patient_part_') &&
      !f.filename.includes('aligned_part_')
    )
    setAvailableFiles(filesForEmbedding)
    setShowEmbeddingModal(true)
    setEmbeddingMessage('')
  }

  /* Ejecutar procesamiento de embedding */
  const executeEmbeddingProcessing = async () => {
    if (!selectedFileForEmbedding) {
      setEmbeddingMessage('⚠️ Selecciona un archivo para procesar')
      return
    }

    setEmbeddingLoading(true)
    setEmbeddingMessage('⏳ Procesando secuencia con NVIDIA NIM... Esto puede tomar unos minutos.')

    try {
      const hdr = tokenHeader()
      const formData = new FormData()
      formData.append('filename', selectedFileForEmbedding)

      const response = await axios.post(
        `http://localhost:8000/pacientes/${dni}/process_embedding`,
        formData,
        { headers: hdr }
      )

      setEmbeddingMessage('✅ Embedding generado exitosamente')
      
      // Recargar la lista de archivos para mostrar el nuevo embedding
      const filesResponse = await axios.get<{ files: S3File[] }>(
        `http://localhost:8000/pacientes/${dni}/files`,
        { headers: hdr }
      )
      
      const filesWithDate = filesResponse.data.files.map(f => ({
        ...f,
        lastModified: formatDate(f.lastModified)
      }))
      setFiles(filesWithDate)

      // Mostrar información adicional del embedding procesado
      if (response.data) {
        const { embedding_shape, sequence_length, api_used, embedding_key } = response.data
        setEmbeddingMessage(
          `✅ Embedding generado exitosamente\n` +
          `📊 Forma: ${embedding_shape.join('x')}\n` +
          `📏 Longitud de secuencia: ${sequence_length.toLocaleString()} bases\n` +
          `🤖 API: ${api_used}\n` +
          `💾 Guardado en S3: ${embedding_key}`
        )
      }

      // No cerrar el modal automáticamente - permitir al usuario cerrarlo manualmente
      
    } catch (error) {
      console.error('Error en procesamiento de embedding:', error)
      let errorMessage = 'Error al procesar la secuencia'
      if (error && typeof error === 'object' && 'response' in error) {
        const axiosError = error as { response?: { data?: { detail?: string } } }
        errorMessage = axiosError.response?.data?.detail || errorMessage
      }
      setEmbeddingMessage(`❌ ${errorMessage}`)
    } finally {
      setEmbeddingLoading(false)
    }
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

              {/* Lista Archivos */}
              <div className="perfil-card files-card">
                <h4 style={{ marginBottom: '1rem', color: '#2c3e50' }}>📁 Archivos FASTA Subidos</h4>
                <div className="files-box">
                  {files.length === 0 ? (
                    <p style={{ color: '#6c757d', fontStyle: 'italic' }}>No hay archivos cargados.</p>
                  ) : (
                    <ul className="files-list">
                      {files.map(f => (
                        <li key={f.key} className="file-item">
                          <span className="file-name">
                            {f.isFolder ? (
                              <>
                                📁 {f.filename} 
                                {f.type === 'aligned_sequences' && <span style={{ color: '#28a745', fontSize: '0.75rem', marginLeft: '0.5rem' }}>(Alineado)</span>}
                                {f.type === 'raw_sequences' && <span style={{ color: '#6c757d', fontSize: '0.75rem', marginLeft: '0.5rem' }}>(Original)</span>}
                                {f.chunkCount && <span style={{ color: '#007bff', fontSize: '0.75rem', marginLeft: '0.5rem' }}>({f.chunkCount} archivos)</span>}
                              </>
                            ) : (
                              <>📄 {f.filename}</>
                            )}
                          </span>
                          <span className="file-date">{f.lastModified}</span>

                          {f.isFolder ? (
                            <button
                              className="file-delete-btn"
                              onClick={() => handleDeleteFolder(f.filename)}
                              style={{ backgroundColor: '#dc3545' }}
                              title="Eliminar carpeta y todo su contenido"
                            >
                              🗑️ Eliminar Carpeta
                            </button>
                          ) : (
                            <>
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
                            </>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </div>

            {/* DERECHA: Upload + lista de archivos */}
            <div className="perfil-right">
              {/* Subir FASTA */}
              <div className="perfil-card upload-card">
                <h4 style={{ marginBottom: '1rem', color: '#2c3e50' }}>📤 Subir Archivo FASTA</h4>
                
                <label htmlFor="fasta-upload" className={`upload-box ${selectedFile ? 'file-selected' : ''}`}>
                  <div className="upload-icon">
                    {selectedFile ? (
                      <div className="file-icon">📄</div>
                    ) : (
                      <FiUploadCloud size={48} />
                    )}
                  </div>
                  <div className="upload-text">
                    {selectedFile ? (
                      <>
                        <p className="file-name">{selectedFile.name}</p>
                        <p className="file-size">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                      </>
                    ) : (
                      <>
                        <p className="main-text">Arrastra tu archivo FASTA aquí</p>
                        <p className="sub-text">o haz click para seleccionar</p>
                        <p className="formats">Formatos: .fasta, .fa, .fna</p>
                      </>
                    )}
                  </div>
                  <input
                    id="fasta-upload"
                    type="file"
                    accept=".fasta,.fa,.fna"
                    onChange={handleFileChange}
                    className="file-input"
                  />
                </label>
                
                {uploadMessage && (
                  <div className={`upload-message ${uploadMessage.includes('✅') ? 'success' : uploadMessage.includes('❌') ? 'error' : uploadMessage.includes('⏳') ? 'loading' : 'warning'}`}>
                    {uploadMessage}
                  </div>
                )}
                
                <button 
                  onClick={handleUpload} 
                  className={`upload-button ${selectedFile ? 'ready' : ''}`}
                  disabled={!selectedFile || uploadMessage.includes('⏳')}
                >
                  {uploadMessage.includes('⏳') ? (
                    <>
                      <span className="loading-spinner"></span>
                      Procesando...
                    </>
                  ) : selectedFile ? (
                    <>🚀 Procesar Secuencia</>
                  ) : (
                    <>Selecciona un archivo</>
                  )}
                </button>
              </div>

              {/* Herramientas */}
              <div className="perfil-card dna-control-card">
                <h4 style={{ marginBottom: '1rem', color: '#2c3e50' }}>🧬 Herramientas </h4>
                
                <div style={{ marginBottom: '1rem' }}>
                  <button
                    className={`load-patient-btn ${loadingPatientFiles ? 'loading' : ''}`}
                    onClick={loadPatientFiles}
                    disabled={loadingPatientFiles}
                    style={{ width: '100%', marginBottom: '0.5rem' }}
                  >
                    {loadingPatientFiles ? (
                      <>🔄 Cargando archivos...</>
                    ) : (
                      <>🧬 Cargar y Visualizar Secuencias</>
                    )}
                  </button>
                  
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button
                      className="alignment-btn"
                      onClick={openAlignmentModal}
                      disabled={files.filter(f => 
                        !f.isFolder && 
                        (f.filename.toLowerCase().endsWith('.fasta') || 
                         f.filename.toLowerCase().endsWith('.fa') || 
                         f.filename.toLowerCase().endsWith('.fna')) &&
                        !f.filename.includes('patient_part_') &&
                        !f.filename.includes('aligned_part_')
                      ).length === 0}
                      style={{ flex: 1 }}
                    >
                      🔬 Alineamiento
                    </button>
                    
                    <button
                      className="embedding-btn"
                      onClick={openEmbeddingModal}
                      disabled={files.filter(f => 
                        !f.isFolder && 
                        (f.filename.toLowerCase().endsWith('.fasta') || 
                         f.filename.toLowerCase().endsWith('.fa') || 
                         f.filename.toLowerCase().endsWith('.fna')) &&
                        !f.filename.includes('patient_part_') &&
                        !f.filename.includes('aligned_part_')
                      ).length === 0}
                      style={{ flex: 1 }}
                    >
                      🧠 Evo 2 Embeddings
                    </button>
                  </div>
                </div>

                {patientFilesStatus && (
                  <div className={`status-message ${patientFilesStatus.includes('✅') ? 'success' : patientFilesStatus.includes('❌') ? 'error' : 'warning'}`}>
                    {patientFilesStatus}
                  </div>
                )}

                {sequenceLoaded && (
                                      <div className="sequence-info">
                      <p><strong>Estado:</strong> {patientFiles.length} fragmentos de secuencia cargados {patientChunksInfo?.chunk_type === 'aligned_chunks' ? '(Alineados con BLAST)' : '(Procesados)'}</p>
                    <p><strong>Fragmentos:</strong> {patientFiles.length === 1000 ? 'Secuencia completa dividida en 1000 fragmentos iguales' : `${patientFiles.length} fragmentos disponibles`}</p>
                    <p><strong>Tamaño estimado:</strong> ~{patientChunksInfo?.total_length?.toLocaleString() || (patientFiles.length > 0 ? 'Longitud igual a secuencia de referencia' : '0')} bases</p>
                    {patientChunksInfo?.first_match_position !== undefined && (
                      <p><strong>Posición de match:</strong> Inicia en posición {patientChunksInfo.first_match_position + 1} {patientChunksInfo?.chunk_type === 'aligned_chunks' ? '(Auto-cargada)' : ''}</p>
                    )}
                    {patientChunksInfo?.navigation_info && (
                      <div>
                        <p><strong>Navegación BLAST:</strong></p>
                        <ul style={{ marginLeft: '1rem', fontSize: '0.875rem' }}>
                          <li>Chunk del paciente: <strong>{patientChunksInfo.navigation_info.recommended_chunk}/1000</strong> (contiene el alineamiento)</li>
                          {patientChunksInfo.navigation_info.recommended_reference_chunk && (
                            <li>Chunk de referencia: <strong>{patientChunksInfo.navigation_info.recommended_reference_chunk}/1000</strong></li>
                          )}
                          <li>Posición en chunk: <strong>{patientChunksInfo.navigation_info.position_in_chunk}</strong></li>
                        </ul>
                      </div>
                    )}
                    {patientChunksInfo?.chunk_type === 'aligned_chunks' && (
                      <div>
                        <p style={{ color: '#28a745', fontSize: '0.875rem', fontWeight: '500' }}>✅ Secuencias alineadas - La visualización iniciará automáticamente en la posición de match</p>
                        {patientChunksInfo?.reference_filename && patientChunksInfo?.navigation_info?.recommended_reference_chunk && (
                          <p style={{ color: '#007bff', fontSize: '0.875rem' }}>📄 Fragmento de referencia: {patientChunksInfo.reference_filename} (Parte {patientChunksInfo.navigation_info.recommended_reference_chunk}/1000)</p>
                        )}
                      </div>
                    )}
                  </div>
                )}
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
          <div className="perfil-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h4 style={{ margin: 0, color: '#2c3e50' }}>🧬 Comparación de Secuencias de ADN</h4>
              {sequenceLoaded && (
                <span style={{ 
                  background: '#d4edda', 
                  color: '#155724', 
                  padding: '0.25rem 0.75rem', 
                  borderRadius: '15px',
                  fontSize: '0.875rem',
                  fontWeight: '500'
                }}>
                  ✅ Secuencias cargadas
                </span>
              )}
            </div>
            
          {sequenceLoaded ? (
            <DNAViewer
                patientFiles={patientFiles}
                patientDni={dni}
                title1={`Referencia ${patientChunksInfo?.reference_filename ? `(${patientChunksInfo.reference_filename})` : ''}`}
                title2={`Paciente ${paciente.nombres}`}
                autoLoadPosition={patientChunksInfo?.first_match_position || 0}
                referenceFilename={patientChunksInfo?.reference_filename}
                patientChunkNumber={patientChunksInfo?.reference_chunk_number}
                totalSequenceLength={patientChunksInfo?.total_length}
                hasAlignedSequences={patientChunksInfo?.chunk_type === 'aligned_chunks'}
                blastNavigationInfo={patientChunksInfo?.navigation_info}
            />
          ) : (
              <div className="dna-placeholder" style={{ 
                textAlign: 'center', 
                padding: '3rem 2rem',
                background: '#f8f9fa',
                borderRadius: '8px',
                border: '2px dashed #dee2e6'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🧬</div>
                <h5 style={{ color: '#6c757d', marginBottom: '0.5rem' }}>Visualizador de Secuencias</h5>
                <p style={{ color: '#6c757d', margin: 0 }}>
                  Haz clic en "🧬 Cargar y Visualizar Secuencias" para comenzar la comparación de ADN
                </p>
              </div>
            )}
            </div>
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

      {/* ----- Modal de Alineamiento ----- */}
      {showAlignmentModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h4>🔬 Alineamiento con Cromosoma 13</h4>
              <button
                className="modal-close-btn"
                onClick={() => setShowAlignmentModal(false)}
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <p style={{ marginBottom: '1rem', color: '#6c757d' }}>
                Selecciona un archivo para alinear con la secuencia de referencia cr13.fasta:
              </p>

              {availableFiles.length === 0 ? (
                <p className="empty-text">No hay archivos disponibles para alineamiento.</p>
              ) : (
                <div className="file-selector">
                  <select
                    value={selectedFileForAlignment}
                    onChange={(e) => setSelectedFileForAlignment(e.target.value)}
                    className="alignment-file-select"
                    style={{ width: '100%', padding: '0.5rem', marginBottom: '1rem' }}
                  >
                    <option value="">Selecciona un archivo...</option>
                    {availableFiles.map(file => (
                      <option key={file.key} value={file.filename}>
                        {file.filename}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {alignmentMessage && (
                <div className={`status-message ${alignmentMessage.includes('✅') ? 'success' : alignmentMessage.includes('❌') ? 'error' : alignmentMessage.includes('⏳') ? 'loading' : 'warning'}`}>
                  {alignmentMessage}
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button
                className="control-btn"
                onClick={() => setShowAlignmentModal(false)}
                disabled={alignmentLoading}
              >
                Cancelar
              </button>
              <button
                className={`upload-button ${selectedFileForAlignment ? 'ready' : ''}`}
                onClick={executeAlignment}
                disabled={!selectedFileForAlignment || alignmentLoading}
              >
                {alignmentLoading ? (
                  <>
                    <span className="loading-spinner"></span>
                    Alineando...
                  </>
                ) : (
                  <>🔬 Ejecutar Alineamiento</>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ----- Modal de Procesamiento de Embeddings ----- */}
      {showEmbeddingModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h4>🧠 Generar Embeddings con IA</h4>
              <button
                className="modal-close-btn"
                onClick={() => setShowEmbeddingModal(false)}
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <p style={{ marginBottom: '0.5rem', color: '#6c757d' }}>
                Selecciona un archivo FASTA para generar embeddings usando NVIDIA NIM EVO-2 40B:
              </p>
              <p style={{ marginBottom: '1rem', color: '#856404', fontSize: '0.875rem', fontStyle: 'italic' }}>
                ⚠️ Este proceso puede tomar varios minutos dependiendo del tamaño de la secuencia.
              </p>

              {availableFiles.length === 0 ? (
                <p className="empty-text">No hay archivos FASTA disponibles para procesar.</p>
              ) : (
                <div className="file-selector">
                  <select
                    value={selectedFileForEmbedding}
                    onChange={(e) => setSelectedFileForEmbedding(e.target.value)}
                    className="embedding-file-select"
                    style={{ width: '100%', padding: '0.5rem', marginBottom: '1rem' }}
                  >
                    <option value="">Selecciona un archivo...</option>
                    {availableFiles.map(file => (
                      <option key={file.key} value={file.filename}>
                        {file.filename}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {embeddingMessage && (
                <div className={`status-message ${embeddingMessage.includes('✅') ? 'success' : embeddingMessage.includes('❌') ? 'error' : embeddingMessage.includes('⏳') ? 'loading' : 'warning'}`}>
                  <pre style={{ margin: 0, fontFamily: 'inherit', whiteSpace: 'pre-wrap' }}>
                    {embeddingMessage}
                  </pre>
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button
                className="control-btn"
                onClick={() => setShowEmbeddingModal(false)}
                disabled={embeddingLoading}
              >
                Cancelar
              </button>
              <button
                className={`upload-button ${selectedFileForEmbedding ? 'ready' : ''}`}
                onClick={executeEmbeddingProcessing}
                disabled={!selectedFileForEmbedding || embeddingLoading}
              >
                {embeddingLoading ? (
                  <>
                    <span className="loading-spinner"></span>
                    Procesando con IA...
                  </>
                ) : (
                  <>🧠 Generar Embeddings</>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Perfil
