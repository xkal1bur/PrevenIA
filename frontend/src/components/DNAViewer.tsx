import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import axios from 'axios'
import './DNAViewer.css'

const extractSequenceFromFasta = (text: string): string =>
  text
    .split("\n")
    .filter(l => !l.startsWith(">") && l.trim())
    .map(l => l.trim().toUpperCase().replace(/[^ATCG]/g, ""))
    .join("")

interface DNAViewerProps {
  sequence1?: string
  sequence2?: string
  title1?: string
  title2?: string
  allowExport?: boolean
  patientFiles?: string[] // Array de nombres de archivos del paciente
  patientDni?: string
  autoLoadPosition?: number // Posición automática para cargar
  referenceFilename?: string // Nombre del archivo de referencia a cargar automáticamente
  patientChunkNumber?: number // Número específico del chunk del paciente a cargar (1-1000)
  totalSequenceLength?: number // Longitud total de la secuencia (para chunks alineados)
  hasAlignedSequences?: boolean // Si las secuencias están alineadas con BLAST
  // Nueva prop para información de navegación de BLAST
  blastNavigationInfo?: {
    match_start_position: number;
    match_end_position: number;
    recommended_chunk: number;
    position_in_chunk: number;
  }
  // Callback para notificar al padre cuando se sube un FASTA con mismatches
  onUploadSuccess?: () => void;
}

interface SequenceChunk {
  data: string
  startPosition: number
  endPosition: number
}

interface FragmentCache {
  [key: string]: string
}

const DNAViewer: React.FC<DNAViewerProps> = ({ 
  sequence1: initialSeq1 = "", 
  title1 = "Secuencia Referencia",
  title2 = "Secuencia Paciente",
  allowExport = true,
  patientFiles = [],
  patientDni,
  autoLoadPosition = 0,
  referenceFilename,
  patientChunkNumber,
  totalSequenceLength,
  hasAlignedSequences = false,
  blastNavigationInfo,
  onUploadSuccess
}) => {
  const [referenceSequence, setReferenceSequence] = useState<string>(initialSeq1)
  const [viewStart, setViewStart] = useState<number>(0)
  const [totalLength, setTotalLength] = useState<number>(0)
  const [currentChunk, setCurrentChunk] = useState<SequenceChunk | null>(null)
  const [patientChunk, setPatientChunk] = useState<SequenceChunk | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  
  // Cache para fragmentos ya descargados
  const [patientFragmentCache, setPatientFragmentCache] = useState<FragmentCache>({})
  const [referenceFragmentCache, setReferenceFragmentCache] = useState<FragmentCache>({})
  
  const sequenceRef = useRef<HTMLDivElement>(null)
  const [refFiles, setRefFiles] = useState<string[]>([])
  const [showRefPicker, setShowRefPicker] = useState(false)

  // Constantes optimizadas para 1000 fragmentos
  const CHUNK_SIZE = 1000 // Bases por chunk visible (reducido)
  const NAVIGATION_STEP = 500 // Paso de navegación
  const TOTAL_FRAGMENTS = 1000 // Ahora usamos 1000 fragmentos

  // Función para obtener el número de fragmento basado en posición
  const getFragmentNumber = useCallback((position: number): number => {
    if (totalLength === 0) return 1
    const fragmentSize = Math.ceil(totalLength / TOTAL_FRAGMENTS)
    return Math.min(Math.floor(position / fragmentSize) + 1, TOTAL_FRAGMENTS)
  }, [totalLength])

  // Calcular título dinámico de referencia basado en posición actual
  const currentReferenceTitle = useMemo(() => {
    const currentFragmentNumber = getFragmentNumber(viewStart)
    const referenceFilename = `default_part_${currentFragmentNumber.toString().padStart(4, '0')}.fasta`
    return `${title1.replace(/\s*\([^)]*\)/, '')} (${referenceFilename})`
  }, [title1, viewStart, getFragmentNumber])

  // Calcular título dinámico del paciente basado en posición actual
  const currentPatientTitle = useMemo(() => {
    const currentFragmentNumber = getFragmentNumber(viewStart)
    const hasAligned = patientFiles.some(f => f.includes('aligned_part_'))
    const prefix = hasAligned ? 'aligned_part_' : 'patient_part_'
    const patientFilename = `${prefix}${currentFragmentNumber.toString().padStart(4, '0')}.fasta`
    const baseTitle = title2.replace(/\s*\([^)]*\)/, '')
    return `${baseTitle} (${patientFilename}) - Fragmento ${currentFragmentNumber}/1000`
  }, [title2, viewStart, getFragmentNumber, patientFiles])

  // Función para cargar fragmento de referencia específico
  const loadReferenceFragment = useCallback(async (fragmentNumber: number): Promise<string> => {
    const fragmentKey = `ref_${fragmentNumber}`
    
    // Verificar cache primero
    if (referenceFragmentCache[fragmentKey]) {
      return referenceFragmentCache[fragmentKey]
    }

    try {
      const filename = `default_part_${fragmentNumber.toString().padStart(4, '0')}.fasta`
      const response = await axios.get(
        `http://localhost:8000/split_fasta_files/${encodeURIComponent(filename)}`,
        { responseType: 'text' }
      )
      
      const fragmentData = extractSequenceFromFasta(response.data)
      
      // Guardar en cache
      setReferenceFragmentCache(prev => ({
        ...prev,
        [fragmentKey]: fragmentData
      }))
      
      return fragmentData
    } catch (error) {
      console.error(`Error loading reference fragment ${fragmentNumber}:`, error)
      return ''
    }
  }, [referenceFragmentCache])

  // Función para cargar fragmento del paciente específico
  const loadPatientFragment = useCallback(async (fragmentNumber: number): Promise<string> => {
    if (!patientDni || patientFiles.length === 0) return ''
    
    const fragmentKey = `patient_${fragmentNumber}`
    
    // Verificar cache primero
    if (patientFragmentCache[fragmentKey]) {
      return patientFragmentCache[fragmentKey]
    }

    try {
      const headers = {
        Authorization: `Bearer ${localStorage.getItem('token')}`
      }

      // Determinar el tipo de archivo basado en patientFiles
      let filename = ''
      if (patientFiles.some(f => f.includes('aligned_part_'))) {
        filename = `aligned_part_${fragmentNumber.toString().padStart(4, '0')}.fasta`
      } else {
        filename = `patient_part_${fragmentNumber.toString().padStart(4, '0')}.fasta`
      }

      const response = await axios.get(
        `http://localhost:8000/pacientes/${patientDni}/files/${encodeURIComponent(filename)}`,
        { headers, responseType: 'text' }
      )
      
      const fragmentData = response.data.trim().toUpperCase()
      
      // Guardar en cache
      setPatientFragmentCache(prev => ({
        ...prev,
        [fragmentKey]: fragmentData
      }))
      
      return fragmentData
    } catch (error) {
      console.error(`Error loading patient fragment ${fragmentNumber}:`, error)
      return ''
    }
  }, [patientDni, patientFiles, patientFragmentCache])

  // Función optimizada para obtener chunk de referencia
  const getReferenceChunk = useCallback(async (start: number, size: number): Promise<SequenceChunk> => {
    const end = Math.min(start + size, totalLength)
    
    // Determinar qué fragmentos necesitamos
    const startFragment = getFragmentNumber(start)
    const endFragment = getFragmentNumber(end - 1)
    
    let combinedData = ''
    const fragmentSize = Math.ceil(totalLength / TOTAL_FRAGMENTS)
    
    // Cargar solo los fragmentos necesarios
    for (let fragNum = startFragment; fragNum <= endFragment; fragNum++) {
      const fragmentData = await loadReferenceFragment(fragNum)
      
      // Calcular qué parte de este fragmento necesitamos
      const fragStart = (fragNum - 1) * fragmentSize
      
      const chunkStartInFrag = Math.max(0, start - fragStart)
      const chunkEndInFrag = Math.min(fragmentData.length, end - fragStart)
      
      if (chunkStartInFrag < chunkEndInFrag && fragmentData) {
        combinedData += fragmentData.slice(chunkStartInFrag, chunkEndInFrag)
      }
    }

    return {
      data: combinedData,
      startPosition: start,
      endPosition: start + combinedData.length - 1
    }
  }, [totalLength, getFragmentNumber, loadReferenceFragment])

  // Función optimizada para cargar chunk del paciente
  const loadPatientChunk = useCallback(async (start: number, size: number): Promise<SequenceChunk> => {
    if (!patientDni || patientFiles.length === 0) {
      return {
        data: '-'.repeat(size),
        startPosition: start,
        endPosition: start + size - 1
      }
    }

    try {
      setLoading(true)
      
      const end = Math.min(start + size, totalLength)
      
      // Determinar qué fragmentos necesitamos
      const startFragment = getFragmentNumber(start)
      const endFragment = getFragmentNumber(end - 1)
      
      let combinedData = ''
      const fragmentSize = Math.ceil(totalLength / TOTAL_FRAGMENTS)
      
      // Cargar solo los fragmentos necesarios
      for (let fragNum = startFragment; fragNum <= endFragment; fragNum++) {
        const fragmentData = await loadPatientFragment(fragNum)
        
        if (fragmentData) {
          // Calcular qué parte de este fragmento necesitamos
          const fragStart = (fragNum - 1) * fragmentSize
          
          const chunkStartInFrag = Math.max(0, start - fragStart)
          const chunkEndInFrag = Math.min(fragmentData.length, end - fragStart)
          
          if (chunkStartInFrag < chunkEndInFrag) {
            combinedData += fragmentData.slice(chunkStartInFrag, chunkEndInFrag)
          }
        } else {
          // Si no hay datos, llenar con '-'
          const expectedSize = Math.min(fragmentSize, end - (fragNum - 1) * fragmentSize)
          combinedData += '-'.repeat(expectedSize)
        }
      }

      return {
        data: combinedData,
        startPosition: start,
        endPosition: start + combinedData.length - 1
      }
    } catch (error) {
      console.error('Error loading patient chunk:', error)
      return {
        data: '-'.repeat(size),
        startPosition: start,
        endPosition: start + size - 1
      }
    } finally {
      setLoading(false)
    }
  }, [patientDni, patientFiles, totalLength, getFragmentNumber, loadPatientFragment])

  // Función para cargar chunks actuales
  const loadCurrentChunks = useCallback(async () => {
    if (totalLength === 0) return

    const refChunk = await getReferenceChunk(viewStart, CHUNK_SIZE)
    setCurrentChunk(refChunk)

    const patChunk = await loadPatientChunk(viewStart, CHUNK_SIZE)
    setPatientChunk(patChunk)
  }, [viewStart, totalLength, getReferenceChunk, loadPatientChunk])

  // Efecto para cargar chunks cuando cambia la vista
  useEffect(() => {
    loadCurrentChunks()
  }, [viewStart, totalLength]) // Usar dependencias específicas en lugar de la función

  // Efecto para cargar automáticamente el archivo de referencia
  useEffect(() => {
    if (referenceFilename && !referenceSequence) {
      const loadReferenceFile = async () => {
        try {
          const res = await axios.get<string>(
            `http://localhost:8000/split_fasta_files/${encodeURIComponent(referenceFilename)}`,
            { responseType: "text" }
          );
          const seq = extractSequenceFromFasta(res.data);
          setReferenceSequence(seq);
          console.log(`✅ Archivo de referencia ${referenceFilename} cargado automáticamente`);
        } catch (err) {
          console.error(`Error cargando archivo de referencia ${referenceFilename}:`, err);
        }
      };
      
      loadReferenceFile();
    }
  }, [referenceFilename]); // Removido referenceSequence de las dependencias para evitar bucle

  // Efecto para auto-cargar en la posición de match cuando se cargan archivos del paciente
  useEffect(() => {
    if (patientFiles.length > 0 && totalLength > 0) {
      let targetPosition = 0;
      
      // Usar información de BLAST si está disponible (más precisa)
      if (blastNavigationInfo && hasAlignedSequences) {
        targetPosition = blastNavigationInfo.match_start_position;
        console.log(`🎯 Navegando usando BLAST: posición ${targetPosition} (chunk recomendado: ${blastNavigationInfo.recommended_chunk})`);
      } 
      // Fallback a autoLoadPosition si no hay info de BLAST
      else if (autoLoadPosition > 0) {
        targetPosition = autoLoadPosition;
        console.log(`📍 Navegando usando autoLoadPosition: ${targetPosition}`);
      }
      
      if (targetPosition > 0) {
        const adjustedPosition = Math.max(0, Math.min(
          targetPosition,
          totalLength - CHUNK_SIZE
        ));
        setViewStart(adjustedPosition);
      }
    }
  }, [patientFiles, autoLoadPosition, totalLength, blastNavigationInfo, hasAlignedSequences])

  // Efecto inicial para establecer la longitud total
  useEffect(() => {
    if (totalSequenceLength && hasAlignedSequences) {
      // Si tenemos la longitud exacta de chunks alineados, usarla
      setTotalLength(totalSequenceLength)
    } else if (referenceSequence) {
      setTotalLength(referenceSequence.length)
    } else if (patientChunkNumber) {
      // Si se especifica un chunk específico, usar una longitud estimada para un fragmento
      setTotalLength(referenceSequence.length || 250000) // longitud típica de un fragmento
    } else if (patientFiles.length === TOTAL_FRAGMENTS) {
      // Si hay exactamente 1000 archivos del paciente, usar la longitud de la secuencia de referencia
      // o una longitud estimada basada en el tamaño típico de secuencias genómicas
      setTotalLength(referenceSequence.length || 5000000) // usar referencia o 5M por defecto
    } else if (patientFiles.length > 0) {
      // Si hay archivos del paciente pero no 1000, estimar basándose en los disponibles
      setTotalLength(referenceSequence.length || 1000000) // usar referencia o 1M por defecto
    }
  }, [referenceSequence, patientFiles, patientChunkNumber, totalSequenceLength, hasAlignedSequences])

  // Función para comparar chunks
  const compareChunks = useMemo(() => {
    if (!currentChunk || !patientChunk) return { result1: [], result2: [] }

    const maxLength = Math.max(currentChunk.data.length, patientChunk.data.length)
    const result1: Array<{ char: string; isMatch: boolean; index: number; isPatientGap: boolean }> = []
    const result2: Array<{ char: string; isMatch: boolean; index: number; isPatientGap: boolean }> = []
    
    for (let i = 0; i < maxLength; i++) {
      const char1 = currentChunk.data[i] || '-'
      const char2 = patientChunk.data[i] || '-'
      const isMatch = char1 === char2
      const isPatientGap = char2 === '-' // Nuevo: detectar cuando el paciente tiene gap
      const globalIndex = currentChunk.startPosition + i
        
      result1.push({ char: char1, isMatch, index: globalIndex, isPatientGap })
      result2.push({ char: char2, isMatch, index: globalIndex, isPatientGap })
    }
    
    return { result1, result2 }
  }, [currentChunk, patientChunk])

  // Calcular estadísticas del chunk actual
  const { result1, result2 } = compareChunks
  const chunkMismatches = result1.filter(item => !item.isMatch).length
  const chunkMatchPercentage = result1.length > 0 
    ? ((result1.length - chunkMismatches) / result1.length * 100).toFixed(1)
    : '0.0'

  // Funciones de navegación
  const moveLeft = () => {
    setViewStart(prev => Math.max(0, prev - NAVIGATION_STEP))
  }

  const moveRight = () => {
    setViewStart(prev => Math.min(totalLength - CHUNK_SIZE, prev + NAVIGATION_STEP))
  }

  const jumpToPosition = () => {
    const input = prompt(`Ir a posición (1 - ${totalLength}):`)
    if (input) {
      const position = parseInt(input) - 1
      if (position >= 0 && position < totalLength) {
        setViewStart(Math.max(0, Math.min(position, totalLength - CHUNK_SIZE)))
      }
    }
  }

  // Funciones para archivos de referencia
  const loadRefFiles = useCallback(async () => {
    try {
      const res = await axios.get<{ files: string[] }>(
        "http://localhost:8000/reference-files"
      );
      setRefFiles(res.data.files || []);
    } catch (err) {
      console.error("Error listando archivos de referencia:", err);
      setRefFiles([]);
    }
  }, []);

  const handleOpenPicker = () => {
    setShowRefPicker(true)
    loadRefFiles()
  }

  const handleReferenceSelect = useCallback(
    async (key: string) => {
      try {
        const filename = key.replace("split_fasta_files/", "");
        
        // Extraer el número del fragmento del nombre del archivo
        const fragmentMatch = filename.match(/default_part_(\d+)\.fasta/);
        const fragmentNumber = fragmentMatch ? parseInt(fragmentMatch[1]) : 1;
        
        const res = await axios.get<string>(
          `http://localhost:8000/split_fasta_files/${encodeURIComponent(filename)}`,
          { responseType: "text" }
        );
        const seq = extractSequenceFromFasta(res.data);
        setReferenceSequence(seq);
        
        // Navegar al fragmento seleccionado en lugar de resetear a 0
        if (totalLength > 0 && fragmentNumber > 1) {
          const fragmentSize = Math.ceil(totalLength / TOTAL_FRAGMENTS);
          const targetPosition = (fragmentNumber - 1) * fragmentSize;
          const adjustedPosition = Math.max(0, Math.min(
            targetPosition,
            totalLength - CHUNK_SIZE
          ));
          setViewStart(adjustedPosition);
          console.log(`🎯 Navegando al fragmento ${fragmentNumber} (posición ${targetPosition})`);
        } else {
          setViewStart(0); // Solo resetear si es el fragmento 1 o no hay totalLength
        }
        
        // Limpiar cache cuando se cambia de referencia
        setReferenceFragmentCache({});
      } catch (err) {
        console.error("Error cargando referencia:", err);
        alert("No se pudo cargar el archivo de referencia.");
      } finally {
        setShowRefPicker(false);
      }
    },
    [totalLength]
  );

  // Función para navegar al hacer click en el minimap
  const handleMinimapClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const clickX = event.clientX - rect.left
    const clickPercentage = clickX / rect.width
    const targetPosition = Math.floor(clickPercentage * totalLength)
    
    const newViewStart = Math.max(0, Math.min(
      targetPosition - Math.floor(CHUNK_SIZE / 2),
      totalLength - CHUNK_SIZE
    ))
    
    setViewStart(newViewStart)
  }

  // Función para exportar FASTA con ventanas de 129 bp alrededor de cada mismatch
  const uploadMismatchFasta = useCallback(async () => {
    if (!currentChunk || result1.length === 0 || !patientDni) {
      alert('No hay datos suficientes para exportar')
      return
    }

    // Solo los realmente coloreados como 'mismatch' (rojo)
    const mismatches = result1.filter(item => !item.isMatch && !item.isPatientGap && item.char !== '-')

    if (mismatches.length === 0) {
      alert('✅ No se encontraron diferencias en el fragmento actual')
      return
    }

    let fastaContent = ''

    for (let i = 0; i < mismatches.length; i++) {
      const { index: globalPos } = mismatches[i]

      // Calcular ventana (64 antes y 64 después)
      const windowStart = Math.max(0, globalPos - 64)
      const windowEnd   = Math.min(totalLength - 1, globalPos + 64)
      const windowSize  = windowEnd - windowStart + 1

      // Obtener subsecuencia del PACIENTE (no referencia) usando la función optimizada
      const patientSubSeqChunk = await loadPatientChunk(windowStart, windowSize)

      // Añadir entrada al FASTA con la secuencia del paciente (variante)
      fastaContent += `>Mismatch_${i + 1}_pos_${globalPos + 1}_window_${windowStart + 1}-${windowEnd + 1}\n`
      fastaContent += patientSubSeqChunk.data + '\n'
    }

    try {
      const blob = new Blob([fastaContent], { type: 'text/plain' })
      const fileName = `mismatches_${currentChunk.startPosition + 1}-${currentChunk.endPosition + 1}_${new Date().toISOString().replace(/[:T.]/g, '-').split('Z')[0]}.fasta`

      const formData = new FormData()
      formData.append('fasta_file', new File([blob], fileName, { type: 'text/plain' }))

      // Enviar al backend para que lo suba a S3
      const headers = {
        Authorization: `Bearer ${localStorage.getItem('token') || ''}`
      }

      await axios.post(`http://localhost:8000/pacientes/${patientDni}/upload_fasta`, formData, { headers })

      alert(`✅ FASTA con mismatches subido a S3 como ${fileName}`)

      // Notificar al componente padre para refrescar la lista de archivos
      if (onUploadSuccess) onUploadSuccess()
    } catch (err) {
      console.error('Error subiendo mismatches:', err)
      alert('❌ Error al subir el archivo a S3')
    }
  }, [currentChunk, result1, totalLength, getReferenceChunk, patientDni, onUploadSuccess])

  // Validación
  if (totalLength === 0) {
    return (
      <div className="dna-viewer">
        <div className="dna-viewer-header">
          <h3>Visualizador de Secuencias de ADN</h3>
        </div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#6c757d' }}>
          <p>No hay secuencias de ADN para mostrar.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="dna-viewer">
      <div className="dna-viewer-header">
        <h3>Visualizador de Secuencias de ADN</h3>
        <div className="dna-stats">
          <span className="stat">Total: {totalLength.toLocaleString()} bases</span>
          <span className="stat">Chunk: {chunkMatchPercentage}% coincidencias</span>
          <span className="stat mismatches">Diferencias: {chunkMismatches}</span>
          <span className="stat">Cache: R{Object.keys(referenceFragmentCache).length}/P{Object.keys(patientFragmentCache).length}</span>
          {loading && <span className="stat loading">Cargando...</span>}
        </div>
      </div>

      {/* Controles de navegación */}
      <div className="dna-controls">
        <div className="nav-controls">
          <button onClick={moveLeft} className="control-btn" disabled={viewStart <= 0}>◀</button>
          <span className="position-info">
            Posición: {(viewStart + 1).toLocaleString()} - {Math.min(viewStart + CHUNK_SIZE, totalLength).toLocaleString()} de {totalLength.toLocaleString()} | Fragmento: {getFragmentNumber(viewStart)}/1000
          </span>
          <button onClick={moveRight} className="control-btn" disabled={viewStart + CHUNK_SIZE >= totalLength}>▶</button>
          <button onClick={jumpToPosition} className="control-btn">Ir a posición</button>
        </div>
        <div className="zoom-controls">
          {allowExport && (
            <button onClick={uploadMismatchFasta} className="control-btn fasta-export" title="Exportar regiones de mismatch (129 bp)" disabled={!currentChunk}>
              📄 Exportar Mismatches
            </button>
          )}
        </div>
      </div>

      {/* Minimap/Overview */}
        <div className="sequence-overview">
          <div className="overview-track" onClick={handleMinimapClick} style={{ cursor: 'pointer' }}>
            {/* Viewport indicator */}
            <div 
              className="overview-viewport"
              style={{
              left: `${(viewStart / totalLength) * 100}%`,
              width: `${(CHUNK_SIZE / totalLength) * 100}%`
                }}
              />
          </div>
          <div className="minimap-instructions">
            <span style={{ fontSize: '0.75rem', color: '#6c757d' }}>
            💡 Haz click en la barra para navegar rápidamente por la secuencia completa (1000 fragmentos)
            </span>
          </div>
        </div>

      <div className="dna-sequences full-view" ref={sequenceRef}>
        <div className="sequence-row">
          <div className="sequence-label" style={{ display: 'flex', flexDirection: 'column' }}>
            {currentReferenceTitle}:
            <button
              className="control-btn"
              style={{ marginTop: '0.25rem', alignSelf: 'flex-start' }}
              onClick={handleOpenPicker}
            >
              Seleccionar Archivo Ref
            </button>
          </div>
          
          <div className="sequence-display">
            {result1.map((item, index) => {
              // Si el paciente tiene gap en esta posición, mostrar referencia en gris
              const getClassForReference = () => {
                if (item.isPatientGap) return 'patient-gap'
                if (item.char === '-') return 'deletion'
                if (!item.isMatch) return 'mismatch'
                return 'match'
              }
              
              return (
                <span
                  key={`seq1-${index}`}
                  className={`base ${getClassForReference()}`}
                  title={`Posición ${item.index + 1}: ${item.char === '-' ? 'Deleción' : item.char}${item.isPatientGap ? ' (Gap en paciente)' : ''}`}
                  style={{
                    fontSize: '0.75rem',
                    padding: '0.1rem 0.15rem',
                    margin: '0.05rem'
                  }}
                >
                  {item.char === '-' ? '-' : item.char}
                </span>
              )
            })}
          </div>
        </div>

        <div className="sequence-row">
          <div className="sequence-label">{currentPatientTitle}:</div>
          <div className="sequence-display">
            {result2.map((item, index) => {
              // Si el paciente tiene gap, mostrar en gris
              const getClassForPatient = () => {
                if (item.isPatientGap) return 'patient-gap'
                if (item.char === '-') return 'deletion'
                if (!item.isMatch) return 'mismatch'
                return 'match'
              }
              
              return (
                <span
                  key={`seq2-${index}`}
                  className={`base ${getClassForPatient()}`}
                  title={`Posición ${item.index + 1}: ${item.char === '-' ? 'Gap del paciente' : item.char}`}
                  style={{ 
                    fontSize: '0.75rem',
                    padding: '0.1rem 0.15rem',
                    margin: '0.05rem'
                  }}
                >
                  {item.char === '-' ? '-' : item.char}
                </span>
              )
            })}
          </div>
        </div>
      </div>

      {showRefPicker && (
        <div className="modal-overlay">
          <div className="modal-content ref-picker-modal">
            <div className="modal-header">
              <h4>Archivos de Referencia</h4>
              <button
                className="modal-close-btn"
                onClick={() => setShowRefPicker(false)}
              >
                ×
              </button>
            </div>

            <div className="modal-body">
              {refFiles.length === 0 ? (
                <p className="empty-text">No se encontraron archivos.</p>
              ) : (
                <ul className="ref-list">
                  {(refFiles || []).map(key => (
                    <li
                      key={key}
                      className="ref-list-item"
                      onClick={() => handleReferenceSelect(key)}
                    >
                      {key.replace('split_fasta_files/', '')}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="modal-footer">
              <button
                className="control-btn"
                onClick={() => setShowRefPicker(false)}
              >
                Cerrar
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="dna-legend">
        <div className="legend-item">
          <span className="legend-color match"></span>
          <span>Coincidencia</span>
        </div>
        <div className="legend-item">
          <span className="legend-color mismatch"></span>
          <span>Diferencia</span>
        </div>
        <div className="legend-item">
          <span className="legend-color patient-gap"></span>
          <span>Gap del paciente</span>
        </div>
        <div className="legend-instructions">
          <p><strong>Instrucciones:</strong> 
            Usa ◀▶ para navegar • "Ir a posición" para saltar • Click en barra superior para navegación rápida
          </p>
            <p style={{ fontSize: '0.75rem', color: '#6c757d', marginTop: '0.25rem' }}>
            ⚡ Mostrando {CHUNK_SIZE.toLocaleString()} bases de {totalLength.toLocaleString()} • Carga optimizada por fragmentos (1000 total)
            </p>
        </div>
      </div>
    </div>
  )
}

export default DNAViewer 