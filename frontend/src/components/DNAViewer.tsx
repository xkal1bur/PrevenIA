import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import './DNAViewer.css'

interface DNAViewerProps {
  sequence1?: string
  sequence2?: string
  title1?: string
  title2?: string
  allowExport?: boolean
}

const DNAViewer: React.FC<DNAViewerProps> = ({ 
  sequence1 = "ATCGATCGATCGAAGGCTACGTACGTACGTATCGATCGATCGCCGTTAAGGCCTACGTACGTAATCGATCGATCGAAGGCTACGTACGTACGTATCGATCGATCGCCGTTAAGGCCTACGTACGTAATCGATCGATCGAAGGCTACGTACGTACGTATCGATCGATCGCCGTTAAGGCCTACGTACGT", 
  sequence2 = "ATCGATCGATCGAAGGCTACGTCCGTACGTATCGATCGATCGCCGTTAAGGCCTACGTACGTAATCGATCGATCGAAGGCTACGTACGTACGTATCGATCGATCGCCGTTAAGGCCTACGTACGTAATCGATCGATCGAAGGCTACGTACGTACGTATCGATCGATCGCCGTTAAGGCCTACGTACGT",
  title1 = "Secuencia Referencia",
  title2 = "Secuencia Paciente",
  allowExport = true
}) => {
  const [zoomLevel, setZoomLevel] = useState<number>(1)
  const [viewStart, setViewStart] = useState<number>(0)
  const [selection, setSelection] = useState<{ start: number; end: number } | null>(null)
  const [isSelecting, setIsSelecting] = useState<boolean>(false)
  const [selectionStart, setSelectionStart] = useState<number>(0)
  
  const sequenceRef = useRef<HTMLDivElement>(null)
  
  // Constantes para optimización de rendimiento
  const MAX_VISIBLE_BASES = 500 // Máximo número de bases a renderizar a la vez
  const CHUNK_SIZE = 100 // Tamaño de chunk para cálculos
  const OVERVIEW_RESOLUTION = 1000 // Resolución del minimap para secuencias largas
  
  // Función optimizada para comparar las secuencias usando chunks
  const compareSequencesOptimized = useCallback((seq1: string, seq2: string) => {
    const maxLength = Math.max(seq1.length, seq2.length)
    const result1: Array<{ char: string; isMatch: boolean; index: number }> = []
    const result2: Array<{ char: string; isMatch: boolean; index: number }> = []
    
    // Procesar en chunks para evitar bloquear el UI con secuencias muy largas
    for (let chunkStart = 0; chunkStart < maxLength; chunkStart += CHUNK_SIZE) {
      const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, maxLength)
      
      for (let i = chunkStart; i < chunkEnd; i++) {
        const char1 = seq1[i] || '-'
        const char2 = seq2[i] || '-'
        const isMatch = char1 === char2
        
        result1.push({ char: char1, isMatch, index: i })
        result2.push({ char: char2, isMatch, index: i })
      }
    }
    
    return { result1, result2 }
  }, [CHUNK_SIZE])

  const { result1, result2 } = useMemo(() => 
    compareSequencesOptimized(sequence1, sequence2), 
    [sequence1, sequence2, compareSequencesOptimized]
  )
  
  // Lógica de visualización optimizada para secuencias largas
  const totalBases = result1.length
  const isZoomedIn = zoomLevel > 1
  const isLargeSequence = totalBases > 2000
  
  // Calcular qué mostrar basado en el zoom y tamaño de secuencia
  const currentViewStart = (isZoomedIn || isLargeSequence) ? viewStart : 0
  
  let basesPerView: number
  if (isZoomedIn) {
    // Mejorar cálculo del zoom para permitir zoom hasta base individual
    basesPerView = Math.max(1, Math.floor(100 / zoomLevel))
  } else if (isLargeSequence) {
    // Para secuencias largas, siempre usar ventana limitada
    basesPerView = MAX_VISIBLE_BASES
  } else {
    basesPerView = totalBases
  }
  
  const viewEnd = Math.min(currentViewStart + basesPerView, totalBases)
  const displayResult1 = result1.slice(currentViewStart, viewEnd)
  const displayResult2 = result2.slice(currentViewStart, viewEnd)
  
  // Calcular estadísticas
  const mismatches = result1.filter(item => !item.isMatch).length
  const matchPercentage = ((totalBases - mismatches) / totalBases * 100).toFixed(1)

  // Manejadores para la selección
  const handleMouseDown = useCallback((index: number) => {
    setIsSelecting(true)
    const globalIndex = (isZoomedIn || isLargeSequence) ? currentViewStart + index : index
    setSelectionStart(globalIndex)
    setSelection(null)
  }, [currentViewStart, isZoomedIn, isLargeSequence])

  const handleMouseMove = useCallback((index: number) => {
    if (isSelecting) {
      const globalIndex = (isZoomedIn || isLargeSequence) ? currentViewStart + index : index
      const start = Math.min(selectionStart, globalIndex)
      const end = Math.max(selectionStart, globalIndex)
      setSelection({ start, end })
    }
  }, [isSelecting, selectionStart, currentViewStart, isZoomedIn, isLargeSequence])

  const handleMouseUp = useCallback(() => {
    setIsSelecting(false)
  }, [])

  // Funciones de zoom y navegación mejoradas
  const zoomIn = () => {
    const newZoomLevel = Math.min(zoomLevel + 1, 20) // Permitir más zoom
    
    if (zoomLevel === 1 || (!isZoomedIn && isLargeSequence)) {
      // Primera vez que hace zoom, centrarse en el medio o en la selección
      if (selection) {
        const selectionCenter = Math.floor((selection.start + selection.end) / 2)
        const newBasesPerView = Math.floor(50 / newZoomLevel) + 10
        setViewStart(Math.max(0, selectionCenter - Math.floor(newBasesPerView / 2)))
      } else {
        const newBasesPerView = Math.floor(50 / newZoomLevel) + 10
        setViewStart(Math.max(0, Math.floor(totalBases / 2) - Math.floor(newBasesPerView / 2)))
      }
    }
    setZoomLevel(newZoomLevel)
  }

  const zoomOut = () => {
    if (zoomLevel <= 2) {
      setZoomLevel(1)
      setViewStart(0)
    } else {
      setZoomLevel(prev => Math.max(prev - 1, 1))
    }
  }

  const zoomToSelection = () => {
    if (selection) {
      const selectionLength = selection.end - selection.start + 1
      // Calcular zoom para mostrar la selección con un poco de contexto
      const targetBasesVisible = Math.max(selectionLength * 2, 10) // Al menos 10 bases visibles
      const newZoomLevel = Math.max(2, Math.min(Math.floor(100 / targetBasesVisible), 20))
      
      setZoomLevel(newZoomLevel)
      
      // Centrar la vista en la selección
      const selectionCenter = Math.floor((selection.start + selection.end) / 2)
      const newBasesPerView = Math.max(1, Math.floor(100 / newZoomLevel))
      const newViewStart = Math.max(0, Math.min(
        selectionCenter - Math.floor(newBasesPerView / 2),
        totalBases - newBasesPerView
      ))
      
      setViewStart(newViewStart)
      setSelection(null)
    }
  }

  const resetView = () => {
    setZoomLevel(1)
    setViewStart(0)
    setSelection(null)
  }

  const moveLeft = () => {
    if (isZoomedIn || isLargeSequence) {
      setViewStart(prev => Math.max(0, prev - Math.floor(basesPerView / 4)))
    }
  }

  const moveRight = () => {
    if (isZoomedIn || isLargeSequence) {
      setViewStart(prev => Math.min(totalBases - basesPerView, prev + Math.floor(basesPerView / 4)))
    }
  }

  // Función para navegar al hacer click en el minimap
  const handleMinimapClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const clickX = event.clientX - rect.left
    const clickPercentage = clickX / rect.width
    const targetPosition = Math.floor(clickPercentage * totalBases)
    
    // Centrar la vista en la posición clickeada
    const newViewStart = Math.max(0, Math.min(
      targetPosition - Math.floor(basesPerView / 2),
      totalBases - basesPerView
    ))
    
    setViewStart(newViewStart)
  }

  // Función para determinar si una base está en la selección
  const isInSelection = (globalIndex: number) => {
    return selection && globalIndex >= selection.start && globalIndex <= selection.end
  }

  // Calcular el tamaño de base dinámicamente
  const getBaseSize = () => {
    if (isZoomedIn) {
      // Escala logarítmica para zoom extremo
      const scaleFactor = Math.min(0.5 + (zoomLevel * 0.3), 4.0)
      return scaleFactor
    } else if (isLargeSequence) {
      // Para secuencias largas, usar tamaño fijo optimizado
      return 0.75
    } else {
      // En vista completa para secuencias pequeñas, ajustar tamaño según longitud
      const containerWidth = 800 // Ancho aproximado del contenedor
      const maxBaseWidth = containerWidth / totalBases
      return Math.max(0.4, Math.min(maxBaseWidth * 20, 1)) // Entre 0.4 y 1 rem
    }
  }

  // Generar datos del minimap optimizado para secuencias largas
  const getOverviewData = useMemo(() => {
    if (totalBases <= OVERVIEW_RESOLUTION) {
      return result1.map((item, index) => ({ isMatch: item.isMatch, index }))
    }
    
    // Para secuencias muy largas, samplear para el minimap
    const step = Math.ceil(totalBases / OVERVIEW_RESOLUTION)
    const overview = []
    for (let i = 0; i < totalBases; i += step) {
      const chunk = result1.slice(i, i + step)
      const hasMatches = chunk.some(item => item.isMatch)
      overview.push({ isMatch: hasMatches, index: i })
    }
    return overview
  }, [result1, totalBases, OVERVIEW_RESOLUTION])



  // Función para analizar diferencias entre secuencias actuales
  const analyzeDifferences = useCallback(() => {
    const differences: Array<{ position: number; ref: string; patient: string; type: 'substitution' | 'deletion' | 'insertion' }> = []
    
    const maxLength = Math.max(sequence1.length, sequence2.length)
    
    for (let i = 0; i < maxLength; i++) {
      const refBase = sequence1[i] || '-'
      const patientBase = sequence2[i] || '-'
      
      if (refBase !== patientBase) {
        let type: 'substitution' | 'deletion' | 'insertion' = 'substitution'
        
        if (refBase === '-') {
          type = 'insertion'
        } else if (patientBase === '-') {
          type = 'deletion'
        }
        
        differences.push({
          position: i + 1,
          ref: refBase,
          patient: patientBase,
          type
        })
      }
    }
    
    return differences
  }, [sequence1, sequence2])

  // Función para generar archivo FASTA
  const generateFastaFile = useCallback(() => {
    const differences = analyzeDifferences()
    
    // Agrupar diferencias por tipo
    const substitutions = differences.filter(d => d.type === 'substitution')
    const deletions = differences.filter(d => d.type === 'deletion')
    const insertions = differences.filter(d => d.type === 'insertion')
    
    // Generar contenido FASTA
    let fastaContent = ''
    
    // Encabezado para secuencia de referencia
    fastaContent += `>${title1.replace(/\s+/g, '_')}|Length=${sequence1.length}\n`
    fastaContent += formatSequenceForFasta(sequence1) + '\n\n'
    
    // Encabezado para secuencia del paciente con información de diferencias
    fastaContent += `>${title2.replace(/\s+/g, '_')}|Length=${sequence2.length}|Substitutions=${substitutions.length}|Deletions=${deletions.length}|Insertions=${insertions.length}\n`
    
    // Agregar información detallada de diferencias
    if (substitutions.length > 0) {
      fastaContent += `# Sustituciones (${substitutions.length}): `
      fastaContent += substitutions.slice(0, 10).map(d => `${d.position}:${d.ref}>${d.patient}`).join(', ')
      if (substitutions.length > 10) fastaContent += '...'
      fastaContent += '\n'
    }
    
    if (deletions.length > 0) {
      fastaContent += `# Deleciones (${deletions.length}): `
      fastaContent += deletions.slice(0, 10).map(d => `${d.position}:${d.ref}>-`).join(', ')
      if (deletions.length > 10) fastaContent += '...'
      fastaContent += '\n'
    }
    
    if (insertions.length > 0) {
      fastaContent += `# Inserciones (${insertions.length}): `
      fastaContent += insertions.slice(0, 10).map(d => `${d.position}:->${d.patient}`).join(', ')
      if (insertions.length > 10) fastaContent += '...'
      fastaContent += '\n'
    }
    
    fastaContent += formatSequenceForFasta(sequence2) + '\n'
    
    return fastaContent
  }, [sequence1, sequence2, title1, title2, analyzeDifferences])

  // Función para formatear secuencia en líneas de 80 caracteres (estándar FASTA)
  const formatSequenceForFasta = (sequence: string): string => {
    const lineLength = 80
    let formatted = ''
    for (let i = 0; i < sequence.length; i += lineLength) {
      formatted += sequence.substring(i, i + lineLength) + '\n'
    }
    return formatted.trim()
  }

  // Función para descargar archivo FASTA
  const downloadFastaFile = useCallback(() => {
    const fastaContent = generateFastaFile()
    const blob = new Blob([fastaContent], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    
    const link = document.createElement('a')
    link.href = url
    link.download = `secuencias_comparacion_${new Date().toISOString().split('T')[0]}.fasta`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }, [generateFastaFile])

  const baseSize = getBaseSize()

  // Efecto para manejar eventos globales del mouse
  useEffect(() => {
    const handleGlobalMouseUp = () => {
      setIsSelecting(false)
    }

    if (isSelecting) {
      document.addEventListener('mouseup', handleGlobalMouseUp)
      return () => document.removeEventListener('mouseup', handleGlobalMouseUp)
    }
  }, [isSelecting])

  // Validación de secuencias vacías
  if (totalBases === 0) {
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
          <span className="stat">Total: {totalBases} bases</span>
          <span className="stat">Coincidencias: {matchPercentage}%</span>
          <span className="stat mismatches">Diferencias: {mismatches}</span>
          <span className="stat">Zoom: {zoomLevel === 1 ? 'Completo' : `${zoomLevel}x`}</span>
        </div>
      </div>

      {/* Controles de navegación */}
      <div className="dna-controls">
        <div className="zoom-controls">
          <button onClick={zoomIn} className="control-btn" disabled={zoomLevel >= 20}>🔍+</button>
          <button onClick={zoomOut} className="control-btn" disabled={zoomLevel <= 1}>🔍-</button>
          <button onClick={resetView} className="control-btn">⌂</button>
          {selection && (
            <button onClick={zoomToSelection} className="control-btn zoom-selection">
              Zoom a selección
            </button>
          )}
          {allowExport && (
            <button onClick={downloadFastaFile} className="control-btn fasta-export" title="Exportar como archivo FASTA">
              📄 FASTA
            </button>
          )}
        </div>
        {(isZoomedIn || isLargeSequence) && (
          <div className="nav-controls">
            <button onClick={moveLeft} className="control-btn" disabled={currentViewStart <= 0}>◀</button>
            <span className="position-info">
              Posición: {currentViewStart + 1} - {viewEnd} de {totalBases}
              {isLargeSequence && !isZoomedIn && (
                <span style={{ fontSize: '0.75rem', color: '#6c757d', marginLeft: '0.5rem' }}>
                  (Vista optimizada)
                </span>
              )}
            </span>
            <button onClick={moveRight} className="control-btn" disabled={viewEnd >= totalBases}>▶</button>
          </div>
        )}
      </div>

      {/* Minimap/Overview - mostrar para secuencias largas o cuando está en zoom */}
      {(isZoomedIn || isLargeSequence) && (
        <div className="sequence-overview">
          <div className="overview-track" onClick={handleMinimapClick} style={{ cursor: 'pointer' }}>
            {/* Renderizar minimap optimizado para secuencias largas */}
            {isLargeSequence && getOverviewData.map((point, index) => (
              <div
                key={index}
                className={`overview-base ${point.isMatch ? 'match' : 'mismatch'}`}
                style={{
                  left: `${(point.index / totalBases) * 100}%`,
                  width: `${100 / getOverviewData.length}%`
                }}
              />
            ))}
            
            {/* Viewport indicator */}
            <div 
              className="overview-viewport"
              style={{
                left: `${(currentViewStart / totalBases) * 100}%`,
                width: `${(basesPerView / totalBases) * 100}%`
              }}
            />
            
            {/* Selection indicator */}
            {selection && (
              <div 
                className="overview-selection"
                style={{
                  left: `${(selection.start / totalBases) * 100}%`,
                  width: `${((selection.end - selection.start + 1) / totalBases) * 100}%`
                }}
              />
            )}
          </div>
          <div className="minimap-instructions">
            <span style={{ fontSize: '0.75rem', color: '#6c757d' }}>
              💡 Haz click en la barra para navegar rápidamente
            </span>
          </div>
        </div>
      )}

      <div className={`dna-sequences ${isZoomedIn ? 'zoomed' : 'full-view'}`} ref={sequenceRef}>
        <div className="sequence-row">
          <div className="sequence-label">{title1}:</div>
          <div className="sequence-display">
            {displayResult1.map((item, index) => {
              const isDeletion = item.char === '-'
              const baseClass = isDeletion ? 'deletion' : (!item.isMatch ? 'mismatch' : 'match')
              
              return (
                <span
                  key={`seq1-${index}`}
                  className={`base ${baseClass} ${isInSelection(item.index) ? 'selected' : ''}`}
                  title={`Posición ${item.index + 1}: ${isDeletion ? 'Deleción' : item.char}`}
                  onMouseDown={() => handleMouseDown(index)}
                  onMouseMove={() => handleMouseMove(index)}
                  onMouseUp={handleMouseUp}
                  style={{ 
                    fontSize: `${baseSize}rem`,
                    padding: `${baseSize * 0.1}rem ${baseSize * 0.2}rem`,
                    margin: `${baseSize * 0.05}rem`
                  }}
                >
                  {isDeletion ? '-' : item.char}
                </span>
              )
            })}
          </div>
        </div>

        <div className="sequence-row">
          <div className="sequence-label">{title2}:</div>
          <div className="sequence-display">
            {displayResult2.map((item, index) => {
              const isDeletion = item.char === '-'
              const baseClass = isDeletion ? 'deletion' : (!item.isMatch ? 'mismatch' : 'match')
              
              return (
                <span
                  key={`seq2-${index}`}
                  className={`base ${baseClass} ${isInSelection(item.index) ? 'selected' : ''}`}
                  title={`Posición ${item.index + 1}: ${isDeletion ? 'Deleción' : item.char}`}
                  onMouseDown={() => handleMouseDown(index)}
                  onMouseMove={() => handleMouseMove(index)}
                  onMouseUp={handleMouseUp}
                  style={{ 
                    fontSize: `${baseSize}rem`,
                    padding: `${baseSize * 0.1}rem ${baseSize * 0.2}rem`,
                    margin: `${baseSize * 0.05}rem`
                  }}
                >
                  {isDeletion ? '-' : item.char}
                </span>
              )
            })}
          </div>
        </div>
      </div>

      {selection && (
        <div className="selection-info">
          <p>Selección: posición {selection.start + 1} - {selection.end + 1} ({selection.end - selection.start + 1} bases)</p>
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
          <span className="legend-color selected"></span>
          <span>Selección</span>
        </div>
        <div className="legend-instructions">
          <p><strong>Instrucciones:</strong> 
            {isZoomedIn 
              ? "Usa ◀▶ para navegar • Arrastra para seleccionar • 🔍- para alejar"
              : isLargeSequence
                ? "Usa ◀▶ para navegar por la secuencia • 🔍+ para hacer zoom • Arrastra para seleccionar"
                : "🔍+ para hacer zoom • Arrastra para seleccionar una región"
            }
          </p>
          {isLargeSequence && (
            <p style={{ fontSize: '0.75rem', color: '#6c757d', marginTop: '0.25rem' }}>
              ⚡ Secuencia larga detectada: usando visualización optimizada para mejor rendimiento
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export default DNAViewer 