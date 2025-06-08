import React, { useState } from 'react'
import { predictionService } from '../services/predictionService'
import type { PredictionsResponse } from '../services/predictionService'
import './PredictionsPanel.css'

interface PredictionsPanelProps {
  patientDni: string
  patientName: string
}

const PredictionsPanel: React.FC<PredictionsPanelProps> = ({ patientDni, patientName }) => {
  const [predictions, setPredictions] = useState<PredictionsResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const handleGetPredictions = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const token = localStorage.getItem('token')
      if (!token) {
        throw new Error('No hay token de autenticación')
      }
      
      const data = await predictionService.getPredictionsForPatient(patientDni, token)
      setPredictions(data)
    } catch (err) {
      setError('Error al obtener las predicciones. Verifica que el servidor esté funcionando.')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const getConfidenceColor = (confidence: string): string => {
    switch (confidence.toLowerCase()) {
      case 'alta': return '#10b981' // green
      case 'media': return '#f59e0b' // yellow  
      case 'baja': return '#ef4444' // red
      default: return '#6b7280' // gray
    }
  }

  const getPredictionColor = (prediction: string): string => {
    return prediction === 'LOF' ? '#ef4444' : '#10b981'
  }

  return (
    <div className="predictions-panel">
      <div className="predictions-header">
        <h2>🧬 Predicciones de Variantes Genéticas</h2>
        <p>Análisis de variantes BRCA1 para {patientName}</p>
      </div>

      <button 
        className="predictions-button"
        onClick={handleGetPredictions}
        disabled={loading}
      >
        {loading ? '🔄 Cargando...' : '🚀 Obtener Predicciones'}
      </button>

      {error && (
        <div className="predictions-error">
          <p>❌ {error}</p>
        </div>
      )}

      {predictions && (
        <div className="predictions-results">
          <div className="patient-info">
            <h3>👤 {predictions.patient_info.name}</h3>
            <p>DNI: {predictions.patient_info.dni}</p>
          </div>

          <div className="predictions-info">
            <div className="info-item">
              <span className="info-label">Estado:</span>
              <span className="info-value">{predictions.status}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Modelos:</span>
              <span className="info-value">{predictions.total_models}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Muestra:</span>
              <span className="info-value">{predictions.sample_used}</span>
            </div>
          </div>

          <div className="predictions-grid">
            {Object.entries(predictions.predictions).map(([modelName, prediction]) => (
              <div key={modelName} className="prediction-card">
                <div className="card-header">
                  <h3>{modelName}</h3>
                  <span 
                    className="prediction-badge"
                    style={{ backgroundColor: getPredictionColor(prediction.prediction) }}
                  >
                    {prediction.prediction}
                  </span>
                </div>
                
                <div className="card-content">
                  <div className="probability-row">
                    <span>Probabilidad:</span>
                    <span className="probability-value">
                      {(prediction.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="confidence-row">
                    <span>Confianza:</span>
                    <span 
                      className="confidence-value"
                      style={{ color: getConfidenceColor(prediction.confidence) }}
                    >
                      {prediction.confidence}
                    </span>
                  </div>
                  
                  <p className="description">{prediction.description}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="predictions-description">
            <p>{predictions.description}</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default PredictionsPanel 