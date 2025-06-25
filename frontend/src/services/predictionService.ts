import { API_CONFIG, buildApiUrl, getAuthFormHeaders } from '../config/api'

export interface Prediction {
  prediction: string
  probability: number
  confidence: string
  description: string
  model_performance?: string
}

export interface PatientInfo {
  dni: string
  name: string
}

export interface ScenarioInfo {
  scenario_name: string
  risk_level: string
  clinical_significance: string
  consensus: string
  consensus_confidence: string
}

export interface AnalysisSummary {
  models_predicting_pathogenic: number
  models_predicting_benign: number
  average_probability: number
  prediction_agreement: string
}

export interface PredictionsResponse {
  status: string
  total_models: number
  patient_info: PatientInfo
  sample_used: string
  scenario_info?: ScenarioInfo
  analysis_summary?: AnalysisSummary
  clinical_recommendations?: string[]
  predictions: Record<string, Prediction>
  interpretation?: string
  description: string
}

export const predictionService = {
  async getPredictionsWithEmbedding(dni: string, embeddingFilename: string, token: string): Promise<PredictionsResponse> {
    try {
      const formData = new FormData()
      formData.append('embedding_filename', embeddingFilename)

      const url = buildApiUrl(API_CONFIG.ENDPOINTS.PREDICTIONS(dni))
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data: PredictionsResponse = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching predictions with embedding:', error)
      throw error
    }
  },

  async getPredictionsForPatient(dni: string, token: string): Promise<PredictionsResponse> {
    try {
      const url = buildApiUrl(API_CONFIG.ENDPOINTS.PREDICTIONS(dni))
      const response = await fetch(url, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data: PredictionsResponse = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching predictions:', error)
      throw error
    }
  },

  async processMismatchEmbedding(
    dni: string,
    filename: string,
    token: string
  ): Promise<Record<string, unknown>> {
    try {
      const formData = new FormData()
      formData.append('filename', filename)

      const url = buildApiUrl(API_CONFIG.ENDPOINTS.PROCESS_EMBEDDING(dni))
      const response = await fetch(url, {
        method: 'POST',
        headers: getAuthFormHeaders(token),
        body: formData
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Error processing embeddings:', error)
      throw error
    }
  }
} 