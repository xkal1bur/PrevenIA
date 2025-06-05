const API_BASE_URL = 'http://localhost:8000'

export interface Prediction {
  prediction: string
  probability: number
  confidence: string
  description: string
}

export interface PatientInfo {
  dni: string
  name: string
}

export interface PredictionsResponse {
  status: string
  total_models: number
  patient_info: PatientInfo
  sample_used: string
  predictions: Record<string, Prediction>
  description: string
}

export const predictionService = {
  async getPredictionsForPatient(dni: string, token: string): Promise<PredictionsResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/predictions/${dni}`, {
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
  }
} 