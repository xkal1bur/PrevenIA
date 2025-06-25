// Configuración de la API
export const API_CONFIG = {
  // En desarrollo, usar el proxy de Vite
  BASE_URL: import.meta.env.DEV ? '/api' : 'http://localhost:8000',
  
  // Endpoints específicos
  ENDPOINTS: {
    // Doctores
    REGISTER_DOCTOR: '/register/doctor',
    LOGIN_DOCTOR: '/login/doctor',
    DOCTOR_ME: '/doctors/me',
    
    // Pacientes
    REGISTER_PACIENTE: '/register/paciente',
    PACIENTES: '/pacientes',
    PACIENTES_BY_DOCTOR: (doctorId: number) => `/pacientes/doctor/${doctorId}`,
    PACIENTE_BY_DNI: (dni: string) => `/pacientes/dni/${dni}`,
    
    // Archivos
    UPLOAD_FASTA: (dni: string) => `/pacientes/${dni}/upload_fasta`,
    PACIENTE_FILES: (dni: string) => `/pacientes/${dni}/files`,
    DOWNLOAD_FILE: (dni: string, filename: string) => `/pacientes/${dni}/files/${filename}`,
    DELETE_FILE: (dni: string, filename: string) => `/pacientes/${dni}/files/${filename}`,
    DELETE_FOLDER: (dni: string, folderName: string) => `/pacientes/${dni}/folders/${folderName}`,
    
    // Procesamiento
    ALIGN_WITH_CR13: (dni: string) => `/pacientes/${dni}/align_with_cr13`,
    BLAST_SIMPLE: (dni: string) => `/pacientes/${dni}/blast_simple`,
    PROCESS_EMBEDDING: (dni: string) => `/pacientes/${dni}/process_embedding`,
    
    // Predicciones
    PREDICTIONS: (dni: string) => `/predictions/${dni}`,
    
    // Chunks
    PATIENT_CHUNKS_INFO: (dni: string) => `/pacientes/${dni}/patient_chunks/info`,
    ALIGNED_CHUNKS_INFO: (dni: string) => `/pacientes/${dni}/aligned_chunks/info`,
    
    // Notas
    NOTES: (dni: string) => `/pacientes/${dni}/notes`,
    
    // Estadísticas
    MONTHLY_NEW_PATIENTS: '/pacientes/stats/monthly_new',
    ACTIVE_INACTIVE: '/pacientes/stats/active_inactive',
    BUCKET_USAGE: '/stats/bucket_usage',
    
    // Calendario
    CALENDAR_DAY: (fecha: string) => `/calendario/dia/${fecha}`,
    CALENDAR_EVENT: '/calendario/evento',
    
    // Archivos estáticos
    STATIC_PACIENTES: (filename: string) => `/static/pacientes/${filename}`
  }
}

// Helper para construir URLs completas
export const buildApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`
}

// Helper para headers con autenticación
export const getAuthHeaders = (token?: string) => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  }
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  return headers
}

// Helper para headers de form data con autenticación
export const getAuthFormHeaders = (token?: string) => {
  const headers: Record<string, string> = {}
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  return headers
} 