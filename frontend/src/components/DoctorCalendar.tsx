import React, { useState, useEffect } from 'react'
import Calendar from 'react-calendar'
import 'react-calendar/dist/Calendar.css'
import './DoctorCalendar.css'
import axios from 'axios'

export interface Appointment {
  id: number
  fecha_hora: string
  asunto: string
  lugar: string
  descripcion?: string
}

const DoctorCalendar: React.FC = () => {
  const [selectedDate, setSelectedDate] = useState<Date>(new Date())
  const [events, setEvents] = useState<Record<string, Appointment[]>>({})
  const [loading, setLoading] = useState(false)

  const token = localStorage.getItem('token') || ''

  // Carga las citas del día
  const loadDay = async (date: Date) => {
    const isoDay = date.toISOString().split('T')[0]
    setLoading(true)
    try {
      const res = await axios.get<Appointment[]>(
        `http://52.1.220.84:8000/calendario/dia/${isoDay}`,
        { headers: { Authorization: `Bearer ${token}` } }
      )
      setEvents(prev => ({ ...prev, [isoDay]: res.data }))
    } catch {
      setEvents(prev => ({ ...prev, [isoDay]: [] }))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDay(selectedDate)
  }, [selectedDate])

  const dayKey = selectedDate.toISOString().split('T')[0]
  const todays = events[dayKey] || []

  // Estado del formulario
  const [newTime, setNewTime] = useState('09:00')
  const [newAsunto, setNewAsunto] = useState('')
  const [newLugar, setNewLugar] = useState('')
  const [newDescripcion, setNewDescripcion] = useState('')

  const handleCreate = async () => {
    try {
      await axios.post(
        'http://52.1.220.84:8000/calendario/evento',
        {
          fecha_hora: `${dayKey}T${newTime}`,
          asunto: newAsunto,
          lugar: newLugar,
          descripcion: newDescripcion,
        },
        { headers: { Authorization: `Bearer ${token}` } }
      )
      setNewAsunto(''); setNewLugar(''); setNewDescripcion('')
      loadDay(selectedDate)
    } catch {
      alert('Error al crear la cita')
    }
  }

  const handleDelete = async (id: number) => {
    if (!window.confirm('¿Eliminar esta cita?')) return
    try {
      await axios.delete(`http://52.1.220.84:8000/calendario/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      loadDay(selectedDate)
    } catch {
      alert('Error al eliminar la cita')
    }
  }

  return (
    <div className="calendar-container">
      <h3>Calendario de Citas</h3>
      <div className="calendar-grid">
        {/* 1. CALENDARIO */}
        <div className="calendar-box">
          <Calendar
            onChange={d => setSelectedDate(d as Date)}
            value={selectedDate}
            tileClassName={({ date }) => {
              const iso = date.toISOString().split('T')[0]
              return events[iso]?.length ? 'has-event' : ''
            }}
          />
        </div>

        {/* 2. LISTA DE CITAS */}
        <div className="list-box">
          <h4>Citas del {selectedDate.toDateString()}</h4>
          {loading
            ? <p className="no-event">Cargando…</p>
            : todays.length
              ? todays.map(ev => (
                <div key={ev.id} className="evento-item">
                  <div className="evento-info">
                    <strong>{ev.asunto}</strong>
                    <p>{new Date(ev.fecha_hora).toLocaleTimeString()}</p>
                    <p>Lugar: {ev.lugar}</p>
                    {ev.descripcion && <p>{ev.descripcion}</p>}
                  </div>
                  <button
                    className="delete-btn"
                    onClick={() => handleDelete(ev.id)}
                  >🗑️</button>
                </div>
              ))
              : <p className="no-event">No hay citas para este día</p>
          }
        </div>

        {/* 3. FORMULARIO AGREGAR */}
        <div className="form-box">
          <h4>+ Agregar cita</h4>
          <input
            type="time"
            value={newTime}
            onChange={e => setNewTime(e.target.value)}
          />
          <input
            type="text"
            placeholder="Asunto"
            value={newAsunto}
            onChange={e => setNewAsunto(e.target.value)}
          />
          <input
            type="text"
            placeholder="Lugar"
            value={newLugar}
            onChange={e => setNewLugar(e.target.value)}
          />
          <textarea
            placeholder="Descripción (opcional)"
            value={newDescripcion}
            onChange={e => setNewDescripcion(e.target.value)}
          />
          <button
            onClick={handleCreate}
            disabled={!newAsunto || !newLugar}
          >
            Guardar
          </button>
        </div>
      </div>
    </div>
  )
}

export default DoctorCalendar
