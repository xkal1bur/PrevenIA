import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer
} from 'recharts'

interface StatsResponse {
  year: number
  monthly_new: number[]
}

interface YearsResponse {
  years: number[]
}

export default function MonthlyNewChart() {
  const [years, setYears] = useState<number[]>([])
  const [year, setYear]   = useState<number>(new Date().getFullYear())
  const [data, setData]   = useState<{ month: string; count: number }[]>([])

  // 1) Cargar años disponibles
  useEffect(() => {
    const token = localStorage.getItem('token')
    axios
      .get<YearsResponse>(
        'http://localhost:8000/pacientes/stats/years',
        { headers: { Authorization: `Bearer ${token}` } }
      )
      .then(res => {
        const yrs = res.data.years
        setYears(yrs)
        // si el año actual no está, selecciona el último disponible
        if (!yrs.includes(year) && yrs.length) {
          setYear(yrs[yrs.length - 1])
        }
      })
      .catch(console.error)
  }, [])

  // 2) Cada vez que cambie el año, cargar datos
  useEffect(() => {
    if (!year) return
    const token = localStorage.getItem('token')
    axios
      .get<StatsResponse>(
        `http://localhost:8000/pacientes/stats/monthly_new?year=${year}`,
        { headers: { Authorization: `Bearer ${token}` } }
      )
      .then(res => {
        const meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        const chartData = res.data.monthly_new.map((c, i) => ({
          month: meses[i],
          count: c
        }))
        setData(chartData)
      })
      .catch(console.error)
  }, [year])

  return (
    <div style={{ background: 'white', padding: '1rem', borderRadius: 8, margin: '1.5rem 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <h3 style={{ margin: 0 }}>Pacientes nuevos por mes</h3>
        <select value={year} onChange={e => setYear(Number(e.target.value))}>
          {years.map(y => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="#3b82f6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}