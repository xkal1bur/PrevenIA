import React from 'react'
import { FiUser } from 'react-icons/fi'
import './Topbar.css'

interface TopbarProps {
  clinicName: string
  doctorName: string
}

const Topbar: React.FC<TopbarProps> = ({ clinicName, doctorName }) => {
  return (
    <header className="topbar">
      <h1 className="topbar-clinic">{clinicName}</h1>

      <div className="topbar-doctor-info">
        <span className="topbar-doctor-name">{`Dr. ${doctorName}`}</span>
        <FiUser className="topbar-doctor-icon" size={28} />
      </div>
    </header>
  )
}

export default Topbar