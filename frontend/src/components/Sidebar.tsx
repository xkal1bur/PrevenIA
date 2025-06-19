import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  FiHome,
  FiUsers,
  FiLogOut,
  FiMenu,
  FiX,
} from 'react-icons/fi'
import './Sidebar.css'

interface SidebarProps {
  onLogout: () => void
}

const Sidebar: React.FC<SidebarProps> = ({ onLogout }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false)

  const toggleSidebar = (): void => {
    setIsOpen((prev) => !prev)
  }

  const handleLinkClick = (): void => {
    if (isOpen) setIsOpen(false)
  }

  return (
    <>
      <button className="hamburger-btn" onClick={toggleSidebar}>
        {isOpen ? <FiX size={24} /> : <FiMenu size={24} />}
      </button>

      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <img
            src="/images/Logo.png"
            alt="Logo Prevenia"
            className="sidebar-logo"
          />
        </div>

        <nav className="sidebar-nav">
          <NavLink
            to="/home"
            className="sidebar-link"
            onClick={handleLinkClick}
          >
            <FiHome className="sidebar-icon" />
            <span className="link-text">Home</span>
          </NavLink>

          <NavLink
            to="/pacientes"
            className="sidebar-link"
            onClick={handleLinkClick}
          >
            <FiUsers className="sidebar-icon" />
            <span className="link-text">Pacientes</span>
          </NavLink>
        </nav>

        <div className="sidebar-spacer" />

        <button className="sidebar-logout" onClick={onLogout}>
          <FiLogOut className="sidebar-icon-logout" />
          <span className="link-text">Cerrar sesión</span>
        </button>
      </aside>
    </>
  )
}

export default Sidebar
