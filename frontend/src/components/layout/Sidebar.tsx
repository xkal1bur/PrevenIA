import './Sidebar.css';

const Sidebar = () => (
  <aside className="sidebar">
    <div className="sidebar__logo">
      <span role="img" aria-label="ribbon">🎀</span>
      <span className="sidebar__brand">PREVENIA</span>
    </div>
    <nav className="sidebar__nav">
      <a href="#" className="sidebar__link"><span role="img" aria-label="home">🏠</span> Home</a>
      <a href="#" className="sidebar__link sidebar__link--active"><span role="img" aria-label="patients">👥</span> Pacientes</a>
      <a href="#" className="sidebar__link"><span role="img" aria-label="feedback">💬</span> Feedback</a>
    </nav>
    <div className="sidebar__logout">
      <span role="img" aria-label="logout">📤</span> Cerrar sesión
    </div>
  </aside>
);

export default Sidebar; 