import './Header.css';

const Header = () => (
  <header className="header">
    <div className="header__title">NOMBRE DE LA CLINICA</div>
    <div className="header__doctor">
      <span>Nombre del doctor</span>
      <span className="header__avatar" role="img" aria-label="doctor">👤</span>
    </div>
  </header>
);

export default Header; 