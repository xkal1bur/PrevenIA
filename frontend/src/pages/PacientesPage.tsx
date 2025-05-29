import Sidebar from '../components/layout/Sidebar';
import Header from '../components/layout/Header';
import PatientGrid from '../components/ui/PatientGrid';
import './PacientesPage.css';

const PacientesPage = () => (
  <div className="layout">
    <Sidebar />
    <div className="main-content">
      <Header />
      <PatientGrid />
    </div>
  </div>
);

export default PacientesPage; 