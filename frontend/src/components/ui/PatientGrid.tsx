import PatientCard from './PatientCard';
import './PatientGrid.css';

const patients = [
  {
    name: 'Emily Vasquez',
    code: '20250000',
    age: 27,
    phone: '987654321',
    photo: 'https://randomuser.me/api/portraits/women/44.jpg',
  },
  {
    name: 'Ana Cristobal',
    code: '20250001',
    age: 35,
    phone: '987654322',
    photo: 'https://randomuser.me/api/portraits/women/45.jpg',
  },
  {
    name: 'Sofia Burgos',
    code: '20250002',
    age: 47,
    phone: '987654323',
    photo: 'https://randomuser.me/api/portraits/women/46.jpg',
  },
  // Repeat for demo
  {
    name: 'Emily Vasquez',
    code: '20250000',
    age: 27,
    phone: '987654321',
    photo: 'https://randomuser.me/api/portraits/women/44.jpg',
  },
  {
    name: 'Ana Cristobal',
    code: '20250001',
    age: 35,
    phone: '987654322',
    photo: 'https://randomuser.me/api/portraits/women/45.jpg',
  },
  {
    name: 'Sofia Burgos',
    code: '20250002',
    age: 47,
    phone: '987654323',
    photo: 'https://randomuser.me/api/portraits/women/46.jpg',
  },
];

const PatientGrid = () => (
  <div className="patient-grid">
    {patients.map((p, i) => (
      <PatientCard key={i} {...p} />
    ))}
  </div>
);

export default PatientGrid; 