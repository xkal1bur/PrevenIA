import './PatientCard.css';

type Patient = {
  name: string;
  code: string;
  age: number;
  phone: string;
  photo: string;
};

const PatientCard = ({ name, code, age, phone, photo }: Patient) => (
  <div className="patient-card">
    <img src={photo} alt={name} className="patient-card__photo" />
    <div className="patient-card__info">
      <div className="patient-card__name">{name}</div>
      <div className="patient-card__details">Código: {code}</div>
      <div className="patient-card__details">Edad: {age} años</div>
      <div className="patient-card__details">Celular: {phone}</div>
    </div>
  </div>
);

export default PatientCard; 