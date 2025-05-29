import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import PacientesPage from "./pages/PacientesPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: App,
    children: [
      {
        index: true,
        Component: App, // Change later:)
      },
      {
        path: "/pacientes",
        Component: PacientesPage,
      },
    ],
  },
]);
