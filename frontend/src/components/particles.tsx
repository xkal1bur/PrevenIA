import React, { useEffect, useMemo } from "react"
  import Particles, { initParticlesEngine } from "@tsparticles/react"
  import { loadSlim } from "@tsparticles/slim"

  interface ParticlesComponentProps {
    id: string
  }

  const ParticlesComponentInner: React.FC<ParticlesComponentProps> = (props) => {
    // Iniciamos el motor una sola vez:
    useEffect(() => {
      initParticlesEngine(async (engine: any) => {
        await loadSlim(engine)
      })
    }, [])

    const options = useMemo(
      () => ({
        background: { color: { value: "#326071" } },
        fpsLimit: 120,
        interactivity: {
          events: {
            onClick: { enable: true, mode: "repulse" as const },
            onHover: { enable: true, mode: "grab" as const },
          },
          modes: { push: { distance: 200, duration: 15 }, grab: { distance: 150 } },
        },
        particles: {
          color: { value: "#FFFFFF" },
          links: { color: "#FFFFFF", distance: 150, enable: true, opacity: 0.3, width: 1 },
          move: {
            direction: "none" as const,
            enable: true,
            outModes: { default: "bounce" as const },
            random: true,
            speed: 1,
            straight: false,
          },
          number: { density: { enable: true }, value: 150 },
          opacity: { value: 1.0 },
          shape: { type: "circle" as const },
          size: { value: { min: 1, max: 3 } },
        },
        detectRetina: true,
      }),
      []
    )

    return <Particles id={props.id} options={options} />
  }

  // React.memo evitará re-render si 'id' no cambia
  const ParticlesComponent = React.memo(ParticlesComponentInner)

  export default ParticlesComponent