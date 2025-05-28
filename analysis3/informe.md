# PrevenIA: AI para Diagnóstico de Cáncer de Mama

**Equipo:** Arturo Barrantes, Francisco Calle, José Carlos de la Cruz, Leonardo Huaman

## Abstract

Este informe presenta los resultados de la evaluación exhaustiva de modelos de AI para la predicción de variantes de nucleótido único (SNVs) en los genes BRCA1 y BRCA2, componentes críticos en el diagnóstico temprano del cáncer de mama. Tras analizar cuatro modelos de vanguardia en genómica computacional, PrevenIA ha identificado Evo2 40B como la solución óptima para la implementación clínica.

## Objetivo Principal

Seleccionar el modelo de IA más eficaz para la predicción de SNVs en genes BRCA1 y BRCA2, evaluando su rendimiento predictivo, capacidades técnicas y aplicabilidad clínica para el diagnóstico temprano del cáncer de mama.

## Metodología de Evaluación

La evaluación se centró en la capacidad predictiva de cada modelo para identificar variantes patogénicas versus benignas en los genes BRCA1 y BRCA2. Se utilizaron métricas estándar de la industria, incluyendo el Área Bajo la Curva ROC (AUROC) y el Área Bajo la Curva Precision-Recall (AUPR), para comparar el rendimiento de los modelos en partes del adn tanto codificantes como no codificantes.

## Modelos Evaluados

### Familia Evo (Modelos Principales)

**Evo2 40B** representa el modelo de IA más grande desarrollado para biología hasta la fecha, con 40 mil millones de parámetros y una ventana de contexto sin precedentes de 1 millón de nucleótidos. Este modelo fundacional genómico fue entrenado con 9.3 billones de pares de bases de ADN, abarcando más de 128,000 genomas completos de diversos dominios de la vida.

**Evo2 7B** ofrece una alternativa computacionalmente más eficiente con 7 mil millones de parámetros, manteniendo capacidades robustas de predicción zero-shot para función génica y generación de secuencias complejas.

**Evo2 1B** constituye la versión más compacta de la familia, con 1 mil millones de parámetros, diseñada para implementaciones con recursos computacionales limitados.

### Otros Modelos Comparados

**AlphaMissense** se especializa en la predicción de patogenicidad de variantes missense, integrando información evolutiva y estructural de proteínas. Utiliza la arquitectura Evoformer con 48 capas y aprovecha las predicciones estructurales de AlphaFold para contextualizar las variantes dentro de un marco evolutivo y estructural amplio.

**GPN-MSA (Genomic Pre-trained Network with Multiple Sequence Alignment)** representa un Language Model de ADN innovador que integra alineamientos de secuencias múltiples a través de diversas especies, utilizando una arquitectura Transformer flexible con ventana de contexto de 128 pares de bases.

**CADD (Combined Annotation Dependent Depletion)** emplea un enfoque de ML basado en regresión logística que integra más de 100 características derivadas de información genética, conservación de secuencias, actividad epigenética y reguladora, y otras anotaciones genómicas.

## Resultados de Rendimiento

### Evaluación en BRCA1

En la predicción de efectos de variantes BRCA1 mediante evaluación zero-shot, los modelos de la familia Evo2 demostraron rendimiento superior. BioNeMo Evo2 7B alcanzó un AUROC de 0.87, mientras que BioNeMo Evo2 1B obtuvo un AUROC de 0.76. El modelo Arc Evo2 1B registró un AUROC de 0.73. En pruebas más amplias con variantes BRCA1, Evo2 demostró una precisión superior al 90% en la predicción de mutaciones benignas versus potencialmente patogénicas.

### Contextos de Evaluación Específicos

Para SNVs codificantes en BRCA1 y BRCA2, que representan las variantes más frecuentemente asociadas con predisposición hereditaria al cáncer de mama y ovario, Evo2 40B y AlphaMissense mostraron rendimiento comparable y superior a otros modelos evaluados. La capacidad de Evo2 para mantener resolución de nucleótido único mientras procesa ventanas de contexto extensas resultó particularmente valiosa para la interpretación de variantes en regiones codificantes complejas.

En el análisis de SNVs no codificantes en BRCA1 y BRCA2, los modelos Evo2 destacaron significativamente sobre las alternativas. Esta capacidad es especialmente importante dado que las regiones regulatorias no codificantes pueden influir significativamente en la expresión génica y contribuir al riesgo de cáncer, aunque tradicionalmente han sido más difíciles de interpretar que las variantes codificantes.

## Análisis Comparativo de Características Técnicas

### Capacidades de Contexto

Evo2 40B se distingue por su ventana de contexto de 1 millón de nucleótidos, permitiendo la integración de información a lo largo de secuencias genómicas excepcionalmente largas. Esta capacidad es importante para comprender elementos reguladores distantes y mantener sensibilidad a cambios de nucleótido único (SNV).

AlphaMissense opera con una ventana de contexto de 768 pares de bases, optimizada específicamente para el análisis de variantes missense dentro del contexto estructural de proteínas (solo en regiones codificantes). GPN-MSA utiliza ventanas de 128 pares de bases con solapamiento de 64 pares de bases, diseñadas para aprovechar información de alineamientos de secuencias múltiples.

### Arquitecturas y Metodologías

La familia Evo2 emplea la arquitectura StripedHyena, que permite escalado casi lineal de cómputo y memoria en relación con la longitud del contexto, superando las limitaciones cuadráticas de las arquitecturas Transformer tradicionales. 

AlphaMissense integra la arquitectura Evoformer con embeddings estructurales de AlphaFold, combinando conocimientos co-evolutivos derivados de alineamientos de secuencias múltiples con información estructural tridimensional de proteínas.

### Estrategias de Entrenamiento

Evo2 fue entrenado mediante una estrategia sofisticada de dos fases, expandiendo progresivamente su ventana de contexto hasta 1 millón de pares de bases. El modelo demuestra aprendizaje autónomo de características biológicas complejas, incluyendo límites exón-intrón, sitios de unión de factores de transcripción y estructuras proteicas, sin requerir entrenamiento específico de variantes.

CADD utiliza un enfoque único de entrenamiento con variantes "proxy-neutrales" derivadas de la evolución humana post-especiación, contrastadas con variantes "proxy-deletéreas" simuladas, proporcionando un conjunto de datos de entrenamiento menos sesgado que otros métodos.

## Consideraciones de Implementación

### Recursos Computacionales

Evo2 40B requiere recursos computacionales sustanciales, optimizado para arquitectura NVIDIA Hopper con rendimiento eficiente en GPUs H100 y H200. El modelo utiliza precisión FP8 en algunas capas, requiriendo hardware con capacidad de cómputo igual o superior a 8.9.

### Escalabilidad y Eficiencia

Evo2 demuestra escalado lineal con sobrecarga mínima en entornos de computación distribuida con interconexiones rápidas. El modelo procesa aproximadamente 5 millones de variantes por hora utilizando 4 GPU NVIDIA A100, proporcionando throughput adecuado para aplicaciones clínicas de gran volumen.

## Limitaciones y Consideraciones

### Desafíos de Interpretabilidad

La arquitectura compleja de Evo introduce desafíos significativos en interpretabilidad comparado con modelos más simples. Esta característica de "black box" representa un obstáculo para la validación científica y la adopción clínica, donde los insights mecanicistos son frecuentemente requeridos para la confianza diagnóstica.

### Contexto Genómico Completo

Aunque Evo2 maneja hasta 1 millón de nucleótidos, esta capacidad aún no alcanza el contexto necesario para modelar cromosomas humanos completos, que pueden abarcar cientos de millones de nucleótidos.

### Consideraciones Éticas y Regulatorias

La implementación de modelos de IA genómica en entornos clínicos requiere consideración cuidadosa de aspectos relacionados con privacidad del paciente, consentimiento informado para análisis genómico comprehensivo, y cumplimiento con regulaciones de dispositivos médicos.

## Conclusión

Tras la evaluación exhaustiva de modelos de IA para predicción de SNVs en BRCA1 y BRCA2, PrevenIA ha seleccionado **Evo2 7B** como modelo principal. Esta decisión se fundamenta principalmente en su rendimiento superior en múltiples contextos genéticos, capacidad para manejar tanto variantes codificantes como no codificantes, se utilizará un distillation del bloque 20 para tener una representación del latent space y con esta poder crear una red neuronal que pueda funcionar como clasificador.

La implementación de Evo2 40B en el sistema PrevenIA permitirá análisis predictivo de alta precisión para variantes BRCA1/BRCA2, contribuyendo significativamente al diagnóstico temprano y la medicina de precisión en oncología mamaria. El modelo seleccionado representa el estado del arte en IA genómica y posiciona a PrevenIA como líder en aplicación clínica de tecnologías de inteligencia artificial para diagnóstico oncológico.
