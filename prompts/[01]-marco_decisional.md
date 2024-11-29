Dada una sentencia de tutela, por favor analiza y proporciona la siguiente información básica y contextual, asegurando un análisis estructurado y organizado, en formato JSON con el siguiente esquema:
```json
{
  "identificacion": {
    "numero_sentencia": "",
    "magistrado_ponente": "",
    "sala_decision": "",
    "aclaraciones_voto": [
      {
        "magistrado": "",
        "motivo": ""
      }
    ],
    "salvamentos_voto": [
      {
        "magistrado": "",
        "argumento": ""
      }
    ]
  },
  "hechos_juridicamente_relevantes": [
    ""
  ],
  "problema_juridico_enunciado": "",
  "normas_juridicas_relevantes": [
    ""
  ],
  "decision_resuelve": ""
}
```

Notas:

- Algunos elementos pueden no estar presentes en todas las sentencias. Si un elemento no está presente, simplemente déjalo como un valor vacío o exclúyelo del JSON generado.
- Aclaraciones de Voto y Salvamentos de Voto podrían no existir si todos los magistrados estuvieron de acuerdo con la decisión y su justificación.
- El Problema Jurídico Enunciado por la Corte puede no estar claramente formulado en algunos casos, especialmente si no hubo una enunciación explícita del problema.

1. Identificación:

- Número de Sentencia: Indica el número de la sentencia que empieza con la letra T o SU (cuando se trata de una sentencia de unificación). Ejemplo: T-001/92.
- Magistrado Ponente: Proporciona el nombre del magistrado que elaboró la decisión y la propuso a la sala para su aprobación. Se identifica como "MP".
- Sala de Decisión: Describe si la decisión fue tomada por una "sala de decisión de tutela" conformada por tres magistrados, o si se trata de una sentencia de la Sala Plena (SU), la cual se encarga de decisiones de alta trascendencia nacional o que buscan unificar la jurisprudencia.
- Aclaraciones de Voto: Especifica los magistrados que estuvieron de acuerdo con la decisión pero difirieron en la justificación. Resume sus motivos.
- Salvamentos de Voto: Indica los magistrados que no estuvieron de acuerdo con la decisión. Resume sus argumentos.

2. Hechos Jurídicamente Relevantes (HJR):
- Proporciona un listado de los hechos que son importantes para la resolución del caso. Incluye solo los hechos que tienen un impacto directo en la decisión, asegurándote de que los hechos seleccionados sean concretos y objetivos para facilitar la comprensión de la problemática central del caso.

3. Problema Jurídico Enunciado por la Corte (PJC):
- Indica el problema jurídico que la Corte se plantea para guiar su análisis. Busca este problema formulado claramente en un acápice específico de la sentencia y transcríbelo o resúmelo. Este problema representa la cuestión principal que debe ser resuelta por la Corte.

4. Normas Jurídicas Relevantes para el Caso:
- Lista las normas jurídicas (constitucionales, legales, internacionales, reglamentarias, etc.) que son relevantes para resolver el caso.

5. Decisión (Resuelve):
- Indica la decisión final que tomó la Corte, el "resuelve". Si es corto, transcríbelo directamente; si no, proporciona un resumen."
