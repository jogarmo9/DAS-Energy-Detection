# DAS Energy Detection System

Este proyecto implementa un pipeline robusto para la detección automática de eventos en datos de Detección Acústica Distribuida (DAS) basado en el análisis de energía espacio-temporal.

## 🚀 Arquitectura del Proyecto

El sistema está diseñado de forma modular para separar el preprocesado, la lógica de detección y la orquestación.

```text
.
├── config.yaml               # Configuración centralizada de parámetros
├── main_detect_events.py     # Script principal (orquestador)
├── resultados.ipynb          # Visualización avanzada y validación
├── preprocessing/
│   └── das_preprocess.py     # Filtros, normalización y mapas de energía
├── detection/
│   └── event_detector.py     # Algoritmo de extracción de componentes conexas 2D
├── data/                     # Carpeta de entrada (.npz originales)
└── outputs/                  # Resultados organizados por experimento
```

## 🛠 Componentes Principales

### 1. Preprocesado (`preprocessing/das_preprocess.py`)
Aplica un pipeline de señales unificado diseñado para maximizar la relación señal-ruido (SNR) antes de la detección:
- **Lectura y Downsampling**: Soporte nativo para archivos `.npz` con capacidad de reducción de tasa de muestreo (`decimate`) para optimizar el procesamiento en archivos de gran tamaño.
- **Common-mode Removal (CMR)**: Eliminación de ruido instrumental coherente (ruido de procesador, vibraciones de gabinete) mediante la resta de la mediana espacial en cada instante de tiempo.
- **Filtro Pasabanda SOS**: Uso de *Second-Order Sections* para un filtrado estable en rangos de frecuencia específicos (ej. 1-20Hz), eliminando el drift de baja frecuencia y el aliasing de alta.
- **Suavizado Gaussiano 2D**: Aplicación de un kernel Gaussiano en el dominio tiempo-espacio para eliminar el ruido granular (speckle) y conectar visualmente trazas de eventos débiles.
- **Scaling Robusto**: Normalización de los datos entre [-1, 1] basada en percentiles (clipping), lo que evita que picos de ruido aislados dominen la escala de la señal.
- **Mapa de Energía Z-score (Robust)**: Cálculo de energía local normalizada utilizando la Mediana y la Desviación Absoluta de la Mediana (MAD). Esto asegura que el "ruido base" no se vea sesgado por la presencia de eventos de gran magnitud.

### 2. Detector de Eventos (`detection/event_detector.py`)
Utiliza la clase orquestadora `DASEventDetector` para transformar mapas de energía en eventos físicos discretos:
- **Umbralización Adaptativa**: Generación de máscaras binarias donde la energía supera un factor `N` veces el ruido base (MAD).
- **Análisis de Componentes Conexas 2D**: Algoritmo de visión artificial (`scipy.ndimage.label`) para agrupar píxeles de energía adyacentes en estructuras únicas. Esto permite detectar eventos que se mueven o se expanden en el tiempo y el espacio simultáneamente.
- **Filtros de Coherencia Física**:
    - **Temporal**: Descarta detecciones cuya duración sea inferior a `min_duration_sec`.
    - **Espacial**: Excluye ruidos que afectan a menos de `min_sensors`, eliminando falsos positivos causados por fallos en canales individuales.
- **Caracterización de Eventos**: Cada detección genera un diccionario con metadatos precisos: tiempo exacto de inicio/fin, sensores afectados, energía promedio y pico, ideales para su posterior exportación a CSV o bases de datos.

### 3. Orquestador (`main_detect_events.py`)
Automatiza el procesamiento por lotes:
- Lee archivos `.npz` de la carpeta `data/`.
- Aplica downsampling si es necesario.
- Organiza las salidas en subcarpetas dentro de `outputs/`.

## 📁 Estructura de Salida (`outputs/`)

El sistema genera una estructura organizada para facilitar tanto el análisis manual como el entrenamiento futuro de modelos:

```text
outputs/
├── all_events.pkl             # Resumen global de todos los archivos procesados
└── [nombre_del_archivo]/      # Carpeta específica por cada archivo de entrada
    ├── energy.npy             # Mapa de energía 2D (Z-score Robusto)
    ├── mask.npy               # Máscaras binarias de detección (True/False)
    └── events.pkl             # Lista de diccionarios con metadatos de los eventos
```

### Descripción de Archivos:
- **`energy.npy`**: Matriz NumPy `(Tiempo x Sensores)` que contiene los valores de energía normalizados. Es el dato principal para visualizar la intensidad de los eventos.
- **`mask.npy`**: Matriz binaria de las mismas dimensiones que la energía. Indica los "píxeles" que superaron el umbral y pasaron los filtros de coherencia.
- **`events.pkl`**: Contiene la información estructurada de cada evento detectado:
    - `t_start`, `t_end`: Tiempo exacto en segundos.
    - `sensor_start`, `sensor_end`: Rango de sensores afectados.
    - `mean_energy`, `max_energy`: Estadísticas de intensidad.
    - `duration_sec`, `n_sensors`: Métricas de duración y extensión espacial.
- **`all_events.pkl`**: Un consolidado de todos los eventos encontrados en la sesión actual, ideal para generar estadísticas globales o reportes CSV.

## ⚙ Configuración (`config.yaml`)

Puedes ajustar el comportamiento del sistema sin tocar el código:
- `event_threshold`: Sensibilidad de la detección.
- `min_event_sensors`: Mínimo de sensores afectados para considerar un evento (limpia ruido puntual).
- `min_event_duration_sec`: Duración mínima del evento.
- `fmin` / `fmax`: Rango de frecuencias de interés.

## 📊 Visualización (`resultados.ipynb`)

El notebook permite validar los resultados cargando:
- El **dato original** (.npz) con paleta `viridis`.
- **Bounding Boxes**: Rectángulos rojos sobre el dato crudo que marcan exactamente dónde el algoritmo detectó actividad.
- **Mapa de Energía**: Visualización de la intensidad Z-score.
- **Tabla de Eventos**: Resumen detallado con tiempos de inicio/fin y sensores afectados.

## 📋 Requisitos e Instalación

1. Instalar dependencias:
   ```bash
   pip install numpy scipy matplotlib pyyaml joblib pandas
   ```
2. Colocar los archivos `.npz` en la carpeta `data/`.
3. Ejecutar la detección:
   ```bash
   python main_detect_events.py
   ```
4. Ver resultados en `resultados.ipynb`.

## 🎯 Guía de Afinación (Tuning Guide)

Si la detección no es perfecta, ajusta el archivo `config.yaml` siguiendo estas reglas:

| Problema | Solución Recomendada | ¿Por qué? |
| :--- | :--- | :--- |
| **Mucho ruido punctual** | Subir `min_sensors` (ej. 10) | El ruido rara vez afecta a muchos sensores a la vez. |
| **Falsos positivos (speckle)** | Subir `sigma_2d` (ej. 1.5 - 2.0) | "Difumina" el ruido antes de que el detector lo vea. |
| **Evento real no detectado** | Bajar `threshold` (ej. 2.5) | Permite que señales más débiles superen el umbral. |
| **Evento se corta en trozos** | Subir `smooth_sec` (ej. 1.0) | Une las partes de un evento que fluctúa en intensidad. |
| **Ruido de baja frecuencia** | Subir `fmin` (ej. 10.0) | Corta ruidos de motores o vibraciones ambientales lentas. |

> **Tip**: Usa siempre `resultados.ipynb` para ver el efecto de los cambios. Si ves que en el "Mapa de Energía" el evento es evidente pero no tiene rectángulo rojo, baja el `threshold` o la `min_duration_sec`.
