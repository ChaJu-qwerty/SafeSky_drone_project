# SafeSky — Drone Swarm Simulator & Real-Time Vision System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![ROS2](https://img.shields.io/badge/ROS2-Jazzy-22314E?style=for-the-badge&logo=ros&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge)
![PX4](https://img.shields.io/badge/PX4-SITL-5C2D91?style=for-the-badge)
![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-F58025?style=for-the-badge)

**· Diseño y desarrollo de robots ·**

**· Tecnológico de Monterrey · 2026 ·**

*Simulación de enjambres de UAVs con dinámica 6-DOF, navegación híbrida y visión computacional en tiempo real*

</div>

---

## 📋 Tabla de Contenidos

- [Descripción General](#descripcion-general)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [SafeSky — Simulador Python](#safesky-simulador-python)
- [Sistema de Visión YOLOv8](#sistema-de-vision-yolov8)
- [Migración a Gazebo + ROS2 + PX4](#migracion-gazebo)
- [Estructura del Repositorio](#estructura-repositorio)
- [Instalación y Uso](#instalacion-y-uso)
- [Resultados](#resultados)
- [Referencias](#referencias)

---

## 🚁 Descripción General <a name="descripcion-general"></a>

**SafeSky** es un entorno de simulación avanzado para la evaluación de algoritmos de **navegación, control y evasión de colisiones** en enjambres de vehículos aéreos no tripulados (UAVs). El sistema fue construido y validado en dos etapas:

1. **Simulador Python** — Simulación 6-DOF del cuadricóptero DJI F450 con controladores PID en cascada, planificación híbrida (Hybrid A* + ORCA) y perturbaciones de viento estocástico (AR(1)).
2. **Sistema de Visión** — Modelo YOLOv8m entrenado para detección en tiempo real de drones y personas, integrado con el dron físico DJI Tello.

Ambos sistemas fueron posteriormente validados en **Gazebo Harmonic con PX4 SITL bajo ROS2 Jazzy**, demostrando portabilidad a entornos de mayor fidelidad física.

---

## 🏗️ Arquitectura del Sistema <a name="arquitectura-del-sistema"></a>

```
┌─────────────────────────────────────────────────────────────┐
│                     ENTRADAS                                │
│   Coordenadas de meta · Estados iniciales · Mapa del entorno│
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     Planificador de Ruta   │
         │   Hybrid A*  ·  ORCA       │
         └─────────────┬─────────────┘
                       │
         ┌─────────────▼─────────────┐
         │   Controlador de Vuelo PID │
         │   Lazo Posición · Actitud  │
         └─────────────┬─────────────┘
                       │        ↑ Perturbación: Viento AR(1)
         ┌─────────────▼─────────────┐
         │   Dinámica 6-DOF           │
         │   Modelo DJI F450          │
         └─────────────┬─────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     SALIDAS                                 │
│      Estado actual del dron · Indicadores de desempeño (KPI)│
└─────────────────────────────────────────────────────────────┘
```

---

## 🖥️ SafeSky — Simulador Python <a name="safesky-simulador-python"></a>

### Modelo Físico (DJI F450)

El simulador implementa la dinámica completa de **6 grados de libertad** basada en Luukkonen (2011):

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `m` | 1.0 kg | Masa total (chasis + motores + batería) |
| `l` | 0.225 m | Longitud del brazo |
| `Ixx = Iyy` | 7.75×10⁻³ kg·m² | Inercia roll/pitch |
| `Izz` | 13.6×10⁻³ kg·m² | Inercia yaw |
| `kf` | 2.98×10⁻⁶ N/(rad/s)² | Coeficiente de empuje |
| `kb` | 1.14×10⁻⁷ N·m/(rad/s)² | Coeficiente de par |

La integración numérica utiliza **Runge-Kutta de 4° orden** con `dt = 0.05 s`.

### Controlador PID en Cascada

Ganancias sintonizadas con **Ziegler-Nichols + desintonización conservadora**:

| Lazo | Kp | Ki | Kd |
|------|----|----|-----|
| Altitud (Z) | 5.0 | 0.12 | 2.8 |
| Posición (XY) | 1.0 | 0.03 | 1.4 |
| Actitud (φ, θ) | 6.0 | 0.04 | 2.2 |
| Actitud (ψ) | 3.0 | 0.02 | 0.9 |

### Planificación Híbrida

- **Hybrid A\*** — Planificador global con resolución de grilla `0.40 m`, suavizado de trayectoria y margen de seguridad `D_SAFE = 0.65 m` a obstáculos.
- **ORCA** — Planificador local para evasión recíproca entre agentes con `tau = 2.0 s` y radio físico `r = 0.495 m`.

### Viento Estocástico AR(1)

```
w[k+1] = α·w[k] + η[k],   η ~ N(0, σ²I₃)
F_wind = 0.5 · ρ · Cd_A · |w| · w
```

Parámetros de simulación: `α = 0.85`, `σ = 0.60` (viento urbano moderado).

### Subsistemas (`safesky_v7_final/`)

| Archivo | Función |
|---------|---------|
| `safesky_main_7.py` | Motor principal, GUI (tkinter + matplotlib), bucle de simulación |
| `physics.py` | Dinámica 6-DOF, controlador PID cascada, viento AR(1) |
| `planner.py` | Hybrid A*, ORCA, modelo de obstáculos elipsoidales |
| `kpis.py` | Rastreo de KPIs en tiempo real |

---

## 👁️ Sistema de Visión YOLOv8 <a name="sistema-de-vision-yolov8"></a>

### Pipeline de Procesamiento

```
Cámara del dron
      │
      ▼
Transmisión de video (WiFi)
      │
      ▼
Captura de frames (DJITelloPy)
      │
      ▼
Preprocesamiento de imagen (OpenCV)
      │
      ▼
Inferencia del modelo YOLO
      │
      ▼
Visualización y generación de alertas
```

### Dataset y Entrenamiento

El modelo fue entrenado con **3 datasets combinados** de Kaggle:

- *Drone Dataset (UAV)* — imágenes stock de UAVs reales
- *Drone Detection* — variedad de entornos y condiciones de luz
- *Person Detection UAV (PASCAL_VOC_UAQ_MSUAV)* — personas vistas desde dron

**División del dataset:** 70% entrenamiento · 20% validación · 10% prueba  
**Data Augmentation:** rotación en eje Y, ruido Gaussiano, combinaciones.

**Configuración:** 50 épocas · batch size 16 · modelo base YOLOv8m

### Métricas del Modelo Entrenado

| Métrica | Valor |
|---------|-------|
| **mAP@50** | **90.77%** |
| **mAP@50-95** | **61.25%** |
| Precisión | 85.59% |
| Recall | 87.19% |
| Box Loss | 0.9546 |
| Classification Loss | 0.6710 |

### Estructura del proyecto de visión (`Vision project/`)

```
Vision project/
├── drone_person_detection_fixed.ipynb   # Entrenamiento del modelo
├── drone_person_laptop.ipynb            # Pruebas en laptop/webcam
├── drone_project/                       # Integración con DJI Tello
│   ├── tello_detection.py               # Control + detección en tiempo real
│   └── ...
├── training/                            # Pesos y logs de entrenamiento
│   └── ...
├── runs/                                # Resultados de experimentos
├── yolov8m.pt                           # Modelo base YOLOv8m
└── yolo26n.pt                           # Modelo ligero alternativo
```

---

## 🌐 Migración a Gazebo + ROS2 + PX4 <a name="migracion-gazebo"></a>

El sistema fue validado en un entorno de mayor fidelidad física:

- **Plataforma:** Gazebo Harmonic + PX4 SITL + ROS2 Jazzy
- **Dron:** Modelo X500
- **Controlador:** Offboard PX4 con setpoints de velocidad calculados por ORCA
- **Resultado:** Misión A→B (15 m) completada en ~18 s, distancia mínima a obstáculos de 4.5 m, error de posición final de 0.0 m

Para lanzar la simulación en Gazebo:

```bash
chmod +x launch_safesky.sh
./launch_safesky.sh
```

---

## 📁 Estructura del Repositorio <a name="estructura-repositorio"></a>

```
SafeSky_drone_project/
│
├── README.md
├── launch_safesky.sh                    # Lanzador automático Gazebo
│
├── Python simulation/
│   ├── safesky_v7_final/
│   │   ├── safesky_main_7.py            # Aplicación principal + GUI
│   │   ├── physics.py                   # Dinámica 6-DOF + PID
│   │   ├── planner.py                   # Hybrid A* + ORCA
│   │   └── kpis.py                      # Indicadores de desempeño
│   └── zn_safesky.py                    # Justificación Z-N (figuras)
│
├── Vision project/
    ├── drone_person_detection_fixed.ipynb
    ├── drone_person_laptop.ipynb
    ├── drone_project/
    │   └── tello_detection.py
    ├── training/
    ├── runs/
    ├── yolov8m.pt
    └── yolo26n.pt


```

---

## ⚙️ Instalación y Uso <a name="instalacion-y-uso"></a>

### Requisitos

```bash
Python >= 3.10
pip install numpy matplotlib scipy ultralytics opencv-python djitellopy
```

### Ejecutar el Simulador SafeSky

```bash
cd "Python simulation/safesky_v7_final"
python safesky_main_7.py
```

La interfaz permite configurar:
- Modo de misión: A→B paralelo, A→B cruzado, A→B línea recta, Circular
- Número de drones (3–6) y obstáculos (0–8)
- Modo de obstáculos: fijo, aleatorio, dinámico
- Parámetros de viento AR(1): persistencia α y σ intensidad
- Inyección de ráfaga súbita y objeto intruso durante la simulación

### Ejecutar la Detección con DJI Tello

> Conectar el equipo a la red WiFi del Tello antes de ejecutar.

```bash
cd "Vision project/drone_project"
python tello_detection.py
```

**Controles del teclado:**
- `T` — Despegar
- `L` — Aterrizar
- `W/A/S/D` — Movimiento horizontal
- `ESC` — Aterrizaje de emergencia

### Generar Justificación Z-N

```bash
cd "Python simulation"
python zn_safesky.py
```

---

## 📊 Resultados <a name="resultados"></a>

### Simulador SafeSky — Casos de Prueba

| Caso | Escenario | Exactitud Posicional (m) | Colisiones | Carga Cómputo (ms) |
|------|-----------|--------------------------|------------|---------------------|
| NUM 01 | Navegación A→B Cruzada | D1: 0.049 · D2: 0.039 · D3: 0.058 | 0 | 1.26 |
| NUM 02 | Evasión obstáculos volumétricos | D1: 0.003 · D2: 0.027 · D3: 0.058 | 0 | 1.37 |
| NUM 03 | Ráfaga de viento súbita (4 m/s) | D1: 0.060 · D2: 0.065 · D3: 0.161 | 0 | 1.06 |
| NUM 04 | Objeto intruso externo | D1: 0.003 · D2: 0.025 · D3: 0.065 | 0 | 1.23 |
| NUM 05 | Viento turbulento (α=0.4, σ=1) | D1: 0.003 · D2: 0.011 · D3: 0.069 | 0 | 1.25 |

Todos los casos superaron el criterio de aceptación `e_B < 0.10 m` (salvo D3 en NUM 03 por ráfaga extrema) y **cero colisiones** en todos los escenarios.

### Detección en Tiempo Real (DJI Tello)

El modelo detectó simultáneamente un dron físico (confianza **0.74**) y una persona (confianza **0.84**) operando a **2 FPS sobre CPU**, suficiente para demostración funcional.


## 📚 Referencias <a name="referencias"></a>

- Luukkonen, T. (2011). *Modelling and control of quadcopter*. Aalto University.
- Bouabdallah, S., Noth, A., & Siegwart, R. (2004). PID vs LQ control techniques applied to an indoor micro quadrotor. *IEEE/RSJ IROS*.
- Baharuddin, A., & Basri, M. A. M. (2023). Trajectory tracking of a quadcopter UAV using PID controller. *ELEKTRIKA*, 22(2).
- van den Berg, J. et al. (2011). Reciprocal n-body collision avoidance. *ISRR*.
- Özel, M. — [Drone Dataset (UAV)](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav)
- Luis Ángel — [Person Detection UAV](https://www.kaggle.com/datasets/luisngeld/person-detection-uav-pascal-voc-uaq-msuav)
- Singh, S. — [Drone Detection](https://www.kaggle.com/datasets/cybersimar08/drone-detection)

---

<div align="center">

*Tecnológico de Monterrey · Campus Hermosillo · MR3001B · 2026*

</div>
