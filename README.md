## PneumoScan

PneumoScan es un sistema modular para la detección asistida de neumonía mediante aprendizaje profundo. La solución incluye una API basada en FastAPI, una interfaz de visualización en Streamlit y utilidades para preprocesamiento y explicabilidad del modelo.

### Estructura del proyecto

```
pneumoscan/
├─ pneumoscan/                  # Paquete principal con utilidades de ML
│  ├─ utils/                    # Entrada/salida y utilidades comunes
│  ├─ model/                    # Carga, predicción y explicabilidad
│  └─ data/                     # Preprocesamiento
├─ api/                         # Servicio FastAPI para inferencia
├─ app/                         # Aplicación Streamlit para usuarios finales
├─ models/                      # Pesos entrenados y mapa de clases
├─ tests/                       # Tests unitarios
├─ Dockerfile.api               # Imagen para desplegar la API
├─ Dockerfile.app               # Imagen para desplegar la app Streamlit
├─ pyproject.toml               # Dependencias y metadatos del proyecto
└─ Makefile                     # Atajos para tareas comunes
```

### Primeros pasos

1. Crear y activar un entorno virtual.
2. Instalar dependencias:
   ```bash
   make install
   ```
3. Añadir el archivo de pesos reales en `models/efficientnet_pneumonia.weights.h5`.
4. Ejecutar los tests:
   ```bash
   pytest
   ```

### Variables de entorno

Completa el archivo `.env.example`, renómbralo a `.env` y ajusta las rutas de pesos si fuera necesario. También puedes definir `PNEUMOSCAN_API_ENDPOINT` para que el front-end apunte a otro servicio por defecto.

### Desarrollo

- `make run-api` levanta la API en modo recarga.
- `make run-app` (alias `make streamlit`) inicia el front de Streamlit (`app/app.py`). Desde la barra lateral puedes alternar entre la demo local y un endpoint externo.

### Licencia

Pendiente de definir.
.
Este push es de prueba
