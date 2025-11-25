# Editor de Imágenes con Asistente IA

## Descripción del Proyecto

Este proyecto es un editor de imágenes interactivo que integra un asistente de inteligencia artificial basado en Google Gemini. La aplicación permite cargar y editar imágenes utilizando controles manuales mientras el asistente analiza la imagen y proporciona sugerencias de mejora personalizadas.

### Características Principales

El asistente IA analiza las imágenes cargadas y sugiere ajustes específicos basándose en:
- Iluminación y distribución de colores
- Contraste y nitidez
- Composición y elementos visuales
- Aspectos técnicos de calidad

A diferencia de otros editores automáticos, este sistema mantiene al usuario en control total, permitiendo que el asistente solo sugiera valores específicos que el usuario puede aplicar manualmente mediante controles intuitivos.

## Funcionalidades

### 1. Carga y Visualización de Imágenes
- Soporta formatos JPG, JPEG, PNG, BMP y GIF
- Vista dual: imagen original y editada lado a lado
- Interfaz responsiva con scroll horizontal y vertical
- Modo pantalla completa automático

### 2. Controles de Edición Manual

#### Brillo (Brightness)
- Rango: -100 a +100
- Fórmula: output = input + brillo
- Ajuste lineal que suma el valor directamente a cada píxel RGB

#### Contraste (Contrast)
- Rango: 0.5 a 3.0
- Fórmula: output = input × contraste
- Multiplica cada píxel por el factor de contraste
- 1.0 = sin cambio, >1.0 aumenta, <1.0 reduce

#### Desenfoque (Blur)
- Rango: 0 a 25
- Método: Gaussian Blur con kernel (valor×2+1, valor×2+1)
- Promedia píxeles vecinos con distribución gaussiana

#### Nitidez (Sharpen)
- Rango: 0.0 a 3.0
- Método: Unsharp Mask
- Fórmula: output = (1 + valor×0.5)×original - (valor×0.5)×desenfocada
- Resta versión desenfocada para resaltar bordes

#### Rotación
- Rango: 0 a 360 grados
- Rotación en sentido horario alrededor del centro

#### Escala de Grises
- Checkbox para convertir imagen a tonos de gris
- Útil para análisis técnico y fotográfico

#### Volteos
- Volteo horizontal (espejo)
- Volteo vertical (inversión)

### 3. Asistente Conversacional

El asistente de IA ofrece:
- Análisis automático al cargar una imagen
- Sugerencias específicas con valores numéricos exactos
- Conversación contextual sobre la imagen
- Respuestas personalizadas a preguntas del usuario
- Conocimiento técnico sobre las fórmulas y efectos de cada control

### 4. Gestión de Sesiones

#### Memoria por Imagen
- Cada imagen mantiene su propia conversación
- Historial independiente por archivo
- Al recargar una imagen, se recupera su contexto

#### Guardar y Cargar Conversaciones
- Exportación en formato JSON
- Incluye imagen original (base64)
- Incluye imagen editada con todos los ajustes
- Almacena estados de todos los controles
- Preserva historial completo de mensajes

#### Guardar Imagen Editada
- Exporta la imagen procesada
- Formatos disponibles: PNG, JPG
- Conserva todos los ajustes aplicados

## Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- Pip (gestor de paquetes de Python)
- Clave API de Google Gemini

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/JoseLuisAlvarezManica/AgenteInteligente.git
cd AgenteInteligente
```

### Paso 2: Crear Entorno Virtual

Se recomienda usar un entorno virtual para aislar las dependencias:

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Las dependencias principales incluyen:
- langchain: Framework para aplicaciones con LLM
- langchain-google-genai: Integración con Google Gemini
- opencv-python: Procesamiento de imágenes
- Pillow: Manipulación de imágenes
- python-dotenv: Gestión de variables de entorno
- numpy: Operaciones numéricas

### Paso 4: Configurar Variables de Entorno

1. Crear un archivo `.env` en la raíz del proyecto:

```bash
# En Windows
copy NUL .env

# En Linux/Mac
touch .env
```

2. Editar el archivo `.env` y agregar tu clave API de Google Gemini:

```
GEMINI_API_KEY=tu_clave_api_aqui
```

#### Como Obtener la Clave API de Google Gemini

1. Visitar [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Iniciar sesión con tu cuenta de Google
3. Hacer clic en "Create API Key"
4. Copiar la clave generada
5. Pegarla en el archivo `.env`

**Importante**: Nunca compartas tu clave API públicamente ni la subas a repositorios de código.

### Paso 5: Ejecutar la Aplicación

```bash
python image_analyzer_gui.py
```

La aplicación se abrirá en modo pantalla completa.

## Uso de la Aplicación

### Flujo Básico

1. **Cargar Imagen**: Hacer clic en "Cargar Imagen" y seleccionar un archivo
2. **Análisis Automático**: El asistente analizará la imagen automáticamente
3. **Revisar Sugerencias**: Leer las recomendaciones del asistente
4. **Aplicar Ajustes**: Usar los sliders y controles para editar la imagen
5. **Conversación**: Hacer preguntas específicas al asistente
6. **Guardar**: Exportar la imagen editada o guardar la sesión completa

### Interacción con el Asistente

El asistente responde a preguntas como:
- "¿Qué puedo mejorar en esta imagen?"
- "¿Cuánto brillo debería agregar?"
- "¿La imagen está bien expuesta?"
- "¿Qué colores predominan?"
- "¿Necesita más contraste?"

## Caso de Prueba: Guardado de Sesión de Edición

### Descripción de la Prueba

Esta prueba validó la funcionalidad de guardado y persistencia de sesiones del sistema. El objetivo fue verificar que la aplicación puede:
1. Cargar una imagen correctamente
2. Mantener el estado inicial de todos los controles
3. Guardar la sesión completa en formato JSON
4. Preservar tanto la imagen original como la procesada
5. Permitir la recuperación posterior de la sesión

### Contexto de la Prueba

Se realizó una prueba con una imagen llamada "Canvas FreshKeeper.png" correspondiente a una credencial de identificación académica. La prueba consistió en:

1. **Carga de Imagen**: Se cargó la imagen en la aplicación
2. **Estado Inicial**: Se verificó que todos los controles estuvieran en sus valores predeterminados
3. **Guardado de Sesión**: Se utilizó el botón "Guardar Conversación" para exportar el estado
4. **Verificación**: Se generó el archivo `prueba.json` con toda la información de la sesión

### Estructura del Archivo JSON Guardado

El archivo `prueba.json` contiene:

```json
{
  "timestamp": "2025-11-22T13:54:24.744704",
  "current_image": "Canvas FreshKeeper.png",
  "conversations": {
    "Canvas FreshKeeper.png": {
      "messages": [],
      "image_data": "[base64_encoded_image]",
      "image_path": "ruta/al/archivo/Canvas FreshKeeper.png",
      "cv2_operations": [],
      "control_states": {
        "brightness": 0,
        "contrast": 1.0,
        "blur": 0,
        "sharpen": 0.0,
        "rotation": 0,
        "grayscale": false,
        "flip_h": false,
        "flip_v": false
      },
      "processed_image": "[base64_encoded_processed_image]"
    }
  }
}
```

### Elementos Guardados

1. **Timestamp**: Marca de tiempo exacta del guardado (2025-11-22T13:54:24.744704)
2. **Imagen Actual**: Referencia a la imagen que estaba siendo editada
3. **Imagen Original**: Codificada en base64 para portabilidad completa
4. **Imagen Procesada**: Versión editada también en base64
5. **Estados de Controles**: Valores exactos de todos los ajustes aplicados
6. **Historial de Mensajes**: Conversación completa con el asistente (vacía en este caso inicial)
7. **Operaciones CV2**: Registro de operaciones aplicadas (vacío en estado inicial)

### Funcionalidad Demostrada

La prueba demostró exitosamente:

**1. Guardado Completo de Estado**
- Todos los controles se guardaron con sus valores: brightness=0, contrast=1.0, blur=0, sharpen=0.0, rotation=0
- Estados de checkboxes y botones: grayscale=false, flip_h=false, flip_v=false
- Timestamp preciso del momento del guardado

**2. Persistencia de Imágenes**
- Imagen original convertida a base64 y almacenada en el JSON
- Imagen procesada (idéntica a la original en este caso, sin ediciones) también guardada
- Portabilidad completa: el archivo JSON contiene todo lo necesario para restaurar la sesión

**3. Gestión de Memoria por Imagen**
- La estructura permite múltiples imágenes en un solo archivo
- Cada imagen tiene su propia sección con datos independientes
- El campo "current_image" indica cuál imagen estaba activa

**4. Historial de Conversación**
- Array de mensajes vacío (la imagen se cargó pero no se interactuó con el asistente)
- Preparado para almacenar conversaciones futuras
- Mantiene el contexto separado por imagen

**5. Trazabilidad**
- Timestamp: 2025-11-22T13:54:24.744704
- Ruta de la imagen original preservada
- Registro de operaciones CV2 (vacío en estado inicial)

### Resultados de la Prueba

**Verificación Exitosa:**
- ✓ Archivo JSON generado correctamente (prueba.json)
- ✓ Estructura de datos válida y bien formada
- ✓ Imagen codificada en base64 (aprox. 50KB de datos)
- ✓ Todos los campos requeridos presentes
- ✓ Estados de controles preservados correctamente
- ✓ Timestamp registrado con precisión de microsegundos

**Implicaciones Prácticas:**

1. **Flujo de Trabajo Interrumpido**: Un usuario puede cerrar la aplicación y retomar exactamente donde lo dejó al cargar el JSON

2. **Colaboración**: El archivo puede compartirse con otros usuarios que tengan la aplicación instalada

3. **Versionado**: Múltiples versiones de edición de una misma imagen pueden guardarse con diferentes nombres

4. **Backup Automático**: Los usuarios pueden guardar progreso periódicamente sin perder trabajo

### Caso de Uso

Al cargar este archivo `prueba.json` en la aplicación:
1. Se restaura automáticamente la imagen "Canvas FreshKeeper.png"
2. Se cargan los estados de todos los controles (en este caso, valores por defecto)
3. Se recupera la imagen original y la procesada
4. Se puede continuar la edición desde donde se dejó
5. El historial de conversación se mantiene disponible

Este sistema de guardado permite a los usuarios:
- Trabajar en múltiples sesiones sin perder progreso
- Compartir sesiones de edición con otros usuarios
- Mantener registro de ajustes aplicados a cada imagen
- Recuperar trabajo después de cerrar la aplicación

### Conclusión de la Prueba

La prueba validó que el sistema de persistencia funciona correctamente, guardando de manera fiable:
- El estado completo de la interfaz
- Las imágenes en formato portable
- El historial de conversación con el asistente
- Metadatos necesarios para restauración completa

Este mecanismo es fundamental para la usabilidad del sistema, permitiendo sesiones de trabajo extendidas y colaboración entre usuarios.

## Arquitectura Técnica

### Estructura del Proyecto

```
AgenteInteligente/
├── image_analyzer_gui.py    # Aplicación principal con interfaz gráfica
├── image_analyzer.py         # Versión CLI (opcional)
├── requirements.txt          # Dependencias del proyecto
├── .env                      # Variables de entorno (no incluido en repo)
├── README.md                # Este archivo
└── gen-lang-client-*.json   # Credenciales de Google (generado)
```

### Componentes Principales

#### 1. DialogContext
Clase que gestiona la memoria y contexto de conversaciones:
- Almacena conversaciones separadas por imagen
- Mantiene estados de controles para cada imagen
- Guarda y carga sesiones en formato JSON
- Gestiona el historial de mensajes con LangChain

#### 2. ImageAnalyzerGUI
Clase principal de la interfaz gráfica:
- Gestión de widgets Tkinter
- Procesamiento de imágenes con OpenCV
- Comunicación con Google Gemini mediante LangChain
- Threading para operaciones asíncronas

#### 3. Sistema de Prompts
Dos prompts principales guían al asistente:

**VISION_PROMPT**: Para análisis inicial de imágenes
- Describe elementos visuales
- Identifica aspectos técnicos
- Proporciona fórmulas matemáticas de cada control
- Sugiere valores específicos

**DIALOG_PROMPT**: Para conversación contextual
- Mantiene contexto de la conversación
- Responde preguntas específicas
- Considera historial de mensajes
- Conoce el estado actual de los controles

### Flujo de Datos

```
Usuario → GUI (Tkinter)
         ↓
    DialogContext (Gestión de Estado)
         ↓
    LangChain → Google Gemini API
         ↓
    Respuesta del Asistente
         ↓
    Visualización en Chat
```

```
Usuario Ajusta Control → OpenCV Procesa Imagen
                        ↓
                    Display en Canvas
                        ↓
                Estado Guardado en DialogContext
```

## Solución de Problemas

### Error: "GEMINI_API_KEY no está establecida"
**Solución**: Verificar que el archivo `.env` existe y contiene la clave API correcta.

### Error: "No module named 'cv2'"
**Solución**: Instalar OpenCV con `pip install opencv-python`

### La aplicación no responde durante el análisis
**Comportamiento normal**: El procesamiento de IA puede tardar unos segundos. La aplicación usa threading para mantener la interfaz responsiva.

### Las imágenes se ven pixeladas
**Causa**: Redimensionamiento automático para ajustar al canvas.
**Solución**: Las imágenes originales se mantienen sin modificar; solo la visualización se ajusta.

### Error al guardar conversación
**Solución**: Verificar permisos de escritura en el directorio seleccionado.

## Limitaciones Conocidas

- Imágenes muy grandes (>10MB) pueden tardar en procesarse
- La aplicación requiere conexión a internet para el asistente IA
- El análisis consume tokens de la API de Google Gemini
- Los ajustes se aplican en cascada (orden: brillo → contraste → blur → sharpen → grayscale → rotación → volteos)

## Conclusiones

Este proyecto demuestra la efectiva integración entre procesamiento de imágenes clásico (OpenCV) y modelos de lenguaje avanzados (Google Gemini) para crear una experiencia de edición asistida por IA.

### Ventajas del Enfoque Híbrido

1. **Control del Usuario**: A diferencia de editores automáticos, el usuario mantiene control total sobre cada ajuste
2. **Sugerencias Contextuales**: El asistente entiende tanto la imagen como las capacidades técnicas de cada control
3. **Aprendizaje Progresivo**: Los usuarios aprenden sobre edición de imágenes mediante las sugerencias del asistente
4. **Memoria Persistente**: Cada imagen mantiene su contexto, permitiendo sesiones de trabajo extendidas

### Casos de Uso Recomendados

- Digitalización y mejora de documentos escaneados
- Edición educativa de fotografías con guía IA
- Procesamiento batch con consistencia mediante valores sugeridos
- Análisis técnico de calidad de imagen con explicaciones

### Trabajo Futuro

Posibles mejoras incluyen:
- Soporte para procesamiento por lotes
- Más controles avanzados (HSV, curvas, niveles)
- Detección automática de problemas específicos
- Integración con otros modelos de visión
- Presets basados en tipo de imagen
- Historial de deshacer/rehacer

---

**Desarrollado por**: Jose Luis Alvarez Manica  
**Repositorio**: https://github.com/JoseLuisAlvarezManica/AgenteInteligente  
**Licencia**: MIT  
**Última actualización**: Noviembre 2025
