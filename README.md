# 🤖 Agente Inteligente - Analizador de Imágenes con Gemini

Asistente conversacional de IA especializado en análisis de imágenes utilizando Google Gemini a través de LangChain. Este agente permite mantener conversaciones naturales sobre imágenes, con capacidad de guardar y retomar conversaciones previas.

## 📋 Características Técnicas

### Tecnologías Principales
- **Modelo de IA**: Google Gemini 2.5 Flash
- **Framework**: LangChain para gestión de conversaciones
- **Lenguaje**: Python 3.8+
- **Análisis**: Visión por computadora multimodal (texto + imagen)

### Funcionalidades

#### 🖼️ Análisis de Imágenes
- Análisis detallado de imágenes con descripción de:
  - Elementos principales
  - Colores predominantes y distribución
  - Contexto y ambiente
  - Detalles relevantes e interesantes
- Soporte para formatos: JPEG, PNG, y otros formatos de imagen comunes

#### 💬 Conversación Interactiva
- Diálogo natural y contextual sobre las imágenes analizadas
- Memoria conversacional que mantiene el contexto durante toda la sesión
- Referencias a puntos mencionados previamente
- Preguntas abiertas para mantener el diálogo activo

#### 💾 Persistencia de Conversaciones
- **Guardado en JSON**: Almacena conversaciones completas incluyendo:
  - Historial completo de mensajes
  - Imagen en formato base64
  - Timestamp de guardado
  - Metadatos de la conversación
- **Carga de conversaciones**: Restaura sesiones previas para continuar donde se quedó
- **Guardado automático**: Opción al salir para no perder el progreso

#### 🧠 Gestión de Memoria
- `ConversationBufferMemory` de LangChain para mantener contexto
- Separación clara entre mensajes del usuario y del asistente
- Formato estructurado para fácil recuperación

## 🛠️ Requisitos del Sistema

### Requisitos de Software
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conexión a internet (para API de Google Gemini)

### Dependencias Principales
```
langchain==0.3.27
langchain-core==0.3.79
langchain-google-genai==2.0.8
google-generativeai==0.8.5
python-dotenv==1.1.1
```

## 🚀 Instalación y Despliegue Local

### 1. Clonar el Repositorio
```bash
git clone https://github.com/JoseLuisAlvarezManica/AgenteInteligente.git
cd AgenteInteligente
```

### 2. Crear Entorno Virtual (Recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:
```env
GEMINI_API_KEY=tu_clave_api_de_gemini_aqui
```

#### Obtener una API Key de Google Gemini:
1. Visita [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Inicia sesión con tu cuenta de Google
3. Crea una nueva API Key
4. Copia la clave y pégala en el archivo `.env`

### 5. Ejecutar el Agente
```bash
python image_analyzer.py
```

## 📖 Guía de Uso

### Inicio de Sesión

Al ejecutar el programa, verás el menú principal:
```
=== Analizador de Imágenes con Gemini ===
----------------------------------------

¿Quieres cargar una conversación previa? (s/n):
```

#### Opción 1: Nueva Conversación
1. Responde `n` para iniciar una nueva conversación
2. Ingresa la ruta de la imagen a analizar
3. El agente generará un análisis inicial detallado
4. Inicia la conversación sobre la imagen

#### Opción 2: Cargar Conversación Previa
1. Responde `s` para cargar una conversación guardada
2. Ingresa el nombre del archivo JSON (ej: `mi_conversacion.json`)
3. El historial se cargará automáticamente
4. Continúa la conversación donde la dejaste

### Comandos Durante la Conversación

| Comando | Descripción |
|---------|-------------|
| `guardar` | Guarda la conversación actual en formato JSON |
| `nueva` | Analiza una nueva imagen (mantiene la sesión) |
| `salir` | Finaliza el programa (ofrece guardar antes de salir) |
| Cualquier texto | Continúa la conversación sobre la imagen |

### Ejemplos de Uso

#### Análisis de Imagen
```
Ingrese la ruta de la imagen a analizar: C:\imagenes\paisaje.jpg

Analizando imagen: C:\imagenes\paisaje.jpg

Descripción inicial:
--------------------------------------------------
Veo un hermoso paisaje montañoso al atardecer...
[análisis detallado]
¿Qué te parece el contraste entre las montañas y el cielo?
--------------------------------------------------

Tú: Me encanta cómo se reflejan los colores en el agua
Asistente: [respuesta contextual...]
```

#### Guardar Conversación
```
Tú: guardar
Nombre del archivo para guardar (sin extensión): paisaje_conversacion
✓ Conversación guardada exitosamente en paisaje_conversacion.json
```

#### Cargar Conversación
```
¿Quieres cargar una conversación previa? (s/n): s
Ingrese el nombre del archivo de conversación (con extensión .json): paisaje_conversacion.json
✓ Conversación cargada exitosamente desde paisaje_conversacion.json

--- Historial de conversación cargado ---
Usuario: Me encanta cómo se reflejan los colores en el agua
Asistente: [respuesta previa...]
--------------------------------------------------
```

## 📁 Estructura del Proyecto

```
AgenteInteligente/
│
├── image_analyzer.py          # Script principal del agente
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Este archivo
├── .env                        # Variables de entorno (no incluido en git)
├── .gitignore                  # Archivos ignorados por git
│
└── [conversaciones guardadas]  # Archivos .json generados
```

## 🔧 Configuración Avanzada

### Ajustar Parámetros del Modelo

En `image_analyzer.py`, línea 22-26:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",    # Modelo a utilizar
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0.7               # Creatividad (0.0 - 1.0)
)
```

- **temperature**: Controla la creatividad de las respuestas
  - `0.0`: Respuestas más determinísticas y predecibles
  - `1.0`: Respuestas más creativas y variadas
  - Recomendado: `0.7` para balance

### Personalizar Prompts

Los prompts del sistema se encuentran en las líneas 29-59:
- `VISION_PROMPT`: Guía el análisis inicial de la imagen
- `DIALOG_PROMPT`: Guía las respuestas durante la conversación

## 🔒 Seguridad

- ⚠️ **Nunca compartas tu archivo `.env`** con nadie
- ⚠️ **No incluyas tu API Key** en el código fuente
- ⚠️ El archivo `.gitignore` ya está configurado para excluir `.env`
- 🔐 Las API Keys son personales y no deben ser compartidas

## 📊 Formato de Datos JSON

Las conversaciones se guardan con la siguiente estructura:
```json
{
  "timestamp": "2025-10-13T15:30:45.123456",
  "messages": [
    {
      "type": "human",
      "content": "Mensaje del usuario"
    },
    {
      "type": "ai",
      "content": "Respuesta del asistente"
    }
  ],
  "image_data": "base64_encoded_image_data..."
}
```

## 🐛 Solución de Problemas

### Error: "La variable de entorno 'GEMINI_API_KEY' no está establecida"
**Solución**: Verifica que el archivo `.env` existe y contiene la clave API correcta.

### Error: "Import langchain_google_genai could not be resolved"
**Solución**: 
```bash
pip install langchain-google-genai==2.0.8
```

### Error al cargar imagen
**Solución**: 
- Verifica que la ruta de la imagen es correcta
- Usa rutas absolutas (ej: `C:\imagenes\foto.jpg`)
- Asegúrate de que el archivo existe y es una imagen válida

### Error de conexión API
**Solución**:
- Verifica tu conexión a internet
- Confirma que tu API Key es válida
- Revisa los límites de uso de tu cuenta de Google AI

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la licencia MIT.

## 👨‍💻 Autor

**Jose Luis Alvarez Manica**
- GitHub: [@JoseLuisAlvarezManica](https://github.com/JoseLuisAlvarezManica)

## 🙏 Agradecimientos

- Google por proporcionar la API de Gemini
- LangChain por el framework de IA conversacional
- La comunidad de Python por las excelentes bibliotecas

---

**Nota**: Este es un proyecto educativo. Para uso en producción, considera implementar:
- Manejo más robusto de errores
- Logging detallado
- Tests unitarios
- Rate limiting
- Validación de entrada de usuario