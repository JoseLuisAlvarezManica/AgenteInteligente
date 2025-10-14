# Importación de bibliotecas para un asistente de IA con Google Gemini usando LangChain
import os
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Importación de variables de entorno
load_dotenv()

# 1. Configuración y creación de un modelo basado en Gemini

# Verificación de la clave API como variable de entorno
if 'GEMINI_API_KEY' not in os.environ:
    print("Error: La variable de entorno 'GEMINI_API_KEY' no está establecida.")
    exit()

# Definición del modelo y cliente de Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0.4
)

# 2. Definición de prompts para el análisis y diálogo
VISION_PROMPT = """
Eres un asistente amigable y conversacional especializado en análisis de imágenes.
Tu objetivo es establecer un diálogo interactivo con el usuario sobre la imagen.

Para el análisis inicial, incluye estos aspectos en orden:
- Elementos principales en la imagen
- Colores predominantes y su distribución
- Contexto y ambiente de la escena
- Detalles relevantes o interesantes
Debes de ser conciso y claro, evitando suposiciones no basadas en la imagen.
Debes de presentar la información en párrafos cortos sin especificar que aspectos estas describiendo.


Después del análisis inicial, debes:
1. Terminar con una pregunta abierta sobre algún aspecto interesante de la imagen
2. Mantener una conversación natural sobre los elementos de la imagen
3. Usar la información previa de la conversación para hacer referencias y conexiones
4. Ser receptivo a las preguntas del usuario y responder de manera detallada
5. Si el usuario menciona algo que no está en la imagen, indícalo amablemente

Mantén un tono conversacional y amigable en todo momento.
"""

DIALOG_PROMPT = """
Continúa la conversación sobre la imagen analizada. Contexto de la conversación hasta ahora:
{context}

Última entrada del usuario: {user_input}

Responde de manera conversacional, mantén la coherencia con el contexto anterior y:
1. Haz referencias a puntos mencionados previamente cuando sea relevante
2. Si el usuario hace preguntas, respóndelas basándote solo en lo visible en la imagen
3. Si se menciona algo que no está en la imagen, indícalo amablemente
4. Termina con una pregunta o comentario que invite a continuar la conversación
5. Mantén el tono amigable y el interés en la perspectiva del usuario

Debes de ser conciso y claro.
Debes de presentar la información en párrafos cortos sin especificar qué aspectos estas describiendo.
"""

# 3. Funciones de procesamiento de imágenes

# Lee una imagen y la convierte a bytes.
def leer_imagen(ruta_imagen):
    #Intentar abrir y leer la imagen
    try:
        with open(ruta_imagen, 'rb') as f:
            return f.read()
    # Error si no se encuentra el archivo
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo: {ruta_imagen}")
        return None
    # Error inesperado al leer la imagen
    except Exception as e:
        print(f"Error inesperado al leer la imagen: {str(e)}")
        return None

# Clase para manejar el contexto de la conversación
class DialogContext:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.image_data = None
        
    def add_to_history(self, is_ai, entry):
        if is_ai:
            self.memory.chat_memory.add_message(AIMessage(content=entry))
        else:
            self.memory.chat_memory.add_message(HumanMessage(content=entry))
        
    def get_context_string(self):
        messages = self.memory.chat_memory.messages
        formatted_messages = []
        
        for message in messages:
            prefix = "Asistente: " if isinstance(message, AIMessage) else "Usuario: "
            formatted_messages.append(f"{prefix}{message.content}")
            
        return "\n".join(formatted_messages)
    
    def set_image_data(self, image_data):
        self.image_data = image_data
    
    #Funcion que guarfda la conversacion en formato JSON
    def save_conversation_to_json(self, filename):
        try:
            # Preparar los datos de la conversación
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": [],
                "image_data": None
            }
            
            # Convertir mensajes a formato serializable
            for message in self.memory.chat_memory.messages:
                message_data = {
                    "type": "ai" if isinstance(message, AIMessage) else "human",
                    "content": message.content
                }
                conversation_data["messages"].append(message_data)
            
            # Convertir imagen a base64 para guardarla
            if self.image_data:
                conversation_data["image_data"] = base64.b64encode(self.image_data).decode('utf-8')
            
            # Guardar en archivo JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            return True, f"Conversación guardada exitosamente en {filename}"
        
        except Exception as e:
            return False, f"Error al guardar la conversación: {str(e)}"
    
    #Funcion que carga la conversacion desde un archivo JSON
    def load_conversation_from_json(self, filename):
        try:
            # Leer el archivo JSON
            with open(filename, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Limpiar la memoria actual
            self.memory.chat_memory.clear()
            
            # Restaurar mensajes
            for message_data in conversation_data.get("messages", []):
                if message_data["type"] == "ai":
                    self.memory.chat_memory.add_message(AIMessage(content=message_data["content"]))
                else:
                    self.memory.chat_memory.add_message(HumanMessage(content=message_data["content"]))
            
            # Restaurar imagen si existe
            if conversation_data.get("image_data"):
                self.image_data = base64.b64decode(conversation_data["image_data"])
            
            return True, f"Conversación cargada exitosamente desde {filename}"
        
        except FileNotFoundError:
            return False, f"No se encontró el archivo: {filename}"
        except json.JSONDecodeError:
            return False, f"El archivo {filename} no tiene un formato JSON válido"
        except Exception as e:
            return False, f"Error al cargar la conversación: {str(e)}"

# Analiza una imagen usando el modelo Gemini y genera una descripción detallada.
def analizar_imagen(ruta_imagen, dialog_context):
    try:
        # Leer la imagen
        img_bytes = leer_imagen(ruta_imagen)
        # Verificar si la imagen se leyó correctamente
        if img_bytes is None:
            return False, "No se pudo leer la imagen"
        
        # Guardar los bytes de la imagen en el contexto
        dialog_context.set_image_data(img_bytes)
        
        # Codificar la imagen en base64 para LangChain
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Crear el mensaje con la imagen para LangChain
        message = HumanMessage(
            content=[
                {"type": "text", "text": VISION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                }
            ]
        )
        
        # Generar la descripción inicial usando LangChain
        response = llm.invoke([message])

        if response.content:
            # Guardar la descripción inicial en el historial usando Langchain
            dialog_context.add_to_history(True, response.content)
            return True, response.content
        else:
            return False, "No se pudo generar una descripción de la imagen"
            
    except Exception as e:
        return False, f"Error al analizar la imagen: {str(e)}"

# Continúa el diálogo con el usuario basado en la conversación previa y la imagen.
def continuar_dialogo(user_input, dialog_context):
    try:
        # Añadir la entrada del usuario al historial usando Langchain
        dialog_context.add_to_history(False, user_input)
        
        # Generar la respuesta usando el contexto de la conversación de Langchain
        prompt_with_context = DIALOG_PROMPT.format(
            context=dialog_context.get_context_string(),
            user_input=user_input
        )
        
        # Codificar la imagen en base64 para LangChain
        img_base64 = base64.b64encode(dialog_context.image_data).decode('utf-8')
        
        # Crear el mensaje con la imagen y el contexto para LangChain
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_with_context},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                }
            ]
        )
        
        # Generar la respuesta usando LangChain
        response = llm.invoke([message])
        
        if response.content:
            # Guardar la respuesta en el historial usando Langchain
            dialog_context.add_to_history(True, response.content)
            return True, response.content
        else:
            return False, "No se pudo generar una respuesta"
            
    except Exception as e:
        return False, f"Error en el diálogo: {str(e)}"
    

# 4. Función principal


# Función principal que maneja la interacción con el usuario y el flujo del programa.

def main():

    print("\n=== Analizador de Imágenes con Gemini ===")
    print("----------------------------------------")
    
    while True:
        # Inicializar el contexto del diálogo
        dialog_context = DialogContext()
        
        # Preguntar si quiere cargar una conversación previa
        cargar_conversacion = input("\n¿Quieres cargar una conversación previa? (s/n): ").strip().lower()
        
        if cargar_conversacion == 's':
            archivo_conversacion = input("Ingrese el nombre del archivo de conversación (con extensión .json): ").strip()
            if os.path.exists(archivo_conversacion):
                exito, mensaje = dialog_context.load_conversation_from_json(archivo_conversacion)
                if exito:
                    print(f"✓ {mensaje}")
                    # Si se carga una conversación, mostrar el historial
                    print("\n--- Historial de conversación cargado ---")
                    print(dialog_context.get_context_string())
                    print("-" * 50)
                    
                    # Ir directamente al diálogo interactivo si hay una imagen cargada
                    if dialog_context.image_data:
                        print("\nContinúa la conversación sobre la imagen cargada.")
                        print("Escribe 'nueva' para analizar otra imagen, 'guardar' para guardar la conversación, o 'salir' para terminar.")
                        
                        while True:
                            user_input = input("\nTú: ").strip()
                            
                            if user_input.lower() == 'salir':
                                # Preguntar si quiere guardar antes de salir
                                if dialog_context.memory.chat_memory.messages:
                                    guardar = input("\n¿Quieres guardar la conversación actual? (s/n): ").strip().lower()
                                    if guardar == 's':
                                        nombre_archivo = input("Nombre del archivo (sin extensión): ").strip() + ".json"
                                        exito_guardado, msg = dialog_context.save_conversation_to_json(nombre_archivo)
                                        print(msg)
                                print("\n¡Hasta luego!")
                                return
                            
                            if user_input.lower() == 'nueva':
                                print("\nCambiando a una nueva imagen...")
                                break
                                
                            if user_input.lower() == 'guardar':
                                nombre_archivo = input("Nombre del archivo para guardar (sin extensión): ").strip() + ".json"
                                exito_guardado, msg = dialog_context.save_conversation_to_json(nombre_archivo)
                                print(msg)
                                continue
                            
                            # Continuar el diálogo
                            exito, respuesta = continuar_dialogo(user_input, dialog_context)
                            
                            if exito:
                                print("\nAsistente:", respuesta)
                            else:
                                print(f"\nError: {respuesta}")
                        
                        print("\n" + "=" * 40)
                        continue
                else:
                    print(f"✗ {mensaje}")
            else:
                print("El archivo especificado no existe.")
        
        # Selección de imagen
        ruta_imagen = input("\nIngrese la ruta de la imagen a analizar (o 'salir' para terminar): ").strip()
        
        if ruta_imagen.lower() == 'salir':
            # Preguntar si quiere guardar antes de salir
            if dialog_context.memory.chat_memory.messages:
                guardar = input("\n¿Quieres guardar la conversación actual? (s/n): ").strip().lower()
                if guardar == 's':
                    nombre_archivo = input("Nombre del archivo (sin extensión): ").strip() + ".json"
                    exito_guardado, msg = dialog_context.save_conversation_to_json(nombre_archivo)
                    print(msg)
            print("\n¡Hasta luego!")
            break
            
        if not os.path.exists(ruta_imagen):
            print("Error: La ruta especificada no existe. Por favor, intente de nuevo.")
            continue
        
        # Analizar la imagen y mostrar el resultado inicial
        print(f"\nAnalizando imagen: {ruta_imagen}")
        exito, resultado = analizar_imagen(ruta_imagen, dialog_context)
        
        if not exito:
            print(f"\nError: {resultado}")
            continue
            
        print("\nDescripción inicial:")
        print("-" * 50)
        print(resultado)
        print("-" * 50)
        
        # Diálogo interactivo
        print("\nAhora puedes hacer preguntas o comentarios sobre la imagen.")
        print("Escribe 'nueva' para analizar otra imagen, 'guardar' para guardar la conversación, o 'salir' para terminar.")
        
        while True:
            user_input = input("\nTú: ").strip()
            
            if user_input.lower() == 'salir':
                # Preguntar si quiere guardar antes de salir
                if dialog_context.memory.chat_memory.messages:
                    guardar = input("\n¿Quieres guardar la conversación actual? (s/n): ").strip().lower()
                    if guardar == 's':
                        nombre_archivo = input("Nombre del archivo (sin extensión): ").strip() + ".json"
                        exito_guardado, msg = dialog_context.save_conversation_to_json(nombre_archivo)
                        print(msg)
                print("\n¡Hasta luego!")
                return
            
            if user_input.lower() == 'nueva':
                print("\nCambiando a una nueva imagen...")
                break
                
            if user_input.lower() == 'guardar':
                nombre_archivo = input("Nombre del archivo para guardar (sin extensión): ").strip() + ".json"
                exito_guardado, msg = dialog_context.save_conversation_to_json(nombre_archivo)
                print(msg)
                continue
            
            # Continuar el diálogo
            exito, respuesta = continuar_dialogo(user_input, dialog_context)
            
            if exito:
                print("\nAsistente:", respuesta)
            else:
                print(f"\nError: {respuesta}")
                
        print("\n" + "=" * 40)

if __name__ == "__main__":
    main()