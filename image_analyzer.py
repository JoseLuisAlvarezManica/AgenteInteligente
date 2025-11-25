# Importación de bibliotecas para un asistente de IA con Google Gemini usando LangChain
import os
# Importación de bibliotecas para la interfaz gráfica
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import os
import json
import base64
from datetime import datetime
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Importación de variables de entorno
load_dotenv()

# Verificación de la clave API
if 'GEMINI_API_KEY' not in os.environ:
    raise ValueError("Error: La variable de entorno 'GEMINI_API_KEY' no está establecida.")

# Definición del modelo LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv('GEMINI_API_KEY'),
    temperature=0.2
)

# Prompts del sistema
VISION_PROMPT = """
Eres un asistente especializado en análisis y procesamiento de imágenes.
Tu objetivo es analizar la imagen y SUGERIR modificaciones al usuario.

IMPORTANTE: NO apliques cambios automáticamente. Solo SUGIERE al usuario qué ajustes podría hacer.

Para el análisis inicial:
- Describe brevemente los elementos principales
- Colores predominantes y distribución
- Contexto y ambiente de la escena
- Aspectos técnicos (contraste, iluminación, nitidez)

Cuando sugieras mejoras, menciona específicamente los controles que el usuario puede usar:
No es necesario sugerir o comentar todos los controles, solo los que requieran ser modificados.

CONTROLES DISPONIBLES EN LA INTERFAZ:

1. BRILLO (Brightness): Slider de -100 a +100
   - Fórmula aplicada: output = input + brillo
   - Suma el valor directamente a cada píxel (ajuste lineal)
   - Valores negativos oscurecen, positivos iluminan
   - Ejemplo: brillo +30 suma 30 a cada valor RGB
   - Sugiere valores específicos (ej: "Aumenta el brillo a +30 para mejorar visibilidad")

2. CONTRASTE (Contrast): Slider de 0.5 a 3.0
   - Fórmula aplicada: output = input × contraste
   - Multiplica cada píxel por el factor de contraste
   - 1.0 = sin cambio, >1.0 expande rango tonal, <1.0 comprime
   - Ejemplo: contraste 1.5 multiplica cada RGB por 1.5
   - Sugiere valores específicos (ej: "Ajusta el contraste a 1.5 para resaltar detalles")

3. DESENFOQUE (Blur): Slider de 0 a 25
   - Método: Gaussian Blur con kernel (valor×2+1, valor×2+1)
   - Promedia píxeles vecinos con distribución gaussiana
   - 0 = sin desenfoque, valores mayores = kernel más grande = más suavizado
   - Ejemplo: desenfoque 5 usa kernel 11×11
   - Sugiere valores específicos (ej: "Aplica un desenfoque de 5 para suavizar ruido")

4. NITIDEZ (Sharpen): Slider de 0.0 a 3.0
   - Método: Unsharp Mask → output = (1 + valor×0.5)×original - (valor×0.5)×desenfocada
   - Resta versión desenfocada de la original para resaltar bordes
   - 0 = sin cambio, valores mayores = más definición en bordes
   - Ejemplo: nitidez 1.5 aplica factor 1.75 a original y -0.75 a desenfocada
   - Sugiere valores específicos (ej: "Aumenta la nitidez a 1.5 para más definición")

5. ROTACIÓN (Rotate): Slider de 0 a 360 grados
   - Sugiere ángulos específicos (ej: "Rota 90 grados para corregir orientación")

6. ESCALA DE GRISES: Checkbox
   - Sugiere cuándo sería útil (ej: "Activa escala de grises para análisis técnico")

7. VOLTEO HORIZONTAL: Botón
   - Sugiere cuándo aplicar (ej: "Voltea horizontalmente para efecto espejo")

8. VOLTEO VERTICAL: Botón
   - Sugiere cuándo aplicar (ej: "Voltea verticalmente para corregir orientación")

RECUERDA:
- NO generes código JSON
- NO apliques cambios automáticamente
- Solo SUGIERE valores específicos que el usuario puede ajustar
- Habla directamente sobre los controles (ej: "Te sugiero aumentar el brillo a +40")
- Sé conversacional y claro
- Si la imagen es adecuada puedes indicarlo

Mantén un tono amigable y didáctico.
"""

DIALOG_PROMPT = """
Continúa la conversación sobre la IMAGEN ACTUAL. 

CONTEXTO DE ESTA IMAGEN:
{context}

Última entrada del usuario: {user_input}

CONTROLES DISPONIBLES EN LA INTERFAZ:
1. Brillo: slider de -100 a +100
2. Contraste: slider de 0.5 a 3.0
3. Desenfoque: slider de 0 a 25
4. Nitidez: slider de 0.0 a 3.0
5. Rotación: slider de 0 a 360 grados
6. Escala de grises: checkbox
7. Volteo horizontal: botón
8. Volteo vertical: botón

IMPORTANTE:
- NO generes código JSON
- NO apliques cambios automáticamente
- SOLO SUGIERE valores específicos que el usuario puede ajustar usando los controles
- Habla directamente sobre los controles de la interfaz
- Ejemplo: "Te sugiero aumentar el brillo a +40 y el contraste a 1.3"
- Si la imagen es adecuada puedes indicarlo

Responde de manera conversacional y amigable.
"""

# Funciones de procesamiento CV2
def apply_cv2_operation(image, operation_data):
    """
    Aplica una operación CV2 basada en los datos del comando
    
    Args:
        image: Imagen OpenCV (numpy array)
        operation_data: Dict con 'operation' y 'params'
    
    Returns:
        Tuple (success, processed_image o error_message, description)
    """
    try:
        operation = operation_data.get("operation", "").lower()
        params = operation_data.get("params", {})
        reason = operation_data.get("reason", "")
        
        result = image.copy()
        description = f"{operation}: {reason}"
        
        if operation == "brightness":
            value = params.get("value", 0)
            result = cv2.convertScaleAbs(image, alpha=1, beta=value)
            description = f"Ajuste de brillo ({value:+d}): {reason}"
        
        elif operation == "contrast":
            alpha = params.get("alpha", 1.0)
            beta = params.get("beta", 0)
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            description = f"Ajuste de contraste (α={alpha}): {reason}"
        
        elif operation == "blur":
            kernel = params.get("kernel", 5)
            # Asegurar que el kernel es impar
            kernel = kernel if kernel % 2 == 1 else kernel + 1
            result = cv2.GaussianBlur(image, (kernel, kernel), 0)
            description = f"Desenfoque Gaussiano (kernel={kernel}): {reason}"
        
        elif operation == "sharpen":
            amount = params.get("amount", 1.5)
            gaussian = cv2.GaussianBlur(image, (5, 5), 0)
            result = cv2.addWeighted(image, 1 + amount, gaussian, -amount, 0)
            description = f"Nitidez (amount={amount}): {reason}"
        
        elif operation == "edge_detection":
            threshold1 = params.get("threshold1", 100)
            threshold2 = params.get("threshold2", 200)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1, threshold2)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            description = f"Detección de bordes Canny ({threshold1}, {threshold2}): {reason}"
        
        elif operation == "grayscale":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            description = f"Escala de grises: {reason}"
        
        elif operation == "rotate":
            angle = params.get("angle", 0)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(image, matrix, (w, h))
            description = f"Rotación ({angle}°): {reason}"
        
        elif operation == "flip":
            direction = params.get("direction", "horizontal")
            flip_code = 1 if direction == "horizontal" else 0
            result = cv2.flip(image, flip_code)
            description = f"Volteo {direction}: {reason}"
        
        elif operation == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
            result = cv2.transform(image, kernel)
            result = np.clip(result, 0, 255).astype(np.uint8)
            description = f"Efecto sepia: {reason}"
        
        elif operation == "negative":
            result = cv2.bitwise_not(image)
            description = f"Negativo: {reason}"
        
        else:
            return False, f"Operación desconocida: {operation}", ""
        
        return True, result, description
    
    except Exception as e:
        return False, f"Error al aplicar {operation}: {str(e)}", ""

def extract_cv2_commands(text):
    """
    Extrae comandos CV2 en formato JSON del texto
    
    Returns:
        List de dicts con operaciones CV2
    """
    commands = []
    
    # Buscar bloques JSON en el texto
    import re
    # Patrón que captura JSON con o sin saltos de línea, con flags DOTALL para . incluya \n
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            # Limpiar el match de espacios extra
            json_text = match.strip()
            command = json.loads(json_text)
            if "operation" in command:
                commands.append(command)
        except json.JSONDecodeError as e:
            # Si falla, intentar encontrar JSON sin bloques de código
            continue
    
    # Si no encontró nada con bloques de código, buscar objetos JSON directamente
    if not commands:
        # Buscar patrones que se vean como JSON de operación
        json_object_pattern = r'\{[^{}]*"operation"\s*:\s*"[^"]+"\s*[^{}]*\}'
        json_matches = re.findall(json_object_pattern, text, re.DOTALL)
        
        for match in json_matches:
            try:
                command = json.loads(match)
                if "operation" in command:
                    commands.append(command)
            except json.JSONDecodeError:
                continue
    
    return commands

# Clase para manejar el contexto de diálogo con memoria por imagen
class DialogContext:
    def __init__(self):
        # Memoria separada por imagen (key: nombre_archivo)
        self.image_conversations = {}
        self.current_image_name = None
        self.current_image_data = None
        self.current_image_path = None
        
    def set_current_image(self, image_data, image_path):
        """Establece la imagen actual y crea/recupera su conversación"""
        self.current_image_data = image_data
        self.current_image_path = image_path
        self.current_image_name = os.path.basename(image_path) if image_path else "unknown"
        
        # Crear nueva conversación para esta imagen si no existe
        if self.current_image_name not in self.image_conversations:
            self.image_conversations[self.current_image_name] = {
                "messages": [],  # Lista simple de mensajes
                "image_data": image_data,
                "image_path": image_path,
                "cv2_operations": [],  # Registro de operaciones aplicadas
                "control_states": {},  # Estados de los controles
                "processed_image": None  # Imagen procesada en base64
            }
    
    def get_current_messages(self):
        """Obtiene la lista de mensajes de la imagen actual"""
        if self.current_image_name in self.image_conversations:
            return self.image_conversations[self.current_image_name]["messages"]
        return []
    
    def add_to_history(self, is_ai, entry):
        """Añade mensaje al historial de la imagen actual"""
        messages = self.get_current_messages()
        if messages is not None:
            if is_ai:
                messages.append(AIMessage(content=entry))
            else:
                messages.append(HumanMessage(content=entry))
    
    def add_cv2_operation(self, operation_data):
        """Registra una operación CV2 aplicada"""
        if self.current_image_name in self.image_conversations:
            self.image_conversations[self.current_image_name]["cv2_operations"].append({
                "timestamp": datetime.now().isoformat(),
                "operation": operation_data
            })
    
    def get_cv2_operations(self):
        """Obtiene operaciones CV2 de la imagen actual"""
        if self.current_image_name in self.image_conversations:
            return self.image_conversations[self.current_image_name]["cv2_operations"]
        return []
    
    def get_context_string(self):
        """Obtiene el contexto de la imagen actual como string"""
        messages = self.get_current_messages()
        if not messages:
            return ""
        
        formatted_messages = []
        
        for message in messages:
            prefix = "Asistente: " if isinstance(message, AIMessage) else "Usuario: "
            formatted_messages.append(f"{prefix}{message.content}")
        
        # Agregar información de operaciones CV2 aplicadas
        ops = self.get_cv2_operations()
        if ops:
            formatted_messages.append("\n[Operaciones CV2 aplicadas a esta imagen:]")
            for op in ops:
                formatted_messages.append(f"- {op['operation'].get('operation', 'unknown')}: {op['operation'].get('reason', '')}")
            
        return "\n".join(formatted_messages)
    
    def get_all_images(self):
        """Obtiene lista de todas las imágenes en memoria"""
        return list(self.image_conversations.keys())
    
    def switch_to_image(self, image_name):
        """Cambia el contexto a otra imagen en memoria"""
        if image_name in self.image_conversations:
            self.current_image_name = image_name
            conv = self.image_conversations[image_name]
            self.current_image_data = conv["image_data"]
            self.current_image_path = conv["image_path"]
            return True
        return False
    
    def save_conversation_to_json(self, filename):
        """Guarda TODAS las conversaciones en formato JSON"""
        try:
            all_conversations = {}
            
            for img_name, conv_data in self.image_conversations.items():
                messages = []
                for message in conv_data["messages"]:
                    messages.append({
                        "type": "ai" if isinstance(message, AIMessage) else "human",
                        "content": message.content
                    })
                
                all_conversations[img_name] = {
                    "messages": messages,
                    "image_data": base64.b64encode(conv_data["image_data"]).decode('utf-8') if conv_data["image_data"] else None,
                    "image_path": conv_data["image_path"],
                    "cv2_operations": conv_data["cv2_operations"],
                    "control_states": conv_data.get("control_states", {}),
                    "processed_image": conv_data.get("processed_image")
                }
            
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "current_image": self.current_image_name,
                "conversations": all_conversations
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            return True, f"Conversaciones guardadas exitosamente en {filename}"
        
        except Exception as e:
            return False, f"Error al guardar la conversación: {str(e)}"
    
    def load_conversation_from_json(self, filename):
        """Carga conversaciones desde un archivo JSON"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Limpiar memoria actual
            self.image_conversations = {}
            
            # Restaurar todas las conversaciones
            for img_name, conv_data in conversation_data.get("conversations", {}).items():
                messages = []
                
                for message_data in conv_data.get("messages", []):
                    if message_data["type"] == "ai":
                        messages.append(AIMessage(content=message_data["content"]))
                    else:
                        messages.append(HumanMessage(content=message_data["content"]))
                
                image_data = None
                if conv_data.get("image_data"):
                    image_data = base64.b64decode(conv_data["image_data"])
                
                self.image_conversations[img_name] = {
                    "messages": messages,
                    "image_data": image_data,
                    "image_path": conv_data.get("image_path"),
                    "cv2_operations": conv_data.get("cv2_operations", []),
                    "control_states": conv_data.get("control_states", {}),
                    "processed_image": conv_data.get("processed_image")
                }
            
            # Restaurar imagen actual
            current_img = conversation_data.get("current_image")
            if current_img and current_img in self.image_conversations:
                self.switch_to_image(current_img)
            
            return True, f"Conversaciones cargadas exitosamente desde {filename}"
        
        except FileNotFoundError:
            return False, f"No se encontró el archivo: {filename}"
        except json.JSONDecodeError:
            return False, f"El archivo {filename} no tiene un formato JSON válido"
        except Exception as e:
            return False, f"Error al cargar la conversación: {str(e)}"

# Clase principal de la aplicación GUI
class ImageAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Imágenes con Gemini - Con Procesamiento CV2 Inteligente")
        
        # Iniciar en pantalla completa
        self.root.state('zoomed')  # Windows: maximizar ventana
        
        # Contexto del diálogo con memoria por imagen
        self.dialog_context = DialogContext()
        
        # Variables de estado
        self.original_image = None
        self.processed_image = None
        self.flip_h = False
        self.flip_v = False
        
        # Configurar la interfaz
        self.setup_ui()
        
    def setup_ui(self):
        # Crear canvas con scrollbars para scroll vertical y horizontal
        canvas_container = tk.Canvas(self.root, highlightthickness=0)
        scrollbar_v = ttk.Scrollbar(self.root, orient="vertical", command=canvas_container.yview)
        scrollbar_h = ttk.Scrollbar(self.root, orient="horizontal", command=canvas_container.xview)
        
        # Frame principal dentro del canvas
        main_frame = ttk.Frame(canvas_container, padding="10")
        
        # Crear ventana en el canvas
        canvas_window = canvas_container.create_window((0, 0), window=main_frame, anchor="nw")
        
        # Configurar scrollbars
        canvas_container.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        # Posicionar canvas y scrollbars
        canvas_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_v.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_h.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configurar el grid para que se expanda
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Actualizar scroll region cuando cambia el tamaño
        def configure_scroll(event):
            canvas_container.configure(scrollregion=canvas_container.bbox("all"))
        main_frame.bind("<Configure>", configure_scroll)
        
        # Bind mousewheel para scroll vertical
        def on_mousewheel(event):
            canvas_container.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_container.bind_all("<MouseWheel>", on_mousewheel)
        
        # Bind Shift+mousewheel para scroll horizontal
        def on_shift_mousewheel(event):
            canvas_container.xview_scroll(int(-1*(event.delta/120)), "units")
        canvas_container.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)
        
        # ===== PANEL IZQUIERDO: Imagen y Controles =====
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        left_frame.columnconfigure(0, weight=1)
        left_frame.columnconfigure(1, weight=1)
        left_frame.rowconfigure(0, weight=3)
        left_frame.rowconfigure(1, weight=1)
        
        # Vista de imágenes (dos canvas lado a lado)
        images_container = ttk.Frame(left_frame)
        images_container.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        images_container.columnconfigure(0, weight=1)
        images_container.columnconfigure(1, weight=1)
        images_container.rowconfigure(0, weight=1)
        
        # Canvas para imagen original
        original_frame = ttk.LabelFrame(images_container, text="📷 Imagen Original", padding="5")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(original_frame, bg='#2b2b2b', width=400, height=400, highlightthickness=0)
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas para imagen procesada
        processed_frame = ttk.LabelFrame(images_container, text="✨ Imagen Editada", padding="5")
        processed_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        processed_frame.columnconfigure(0, weight=1)
        processed_frame.rowconfigure(0, weight=1)
        
        self.processed_canvas = tk.Canvas(processed_frame, bg='#2b2b2b', width=400, height=400, highlightthickness=0)
        self.processed_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Panel de controles
        controls_frame = ttk.LabelFrame(left_frame, text="🎨 Controles de Edición", padding="10")
        controls_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        controls_frame.columnconfigure(1, weight=1)
        
        # Variables de control
        self.brightness_var = tk.IntVar(value=0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.IntVar(value=0)
        self.sharpen_var = tk.DoubleVar(value=0.0)
        self.rotation_var = tk.IntVar(value=0)
        self.grayscale_var = tk.BooleanVar(value=False)
        
        # Crear controles con mejor diseño y actualización de labels
        row = 0
        
        # Brillo
        ttk.Label(controls_frame, text="💡 Brillo:", font=('Arial', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        brightness_slider = ttk.Scale(controls_frame, from_=-100, to=100, variable=self.brightness_var, 
                                      orient=tk.HORIZONTAL, length=300, command=lambda v: self.update_slider_label('brightness', v))
        brightness_slider.grid(row=row, column=1, padx=5, sticky=(tk.W, tk.E))
        self.brightness_label = ttk.Label(controls_frame, text="Brillo: 0", width=15, anchor=tk.W, font=('Arial', 9))
        self.brightness_label.grid(row=row, column=2, padx=5)
        
        # Contraste
        row += 1
        ttk.Label(controls_frame, text="🔆 Contraste:", font=('Arial', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        contrast_slider = ttk.Scale(controls_frame, from_=0.5, to=3.0, variable=self.contrast_var,
                                    orient=tk.HORIZONTAL, length=300, command=lambda v: self.update_slider_label('contrast', v))
        contrast_slider.grid(row=row, column=1, padx=5, sticky=(tk.W, tk.E))
        self.contrast_label = ttk.Label(controls_frame, text="Contraste: 1.00", width=15, anchor=tk.W, font=('Arial', 9))
        self.contrast_label.grid(row=row, column=2, padx=5)
        
        # Desenfoque
        row += 1
        ttk.Label(controls_frame, text="🌫️ Desenfoque:", font=('Arial', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        blur_slider = ttk.Scale(controls_frame, from_=0, to=25, variable=self.blur_var,
                                orient=tk.HORIZONTAL, length=300, command=lambda v: self.update_slider_label('blur', v))
        blur_slider.grid(row=row, column=1, padx=5, sticky=(tk.W, tk.E))
        self.blur_label = ttk.Label(controls_frame, text="Desenfoque: 0", width=15, anchor=tk.W, font=('Arial', 9))
        self.blur_label.grid(row=row, column=2, padx=5)
        
        # Nitidez
        row += 1
        ttk.Label(controls_frame, text="✨ Nitidez:", font=('Arial', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        sharpen_slider = ttk.Scale(controls_frame, from_=0.0, to=3.0, variable=self.sharpen_var,
                                    orient=tk.HORIZONTAL, length=300, command=lambda v: self.update_slider_label('sharpen', v))
        sharpen_slider.grid(row=row, column=1, padx=5, sticky=(tk.W, tk.E))
        self.sharpen_label = ttk.Label(controls_frame, text="Nitidez: 0.0", width=15, anchor=tk.W, font=('Arial', 9))
        self.sharpen_label.grid(row=row, column=2, padx=5)
        
        # Rotación
        row += 1
        ttk.Label(controls_frame, text="🔄 Rotación:", font=('Arial', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        rotation_slider = ttk.Scale(controls_frame, from_=0, to=360, variable=self.rotation_var,
                                    orient=tk.HORIZONTAL, length=300, command=lambda v: self.update_slider_label('rotation', v))
        rotation_slider.grid(row=row, column=1, padx=5, sticky=(tk.W, tk.E))
        self.rotation_label = ttk.Label(controls_frame, text="Rotación: 0°", width=15, anchor=tk.W, font=('Arial', 9))
        self.rotation_label.grid(row=row, column=2, padx=5)
        
        # Escala de grises (checkbox)
        row += 1
        ttk.Label(controls_frame, text=" Escala de grises:", font=('Arial', 9, 'bold')).grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        ttk.Checkbutton(controls_frame, text="Activar", variable=self.grayscale_var,
                       command=self.apply_all_edits).grid(row=row, column=1, sticky=tk.W, padx=5)
        
        # Controles adicionales (botones)
        row += 1
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="↔️ Volteo Horizontal", command=self.flip_horizontal).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="↕️ Volteo Vertical", command=self.flip_vertical).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Resetear", command=self.reset_image).pack(side=tk.LEFT, padx=5)
        
        # ===== PANEL DERECHO: Chat =====
        right_frame = ttk.LabelFrame(main_frame, text="Chat con el Agente", padding="10")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        # Área de chat (terminal)
        self.chat_display = scrolledtext.ScrolledText(
            right_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=30,
            font=("Consolas", 10),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        # Frame para entrada de texto
        input_frame = ttk.Frame(right_frame)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        self.message_entry = ttk.Entry(input_frame, font=("Arial", 10))
        self.message_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.message_entry.bind("<Return>", lambda e: self.send_message())
        
        self.send_button = ttk.Button(input_frame, text="Enviar", command=self.send_message)
        self.send_button.grid(row=0, column=1)
        
        # ===== BARRA DE HERRAMIENTAS SUPERIOR =====
        toolbar = ttk.Frame(main_frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(toolbar, text="📂 Cargar Imagen", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="💾 Guardar Imagen Editada", command=self.save_edited_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="💬 Guardar Conversación", command=self.save_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="📥 Cargar Conversación", command=self.load_conversation).pack(side=tk.LEFT, padx=5)
        
        # Indicador de imagen actual
        self.image_label = ttk.Label(toolbar, text="Sin imagen cargada", foreground="gray")
        self.image_label.pack(side=tk.LEFT, padx=(20, 5))
        
        # Mensaje inicial
        self.add_message("Sistema", "Bienvenido al Editor de Imágenes con Asistente IA\n\nCARACTERÍSTICAS:\n- Carga una imagen y edítala usando los controles\n- El asistente te dará SUGERENCIAS de mejora\n- Guarda tu imagen editada cuando termines\n\nCarga una imagen para comenzar...", "system")
    
    def add_message(self, sender, message, msg_type="user"):
        """Añade un mensaje al chat"""
        self.chat_display.config(state=tk.NORMAL)
        
        if msg_type == "system":
            self.chat_display.insert(tk.END, f"[{sender}] ", "system_tag")
            self.chat_display.insert(tk.END, f"{message}\n\n", "system_msg")
        elif msg_type == "user":
            self.chat_display.insert(tk.END, f"👤 {sender}: ", "user_tag")
            self.chat_display.insert(tk.END, f"{message}\n\n", "user_msg")
        else:  # assistant
            self.chat_display.insert(tk.END, f"🤖 {sender}: ", "ai_tag")
            self.chat_display.insert(tk.END, f"{message}\n\n", "ai_msg")
        
        # Configurar tags de colores
        self.chat_display.tag_config("system_tag", foreground="#FFD700", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("system_msg", foreground="#CCCCCC")
        self.chat_display.tag_config("user_tag", foreground="#4CAF50", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("user_msg", foreground="#E0E0E0")
        self.chat_display.tag_config("ai_tag", foreground="#2196F3", font=("Consolas", 10, "bold"))
        self.chat_display.tag_config("ai_msg", foreground="#FFFFFF")
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def load_image(self):
        """Carga una imagen desde el sistema de archivos"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Leer imagen con OpenCV
                self.original_image = cv2.imread(file_path)
                
                if self.original_image is None:
                    messagebox.showerror("Error", "No se pudo cargar la imagen")
                    return
                
                # Leer bytes de la imagen para el agente
                with open(file_path, 'rb') as f:
                    img_bytes = f.read()
                
                # Establecer imagen actual en el contexto (crea/recupera su memoria)
                self.dialog_context.set_current_image(img_bytes, file_path)
                
                # Resetear variables de control
                self.brightness_var.set(0)
                self.contrast_var.set(1.0)
                self.blur_var.set(0)
                self.sharpen_var.set(0)
                self.rotation_var.set(0)
                self.grayscale_var.set(False)
                self.flip_h = False
                self.flip_v = False
                
                self.processed_image = self.original_image.copy()
                
                # Mostrar imagen
                self.display_images()
                
                # Actualizar etiqueta
                self.image_label.config(text=f"Imagen actual: {os.path.basename(file_path)}", foreground="blue")
                
                # Añadir mensaje al chat
                image_name = os.path.basename(file_path)
                messages = self.dialog_context.get_current_messages()
                self.add_message("Sistema", f"Imagen cargada: {image_name}\nMemoria de esta imagen: {len(messages)} mensajes", "system")
                
                # Analizar imagen automáticamente solo si es nueva (sin memoria)
                if len(messages) == 0:
                    self.analyze_image()
                else:
                    self.add_message("Sistema", "Conversación previa encontrada para esta imagen. Puedes continuar donde lo dejaste.", "system")
                    # Cargar estados de controles guardados
                    self.load_control_states()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar la imagen: {str(e)}")
    
    def display_images(self):
        """Muestra la imagen original y la procesada en sus respectivos canvas"""
        if self.original_image is None:
            return
        
        # Mostrar imagen original
        self.display_image_in_canvas(self.original_image, self.original_canvas)
        
        # Mostrar imagen procesada
        if self.processed_image is not None:
            self.display_image_in_canvas(self.processed_image, self.processed_canvas)
    
    def display_image_in_canvas(self, cv_image, canvas):
        """Muestra una imagen de OpenCV en un canvas específico"""
        # Convertir de BGR a RGB
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para ajustar al canvas
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 400
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 400
        
        h, w = image_rgb.shape[:2]
        scale = min(canvas_width/w, canvas_height/h, 1)
        new_w, new_h = int(w*scale), int(h*scale)
        
        resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Convertir a PIL Image y luego a PhotoImage
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Limpiar canvas y mostrar imagen
        canvas.delete("all")
        canvas.image = photo  # Guardar referencia
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
    
    def update_slider_label(self, control_name, value):
        """Actualiza el texto de la etiqueta del slider y aplica cambios"""
        try:
            # Convertir value a numérico
            numeric_value = float(value)
            
            # Formatear según el tipo de control
            if control_name == 'brightness':
                self.brightness_label.config(text=f"Brillo: {int(numeric_value)}")
            elif control_name == 'contrast':
                self.contrast_label.config(text=f"Contraste: {numeric_value:.2f}")
            elif control_name == 'blur':
                blur_val = int(numeric_value)
                self.blur_label.config(text=f"Desenfoque: {blur_val}")
            elif control_name == 'sharpen':
                self.sharpen_label.config(text=f"Nitidez: {numeric_value:.1f}")
            elif control_name == 'rotation':
                self.rotation_label.config(text=f"Rotación: {int(numeric_value)}°")
            
            # Aplicar cambios
            self.apply_all_edits()
        except Exception as e:
            print(f"Error updating slider label: {e}")
    
    def apply_all_edits(self, event=None):
        """Aplica todas las ediciones de los controles a la imagen"""
        if self.original_image is None:
            return
        
        # Comenzar con la imagen original
        img = self.original_image.copy()
        
        # Aplicar brillo
        brightness = self.brightness_var.get()
        if brightness != 0:
            img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
        
        # Aplicar contraste
        contrast = self.contrast_var.get()
        if contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        
        # Aplicar desenfoque
        blur_amount = self.blur_var.get()
        if blur_amount > 0:
            ksize = blur_amount * 2 + 1  # Debe ser impar
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        # Aplicar nitidez (método más suave y controlado)
        sharpen_amount = self.sharpen_var.get()
        if sharpen_amount > 0:
            # Crear versión desenfocada
            gaussian = cv2.GaussianBlur(img, (0, 0), 3)
            # Mezclar original con desenfocada para aumentar nitidez
            # amount controla la intensidad (valores típicos: 0.5 a 2.0)
            img = cv2.addWeighted(img, 1.0 + sharpen_amount * 0.5, gaussian, -sharpen_amount * 0.5, 0)
        
        # Aplicar escala de grises
        if self.grayscale_var.get():
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convertir de vuelta a 3 canales
        
        # Aplicar rotación
        rotation = self.rotation_var.get()
        if rotation != 0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h))
        
        # Aplicar volteos
        if self.flip_h:
            img = cv2.flip(img, 1)
        if self.flip_v:
            img = cv2.flip(img, 0)
        
        self.processed_image = img
        self.display_images()
        
        # Guardar estado actual
        self.save_control_states()
    
    def flip_horizontal(self):
        """Voltea la imagen horizontalmente"""
        if self.original_image is None:
            return
        self.flip_h = not self.flip_h
        self.apply_all_edits()
    
    def flip_vertical(self):
        """Voltea la imagen verticalmente"""
        if self.original_image is None:
            return
        self.flip_v = not self.flip_v
        self.apply_all_edits()
    
    def reset_image(self):
        """Resetea la imagen y todos los controles a sus valores por defecto"""
        if self.original_image is None:
            return
        
        # Resetear variables de control
        self.brightness_var.set(0)
        self.contrast_var.set(1.0)
        self.blur_var.set(0)
        self.sharpen_var.set(0)
        self.rotation_var.set(0)
        self.grayscale_var.set(False)
        self.flip_h = False
        self.flip_v = False
        
        # Resetear etiquetas de los controles
        self.brightness_label.config(text="Brillo: 0")
        self.contrast_label.config(text="Contraste: 1.00")
        self.blur_label.config(text="Desenfoque: 0")
        self.sharpen_label.config(text="Nitidez: 0.0")
        self.rotation_label.config(text="Rotación: 0°")
        
        # Resetear imagen procesada
        self.processed_image = self.original_image.copy()
        self.display_images()
        self.add_message("Sistema", "Imagen y controles reseteados a estado original", "system")
    
    def save_edited_image(self):
        """Guarda la imagen editada"""
        if self.processed_image is None:
            messagebox.showwarning("Advertencia", "No hay imagen para guardar")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ],
            title="Guardar Imagen Editada"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                self.add_message("Sistema", f"✓ Imagen guardada: {os.path.basename(file_path)}", "system")
                messagebox.showinfo("Éxito", f"Imagen guardada exitosamente en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar la imagen: {str(e)}")
    
    def save_control_states(self):
        """Guarda el estado actual de los controles y la imagen procesada"""
        if self.dialog_context.current_image_name in self.dialog_context.image_conversations:
            # Guardar estados de controles
            control_states = {
                "brightness": self.brightness_var.get(),
                "contrast": self.contrast_var.get(),
                "blur": self.blur_var.get(),
                "sharpen": self.sharpen_var.get(),
                "rotation": self.rotation_var.get(),
                "grayscale": self.grayscale_var.get(),
                "flip_h": self.flip_h,
                "flip_v": self.flip_v
            }
            self.dialog_context.image_conversations[self.dialog_context.current_image_name]["control_states"] = control_states
            
            # Guardar imagen procesada en base64
            if self.processed_image is not None:
                _, buffer = cv2.imencode('.jpg', self.processed_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                self.dialog_context.image_conversations[self.dialog_context.current_image_name]["processed_image"] = img_base64
    
    def load_control_states(self):
        """Carga el estado de los controles y la imagen procesada"""
        if self.dialog_context.current_image_name in self.dialog_context.image_conversations:
            conv_data = self.dialog_context.image_conversations[self.dialog_context.current_image_name]
            control_states = conv_data.get("control_states", {})
            
            if control_states:
                # Cargar estados de controles
                self.brightness_var.set(control_states.get("brightness", 0))
                self.contrast_var.set(control_states.get("contrast", 1.0))
                self.blur_var.set(control_states.get("blur", 0))
                self.sharpen_var.set(control_states.get("sharpen", 0.0))
                self.rotation_var.set(control_states.get("rotation", 0))
                self.grayscale_var.set(control_states.get("grayscale", False))
                self.flip_h = control_states.get("flip_h", False)
                self.flip_v = control_states.get("flip_v", False)
                
                # Actualizar labels
                self.brightness_label.config(text=f"Brillo: {self.brightness_var.get()}")
                self.contrast_label.config(text=f"Contraste: {self.contrast_var.get():.2f}")
                self.blur_label.config(text=f"Desenfoque: {self.blur_var.get()}")
                self.sharpen_label.config(text=f"Nitidez: {self.sharpen_var.get():.1f}")
                self.rotation_label.config(text=f"Rotación: {self.rotation_var.get()}°")
            
            # Cargar imagen procesada si existe
            processed_img_b64 = conv_data.get("processed_image")
            if processed_img_b64:
                img_data = base64.b64decode(processed_img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                self.processed_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Si no hay imagen procesada guardada, aplicar ediciones
                self.apply_all_edits()
    
    def analyze_image(self):
        """Analiza la imagen con el agente IA"""
        if self.dialog_context.current_image_data is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar una imagen")
            return
        
        self.add_message("Sistema", "Analizando imagen...", "system")
        self.send_button.config(state=tk.DISABLED)
        
        # Ejecutar en un hilo separado para no bloquear la UI
        thread = threading.Thread(target=self._analyze_image_thread)
        thread.daemon = True
        thread.start()
    
    def _analyze_image_thread(self):
        """Hilo para analizar la imagen"""
        try:
            # Codificar imagen original
            img_base64_original = base64.b64encode(self.dialog_context.current_image_data).decode('utf-8')
            
            # Obtener valores actuales de los controles
            control_info = f"""\n\nVALORES ACTUALES DE LOS CONTROLES DEL EDITOR:
- Brillo: {self.brightness_var.get()}
- Contraste: {self.contrast_var.get():.2f}
- Desenfoque: {self.blur_var.get()}
- Nitidez: {self.sharpen_var.get():.1f}
- Rotación: {self.rotation_var.get()}°
- Escala de grises: {'Activada' if self.grayscale_var.get() else 'Desactivada'}
- Volteo horizontal: {'Sí' if self.flip_h else 'No'}
- Volteo vertical: {'Sí' if self.flip_v else 'No'}
"""
            
            # Codificar imagen procesada si existe y es diferente
            content_parts = [{"type": "text", "text": VISION_PROMPT + control_info}]
            
            # Agregar imagen original
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64_original}"
                }
            })
            
            # Agregar imagen procesada si existe
            if self.processed_image is not None:
                # Convertir imagen procesada a bytes
                _, buffer = cv2.imencode('.jpg', self.processed_image)
                img_base64_processed = base64.b64encode(buffer).decode('utf-8')
                content_parts.append({
                    "type": "text",
                    "text": "Esta es la imagen después de las ediciones del usuario:"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64_processed}"
                    }
                })
            
            message = HumanMessage(content=content_parts)
            
            response = llm.invoke([message])
            
            if response.content:
                self.dialog_context.add_to_history(True, response.content)
                self.root.after(0, lambda: self._process_agent_response(response.content))
            else:
                self.root.after(0, lambda: self.add_message("Sistema", "No se pudo generar una descripción", "system"))
        
        except Exception as e:
            # Capturar el error en una variable local para la lambda
            error_text = str(e)
            self.root.after(0, lambda err=error_text: self.add_message("Sistema", f"❌ Error: {err}", "system"))
        
        finally:
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
    
    def _process_agent_response(self, response_text):
        """Procesa la respuesta del agente (solo muestra sugerencias)"""
        # Mostrar respuesta del agente
        self.add_message("Asistente", response_text, "assistant")
    
    def send_message(self):
        """Envía un mensaje del usuario al agente"""
        message = self.message_entry.get().strip()
        
        if not message:
            return
        
        if self.dialog_context.current_image_data is None:
            messagebox.showwarning("Advertencia", "Primero debes cargar una imagen")
            return
        
        # Mostrar mensaje del usuario
        self.add_message("Tú", message, "user")
        self.message_entry.delete(0, tk.END)
        self.send_button.config(state=tk.DISABLED)
        
        # Mostrar indicador de procesamiento
        self.add_message("Sistema", "⏳ Procesando mensaje...", "system")
        
        # Procesar respuesta en hilo separado
        thread = threading.Thread(target=self._process_message_thread, args=(message,))
        thread.daemon = True
        thread.start()
    
    def _process_message_thread(self, user_message):
        """Hilo para procesar el mensaje del usuario"""
        try:
            self.dialog_context.add_to_history(False, user_message)
            
            # Obtener valores actuales de los controles
            control_info = f"""\n\nVALORES ACTUALES DE LOS CONTROLES DEL EDITOR:
- Brillo: {self.brightness_var.get()}
- Contraste: {self.contrast_var.get():.2f}
- Desenfoque: {self.blur_var.get()}
- Nitidez: {self.sharpen_var.get():.1f}
- Rotación: {self.rotation_var.get()}°
- Escala de grises: {'Activada' if self.grayscale_var.get() else 'Desactivada'}
- Volteo horizontal: {'Sí' if self.flip_h else 'No'}
- Volteo vertical: {'Sí' if self.flip_v else 'No'}
"""
            
            # Usar solo el contexto de la imagen actual
            # Usar replace en lugar de format para evitar problemas con llaves {} en el contexto
            context_string = self.dialog_context.get_context_string()
            prompt_with_context = DIALOG_PROMPT.replace("{context}", context_string).replace("{user_input}", user_message) + control_info
            
            # Codificar imagen original
            img_base64_original = base64.b64encode(self.dialog_context.current_image_data).decode('utf-8')
            
            # Construir contenido del mensaje
            content_parts = [{"type": "text", "text": prompt_with_context}]
            
            # Agregar imagen original
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64_original}"
                }
            })
            
            # Agregar imagen procesada si existe
            if self.processed_image is not None:
                _, buffer = cv2.imencode('.jpg', self.processed_image)
                img_base64_processed = base64.b64encode(buffer).decode('utf-8')
                content_parts.append({
                    "type": "text",
                    "text": "Esta es la versión editada actual:"
                })
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64_processed}"
                    }
                })
            
            message = HumanMessage(content=content_parts)
            
            response = llm.invoke([message])
            
            if response.content:
                self.dialog_context.add_to_history(True, response.content)
                self.root.after(0, lambda: self._process_agent_response(response.content))
            else:
                self.root.after(0, lambda: self.add_message("Sistema", "No se pudo generar una respuesta", "system"))
        
        except Exception as e:
            # Mostrar error detallado con traceback
            import traceback
            error_msg = f"Error al procesar mensaje:\n{str(e)}\n\nDetalles técnicos:\n{traceback.format_exc()}"
            print(error_msg)  # Imprimir en consola para debug
            # Capturar el error en una variable local para la lambda
            error_text = str(e)
            self.root.after(0, lambda err=error_text: self.add_message("Sistema", f"❌ Error: {err}", "system"))
        
        finally:
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
    
    def save_conversation(self):
        """Guarda la conversación en formato JSON"""
        if not self.dialog_context.image_conversations:
            messagebox.showwarning("Advertencia", "No hay conversación para guardar")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Guardar Conversación"
        )
        
        if file_path:
            success, message = self.dialog_context.save_conversation_to_json(file_path)
            if success:
                self.add_message("Sistema", message, "system")
                messagebox.showinfo("Éxito", message)
            else:
                self.add_message("Sistema", message, "system")
                messagebox.showerror("Error", message)
    
    def load_conversation(self):
        """Carga una conversación desde un archivo JSON"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Cargar Conversación"
        )
        
        if file_path:
            success, message = self.dialog_context.load_conversation_from_json(file_path)
            
            if success:
                self.add_message("Sistema", f"✓ {message}", "system")
                self.add_message("Sistema", f"Imágenes cargadas: {', '.join(self.dialog_context.get_all_images())}", "system")
                
                # Cargar la imagen actual si existe
                if self.dialog_context.current_image_data:
                    # Convertir bytes a imagen OpenCV
                    nparr = np.frombuffer(self.dialog_context.current_image_data, np.uint8)
                    self.original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if self.original_image is not None:
                        # Cargar estados de controles
                        self.load_control_states()
                        
                        # Mostrar ambas imágenes
                        self.display_images()
                        
                        # Actualizar etiqueta
                        self.image_label.config(
                            text=f"Imagen actual: {self.dialog_context.current_image_name}", 
                            foreground="blue"
                        )
                        
                        # Mostrar historial de la imagen actual
                        messages = self.dialog_context.get_current_messages()
                        if messages:
                            self.add_message("Sistema", f"--- Historial de {self.dialog_context.current_image_name} ---", "system")
                            for msg in messages:
                                if isinstance(msg, AIMessage):
                                    self.add_message("Asistente", msg.content, "assistant")
                                else:
                                    self.add_message("Tú", msg.content, "user")
                
                messagebox.showinfo("Éxito", message)
            else:
                self.add_message("Sistema", f"✗ {message}", "system")
                messagebox.showerror("Error", message)

def main():
    root = tk.Tk()
    app = ImageAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
