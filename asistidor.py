import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, Any, Optional, List
import io
import logging

# Configuración de la página
st.set_page_config(
    page_title="CSV Analyzer con Groq - Mejorado",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #4f46e5;
    }
    .calculation-result {
        background: #ecfdf5;
        border: 2px solid #a7f3d0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .error-message {
        background: #fee2e2;
        border: 1px solid #fecaca;
        color: #dc2626;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-message {
        background: #d1fae5;
        border: 1px solid #a7f3d0;
        color: #059669;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CSVAnalyzer:
    def __init__(self):
        self.data = None
        self.headers = []
        
    def load_csv(self, uploaded_file) -> bool:
        """Cargar y procesar archivo CSV con manejo robusto de errores"""
        try:
            # Resetear el puntero del archivo
            uploaded_file.seek(0)
            
            # Lista de separadores a intentar
            separators = [',', ';', '|', '\t']
            
            # Lista de codificaciones a intentar
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
            
            best_result = None
            best_score = 0
            successful_config = None
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        # Resetear el puntero del archivo para cada intento
                        uploaded_file.seek(0)
                        
                        # Intentar leer el CSV
                        df = pd.read_csv(
                            uploaded_file,
                            sep=sep,
                            encoding=encoding,
                            on_bad_lines='skip',
                            low_memory=False,
                            skipinitialspace=True
                        )
                        
                        # Evaluar la calidad del resultado
                        score = 0
                        
                        # Puntos por tener múltiples columnas
                        if len(df.columns) > 1:
                            score += len(df.columns) * 10
                        
                        # Puntos por tener datos
                        if len(df) > 0:
                            score += len(df)
                        
                        # Penalizar si todas las columnas están en una sola
                        if len(df.columns) == 1 and len(df) > 0:
                            first_values = df.iloc[:5, 0].astype(str)
                            if any(sep in str(val) for val in first_values):
                                score = score // 2  # Penalizar este resultado
                        
                        # Si este resultado es mejor, guardarlo
                        if score > best_score:
                            best_score = score
                            best_result = df.copy()
                            successful_config = (sep, encoding)
                            
                    except Exception as e:
                        # Continuar con la siguiente combinación
                        continue
            
            # Si encontramos un resultado válido
            if best_result is not None and best_score > 0:
                self.data = best_result
                
                # Limpiar nombres de columnas
                self.data.columns = [str(col).strip() for col in self.data.columns]
                self.headers = self.data.columns.tolist()
                
                # Información de éxito
                sep_used, enc_used = successful_config
                st.success(f"✅ Archivo cargado exitosamente!")
                st.info(f"📝 Configuración: Separador '{sep_used}', Codificación: {enc_used}")
                
                return True
            else:
                st.error("❌ No se pudo cargar el archivo CSV. Posibles causas:")
                st.write("• El archivo no es un CSV válido")
                st.write("• El separador no es estándar")
                st.write("• Problemas de codificación")
                st.write("• El archivo está vacío o corrupto")
                return False
                
        except Exception as e:
            st.error(f"❌ Error crítico al cargar el archivo: {str(e)}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Generar resumen estadístico detallado"""
        if self.data is None:
            return {}
        
        summary = {}
        
        for column in self.headers:
            try:
                values = self.data[column].dropna()
                non_empty_count = len(values)
                empty_count = len(self.data) - non_empty_count
                
                if non_empty_count > 0:
                    # Intentar convertir a numérico
                    numeric_values = pd.to_numeric(values, errors='coerce').dropna()
                    
                    if len(numeric_values) > 0 and len(numeric_values) == len(values):
                        # Análisis numérico
                        summary[column] = {
                            'type': 'NUMÉRICO',
                            'valid_count': non_empty_count,
                            'empty_count': empty_count,
                            'min': float(numeric_values.min()),
                            'max': float(numeric_values.max()),
                            'mean': float(numeric_values.mean()),
                            'median': float(numeric_values.median()),
                            'sum': float(numeric_values.sum()),
                            'std': float(numeric_values.std()) if len(numeric_values) > 1 else 0
                        }
                    else:
                        # Análisis categórico
                        unique_values = values.unique()
                        most_common = values.value_counts().head(1)
                        
                        summary[column] = {
                            'type': 'TEXTO',
                            'valid_count': non_empty_count,
                            'empty_count': empty_count,
                            'unique_count': len(unique_values),
                            'most_common_value': most_common.index[0] if len(most_common) > 0 else None,
                            'most_common_count': most_common.iloc[0] if len(most_common) > 0 else 0,
                            'examples': [str(x) for x in unique_values[:3]]
                        }
            except Exception as e:
                # Si hay error con una columna específica, continuar con las demás
                st.warning(f"Error procesando columna '{column}': {str(e)}")
                continue
        
        return summary
    
    def detect_column(self, question: str) -> Optional[str]:
        """Detectar columna mencionada en la pregunta"""
        question_words = question.lower().split()
        
        # Buscar coincidencias exactas
        for header in self.headers:
            if header.lower() in question_words:
                return header
        
        # Buscar coincidencias parciales
        for header in self.headers:
            for word in question_words:
                if len(word) > 2 and (header.lower() in word or word in header.lower()):
                    return header
        
        return None
    
    def calculate_average(self, question: str) -> Optional[Dict[str, Any]]:
        """Calcular promedio"""
        column = self.detect_column(question)
        if not column or column not in self.data.columns:
            return None
        
        values = pd.to_numeric(self.data[column], errors='coerce').dropna()
        
        if len(values) == 0:
            return None
        
        return {
            'operation': 'Promedio',
            'column': column,
            'result': round(float(values.mean()), 2),
            'count': len(values),
            'details': f'Calculado sobre {len(values)} valores numéricos válidos'
        }
    
    def calculate_sum(self, question: str) -> Optional[Dict[str, Any]]:
        """Calcular suma"""
        column = self.detect_column(question)
        if not column or column not in self.data.columns:
            return None
        
        values = pd.to_numeric(self.data[column], errors='coerce').dropna()
        
        if len(values) == 0:
            return None
        
        return {
            'operation': 'Suma',
            'column': column,
            'result': round(float(values.sum()), 2),
            'count': len(values),
            'details': f'Suma de {len(values)} valores numéricos válidos'
        }
    
    def calculate_max(self, question: str) -> Optional[Dict[str, Any]]:
        """Calcular máximo"""
        column = self.detect_column(question)
        if not column or column not in self.data.columns:
            return None
        
        values = pd.to_numeric(self.data[column], errors='coerce').dropna()
        
        if len(values) == 0:
            return None
        
        return {
            'operation': 'Máximo',
            'column': column,
            'result': float(values.max()),
            'count': len(values),
            'details': f'Valor máximo de {len(values)} valores numéricos válidos'
        }
    
    def calculate_min(self, question: str) -> Optional[Dict[str, Any]]:
        """Calcular mínimo"""
        column = self.detect_column(question)
        if not column or column not in self.data.columns:
            return None
        
        values = pd.to_numeric(self.data[column], errors='coerce').dropna()
        
        if len(values) == 0:
            return None
        
        return {
            'operation': 'Mínimo',
            'column': column,
            'result': float(values.min()),
            'count': len(values),
            'details': f'Valor mínimo de {len(values)} valores numéricos válidos'
        }
    
    def calculate_count(self, question: str) -> Optional[Dict[str, Any]]:
        """Calcular conteo"""
        column = self.detect_column(question)
        
        if column and column in self.data.columns:
            count = self.data[column].notna().sum()
            return {
                'operation': 'Conteo',
                'column': column,
                'result': int(count),
                'details': f'{count} valores no vacíos en la columna "{column}"'
            }
        else:
            return {
                'operation': 'Conteo total',
                'result': len(self.data),
                'details': 'Total de filas en el dataset'
            }
    
    def perform_group_by(self, question: str) -> Optional[Dict[str, Any]]:
        """Realizar agrupación"""
        column = self.detect_column(question)
        if not column or column not in self.data.columns:
            return None
        
        try:
            groups = self.data.groupby(column).size().to_dict()
            
            return {
                'operation': 'Agrupación',
                'column': column,
                'result': groups,
                'count': len(groups),
                'details': f'Datos agrupados por "{column}" en {len(groups)} grupos'
            }
        except Exception as e:
            st.error(f"Error en agrupación: {str(e)}")
            return None
    
    def perform_local_calculation(self, question: str) -> Optional[Dict[str, Any]]:
        """Realizar cálculo local según la pregunta"""
        question_lower = question.lower()
        
        try:
            if any(word in question_lower for word in ['promedio', 'media']):
                return self.calculate_average(question)
            elif any(word in question_lower for word in ['suma', 'total']):
                return self.calculate_sum(question)
            elif any(word in question_lower for word in ['máximo', 'mayor']):
                return self.calculate_max(question)
            elif any(word in question_lower for word in ['mínimo', 'menor']):
                return self.calculate_min(question)
            elif any(word in question_lower for word in ['contar', 'cantidad']):
                return self.calculate_count(question)
            elif any(word in question_lower for word in ['agrupar', 'por']):
                return self.perform_group_by(question)
        except Exception as e:
            st.error(f"Error en cálculo local: {str(e)}")
            return None
        
        return None
    
    def get_relevant_data_sample(self, question: str, sample_size: int = 15) -> List[Dict]:
        """Obtener muestra relevante de datos"""
        if self.data is None:
            return []
        
        detected_column = self.detect_column(question)
        
        try:
            if detected_column and detected_column in self.data.columns:
                # Mostrar muestra relevante con la columna detectada
                columns_to_show = [detected_column]
                
                # Agregar algunas columnas más para contexto
                for col in self.headers[:3]:
                    if col != detected_column and col not in columns_to_show:
                        columns_to_show.append(col)
                
                sample_data = self.data[columns_to_show].head(sample_size)
            else:
                # Muestra general
                sample_data = self.data.head(min(sample_size, 10))
            
            return sample_data.to_dict('records')
        except Exception as e:
            st.error(f"Error obteniendo muestra de datos: {str(e)}")
            return []

class GroqAPI:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def ask_question(self, prompt: str) -> str:
        """Enviar pregunta a Groq API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.1,
            'max_tokens': 1500
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error de API: {str(e)}")
        except KeyError as e:
            raise Exception(f"Formato de respuesta inesperado: {str(e)}")

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🤖 CSV Analyzer con Groq - Mejorado</h1>
        <p>Análisis preciso de datos con cálculos locales y respuestas exactas</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar el analizador
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CSVAnalyzer()
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración de Groq")
        
        api_key = st.text_input(
            "API Key de Groq:",
            type="password",
            help="Ingresa tu API Key de Groq"
        )
        
        model = st.selectbox(
            "Modelo:",
            [
                "llama3-8b-8192",
                "llama3-70b-8192", 
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ],
            index=0
        )
        
        st.header("📊 Modo de análisis")
        analysis_mode = st.radio(
            "Selecciona el modo:",
            [
                "Híbrido (Cálculo local + IA) - Más preciso",
                "Solo IA - Más rápido pero menos preciso"
            ],
            index=0
        )
        
        # Información de depuración
        if st.session_state.analyzer.data is not None:
            st.header("🔍 Información de depuración")
            st.write(f"Columnas detectadas: {len(st.session_state.analyzer.headers)}")
            st.write("Nombres de columnas:")
            for i, col in enumerate(st.session_state.analyzer.headers):
                st.write(f"{i+1}. {col}")
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Cargar archivo CSV")
        
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV",
            type=['csv'],
            help="Soporta archivos CSV con cualquier delimitador"
        )
        
        if uploaded_file is not None:
            with st.spinner("Procesando archivo..."):
                if st.session_state.analyzer.load_csv(uploaded_file):
                    st.success(f"✅ Archivo cargado exitosamente: {len(st.session_state.analyzer.data)} filas, {len(st.session_state.analyzer.headers)} columnas")
                    
                    # Mostrar estadísticas
                    st.subheader("📊 Estadísticas del Dataset")
                    
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number">{len(st.session_state.analyzer.data)}</div>
                            <div>Filas</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stats_col2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number">{len(st.session_state.analyzer.headers)}</div>
                            <div>Columnas</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stats_col3:
                        try:
                            file_size = round(len(str(st.session_state.analyzer.data.to_json())) / 1024, 2)
                        except:
                            file_size = "N/A"
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-number">{file_size}</div>
                            <div>KB</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Vista previa de datos
                    st.subheader("👀 Vista Previa")
                    try:
                        st.dataframe(
                            st.session_state.analyzer.data.head(),
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error mostrando vista previa: {str(e)}")
                        st.write("Datos cargados pero no se pueden mostrar en formato tabla.")
                    
                else:
                    st.error("❌ Error al procesar el archivo CSV. Verifica el formato.")
    
    with col2:
        if st.session_state.analyzer.data is not None:
            st.header("💬 Análisis de Datos")
            
            # Input para preguntas
            question = st.text_area(
                "Haz una pregunta sobre tus datos:",
                placeholder="Ej: ¿Cuál es el promedio de ventas por mes?",
                height=100
            )
            
            if st.button("🔍 Analizar", type="primary"):
                if not question.strip():
                    st.error("Por favor, ingresa una pregunta.")
                elif not api_key:
                    st.error("Por favor, ingresa tu API Key de Groq.")
                else:
                    with st.spinner("Analizando datos..."):
                        try:
                            # Inicializar API de Groq
                            groq_api = GroqAPI(api_key, model)
                            
                            # Realizar cálculo local si está en modo híbrido
                            local_calculation = None
                            if "Híbrido" in analysis_mode:
                                local_calculation = st.session_state.analyzer.perform_local_calculation(question)
                            
                            # Generar resumen de datos
                            data_summary = st.session_state.analyzer.get_data_summary()
                            relevant_data = st.session_state.analyzer.get_relevant_data_sample(question)
                            
                            # Crear prompt mejorado
                            summary_text = ""
                            for col, info in data_summary.items():
                                if info['type'] == 'NUMÉRICO':
                                    summary_text += f"{col}: NUMÉRICO\n"
                                    summary_text += f"  - Valores válidos: {info['valid_count']}\n"
                                    summary_text += f"  - Valores vacíos: {info['empty_count']}\n"
                                    summary_text += f"  - Mínimo: {info['min']}\n"
                                    summary_text += f"  - Máximo: {info['max']}\n"
                                    summary_text += f"  - Promedio: {info['mean']:.2f}\n"
                                    summary_text += f"  - Mediana: {info['median']}\n"
                                    summary_text += f"  - Suma total: {info['sum']:.2f}\n\n"
                                else:
                                    summary_text += f"{col}: TEXTO\n"
                                    summary_text += f"  - Valores válidos: {info['valid_count']}\n"
                                    summary_text += f"  - Valores vacíos: {info['empty_count']}\n"
                                    summary_text += f"  - Valores únicos: {info['unique_count']}\n"
                                    summary_text += f"  - Valor más común: \"{info['most_common_value']}\" ({info['most_common_count']} veces)\n"
                                    summary_text += f"  - Ejemplos: {', '.join([f'\"{ex}\"' for ex in info['examples']])}\n\n"
                            
                            prompt = f"""
Eres un analista de datos experto. Analiza el siguiente dataset CSV:

INFORMACIÓN DEL DATASET:
- Columnas disponibles: {', '.join(st.session_state.analyzer.headers)}
- Total de filas: {len(st.session_state.analyzer.data)}
- Resumen estadístico detallado:
{summary_text}

MUESTRA DE DATOS RELEVANTES:
{json.dumps(relevant_data, indent=2, ensure_ascii=False)}
"""

                            if local_calculation:
                                prompt += f"""

CÁLCULO LOCAL REALIZADO:
Operación: {local_calculation['operation']}
Columna: {local_calculation.get('column', 'N/A')}
Resultado: {local_calculation['result']}
Detalles: {local_calculation['details']}

IMPORTANTE: Usa este cálculo local como base para tu respuesta, ya que es exacto y preciso.
"""

                            prompt += f"""

PREGUNTA DEL USUARIO: {question}

INSTRUCCIONES:
1. {'Usa el cálculo local proporcionado como resultado principal' if local_calculation else 'Realiza el análisis basándote en los datos disponibles'}
2. Proporciona una respuesta clara y precisa
3. Si haces estimaciones, menciona que son aproximadas
4. Si no puedes responder con certeza, explica por qué
5. Responde en español de manera profesional

Respuesta:"""

                            # Obtener respuesta de Groq
                            answer = groq_api.ask_question(prompt)
                            
                            # Mostrar resultados
                            st.subheader("❓ Pregunta:")
                            st.write(question)
                            
                            st.subheader("🤖 Respuesta:")
                            st.write(answer)
                            
                            # Mostrar cálculo local si existe
                            if local_calculation:
                                st.markdown("""
                                <div class="calculation-result">
                                    <h4>🔢 Cálculo Local Exacto:</h4>
                                """, unsafe_allow_html=True)
                                
                                st.write(f"**Operación:** {local_calculation['operation']}")
                                st.write(f"**Columna:** {local_calculation.get('column', 'N/A')}")
                                st.write(f"**Resultado:** {local_calculation['result']}")
                                st.write(f"**Detalles:** {local_calculation['details']}")
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error al procesar la pregunta: {str(e)}")
        else:
            st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")

if __name__ == "__main__":
    main()