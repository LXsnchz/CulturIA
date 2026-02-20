"""
CulturIA — Generador de Trivia Diaria sobre Cultura General Española
=====================================================================
Este script se ejecuta diariamente vía GitHub Actions.
Usa la API de Groq (modelo llama-3.3-70b-versatile) para generar
3 preguntas nuevas y un "mensaje de burla" diario.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta

try:
    from groq import Groq
except ImportError:
    print("Error: el paquete 'groq' no está instalado.")
    print("Ejecuta: pip install groq")
    sys.exit(1)


# ── Configuración ───────────────────────────────────────────────────
HISTORIAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historial.json")
GROQ_MODEL = "llama-3.3-70b-versatile"
MADRID_TZ = timezone(timedelta(hours=1))  # CET (en CEST sería +2)
MAX_CONTEXT_QUESTIONS = 60  # Últimas N preguntas enviadas a la IA para evitar repeticiones


def load_historial() -> list:
    """Carga el historial existente o devuelve una lista vacía."""
    if os.path.exists(HISTORIAL_PATH):
        with open(HISTORIAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_historial(data: list) -> None:
    """Guarda el historial actualizado en el JSON."""
    with open(HISTORIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_previous_questions(historial: list, max_questions: int = MAX_CONTEXT_QUESTIONS) -> list[str]:
    """Extrae las últimas preguntas del historial para contexto anti-repetición."""
    questions = []
    for day in historial:
        for q in day.get("preguntas", []):
            questions.append(q["pregunta"])
            if len(questions) >= max_questions:
                return questions
    return questions


def build_prompt(previous_questions: list[str], dia_number: int) -> str:
    """Construye el prompt para la IA."""
    prev_q_text = "\n".join(f"  - {q}" for q in previous_questions) if previous_questions else "  (ninguna todavía)"

    return f"""Eres un experto en cultura general española. Tu trabajo es generar preguntas de trivia divertidas, variadas y educativas sobre España.

INSTRUCCIONES ESTRICTAS:
1. Genera exactamente 3 preguntas nuevas sobre cultura general española.
2. Las preguntas deben cubrir temas variados: historia, geografía, gastronomía, arte, música, deportes, ciencia, tradiciones, lengua, etc.
3. Cada pregunta puede ser:
   - Tipo test con 3 opciones (solo 1 correcta)
   - Tipo Verdadero/Falso (2 opciones: "Verdadero" y "Falso")
4. Las preguntas deben ser DIFERENTES a las siguientes preguntas ya existentes:
{prev_q_text}
5. Genera también un "mensaje de burla" diario: una frase graciosa, ingeniosa y picante (con emojis) para mostrar al usuario que falla. Debe ser divertida pero no ofensiva. Estilo humor español.

FORMATO DE RESPUESTA (JSON estricto, sin bloques de código markdown):
{{
  "preguntas": [
    {{
      "pregunta": "texto de la pregunta",
      "opciones": ["opción 1", "opción 2", "opción 3"],
      "respuesta_correcta": 0
    }},
    {{
      "pregunta": "texto de la pregunta V/F",
      "opciones": ["Verdadero", "Falso"],
      "respuesta_correcta": 1
    }},
    {{
      "pregunta": "texto de la pregunta",
      "opciones": ["opción 1", "opción 2", "opción 3"],
      "respuesta_correcta": 2
    }}
  ],
  "mensaje_burla": "¡Frase graciosa con emojis! 😂🥘"
}}

IMPORTANTE:
- "respuesta_correcta" es el ÍNDICE (0, 1 o 2) de la opción correcta.
- Devuelve SOLO el JSON, sin texto extra ni bloques de código.
- Las preguntas deben ser factualmente correctas.
- Este es el día #{dia_number} de CulturIA.
"""


def parse_response(text: str) -> dict:
    """Parsea la respuesta de la IA y valida la estructura."""
    # Limpiar posibles bloques de código markdown
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remover ```json y ``` final
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    data = json.loads(cleaned)

    # Validar estructura
    assert "preguntas" in data, "Falta el campo 'preguntas'"
    assert "mensaje_burla" in data, "Falta el campo 'mensaje_burla'"
    assert len(data["preguntas"]) == 3, f"Se esperaban 3 preguntas, se recibieron {len(data['preguntas'])}"

    for i, q in enumerate(data["preguntas"]):
        assert "pregunta" in q, f"Pregunta {i}: falta 'pregunta'"
        assert "opciones" in q, f"Pregunta {i}: falta 'opciones'"
        assert "respuesta_correcta" in q, f"Pregunta {i}: falta 'respuesta_correcta'"
        assert len(q["opciones"]) in (2, 3), f"Pregunta {i}: debe tener 2 o 3 opciones"
        assert isinstance(q["respuesta_correcta"], int), f"Pregunta {i}: 'respuesta_correcta' debe ser un entero"
        assert 0 <= q["respuesta_correcta"] < len(q["opciones"]), f"Pregunta {i}: índice fuera de rango"

    return data


def generate_trivia() -> None:
    """Flujo principal de generación de trivia."""
    # 1. Verificar API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: La variable de entorno GROQ_API_KEY no está definida.")
        sys.exit(1)

    # ... (mismo código cargar historial y fecha) ...
    historial = load_historial()
    print(f"📂 Historial cargado: {len(historial)} días existentes.")

    dia_number = (historial[0]["dia"] + 1) if historial else 1
    fecha_hoy = datetime.now(MADRID_TZ).strftime("%Y-%m-%d")

    if historial and historial[0].get("fecha") == fecha_hoy:
        print(f"⚠️  Ya existe un registro para hoy ({fecha_hoy}). Saltando generación.")
        return

    prev_questions = get_previous_questions(historial)
    print(f"🧠 Contexto anti-repetición: {len(prev_questions)} preguntas anteriores.")

    prompt = build_prompt(prev_questions, dia_number)

    print(f"🤖 Llamando a Groq ({GROQ_MODEL})...")
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that strictly outputs JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    raw_text = response.choices[0].message.content
    print(f"📥 Respuesta recibida ({len(raw_text)} caracteres).")

    # 6. Parsear y validar
    try:
        new_data = parse_response(raw_text)
    except (json.JSONDecodeError, AssertionError) as e:
        print(f"❌ Error al parsear la respuesta de la IA: {e}")
        print(f"   Respuesta cruda:\n{raw_text}")
        sys.exit(1)

    # 7. Crear entrada del nuevo día
    new_day = {
        "dia": dia_number,
        "fecha": fecha_hoy,
        "preguntas": new_data["preguntas"],
        "mensaje_burla": new_data["mensaje_burla"],
    }

    # 8. Insertar al principio y guardar
    historial.insert(0, new_day)
    save_historial(historial)

    print(f"✅ Día #{dia_number} ({fecha_hoy}) generado correctamente.")
    print(f"   📝 Preguntas:")
    for q in new_data["preguntas"]:
        print(f"      - {q['pregunta']}")
    print(f"   🤡 Burla: {new_data['mensaje_burla']}")
    print(f"💾 historial.json actualizado ({len(historial)} días totales).")


if __name__ == "__main__":
    generate_trivia()
