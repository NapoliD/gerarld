 Recomendación de Enfoque

  🎯 Estrategia General

  Este desafío evalúa 3 pilares: Analytics (30%), Modeling (30%), y Code Quality (30%). Te recomiendo balancear tu tiempo entre ellos y enfocarte en la
  simplicidad y claridad sobre la complejidad.

  🗓️ Etapas Recomendadas (2-3 horas)

  Etapa 1: Setup & Exploración (30 min)

  1. Descargar y revisar los datos
    - Priorizar: orders, order_items, products, customers
    - Explorar estructura, missing values, relaciones entre tablas
  2. Setup del proyecto
    - Crear estructura de carpetas según el template
    - Configurar virtual environment
    - requirements.txt básico: pandas, numpy, scikit-learn, pytest

  Etapa 2: Analytics (45 min)

  Objetivos clave:
  - Top categorías por órdenes y GMV (Gross Merchandise Value)
  - Repeat purchase rate
  - Tiempo promedio entre órdenes
  - Review score distribution

  Tips:
  - Usar joins simples entre dataframes
  - Calcular métricas agregadas
  - Identificar 2 insights no obvios con impacto de negocio
  - Ejemplo de insight: "70% de clientes nunca repiten compra → oportunidad de retención"

  Etapa 3: Modeling (45-60 min)

  Recomendación: Empieza con RECOMMENDATION (más sencillo que prediction)

  Approach sugerido:
  1. Baseline simple: Popularity-based (productos más vendidos)
  2. Mejora: Co-purchase (clientes que compraron X también compraron Y)
  3. Métrica: Precision@K o MAP@K
  4. Evaluación: Train/test split por fecha o por cliente

  Estructura de código:
  src/
  ├── data_loader.py    # Cargar CSVs
  ├── model.py          # RecommenderModel class
  ├── evaluate.py       # precision_at_k()
  └── main.py           # CLI

  Etapa 4: Production Code (30 min)

  1. CLI funcional:
  python -m src.main --customer_id <ID> --top_k 5
  2. Un test simple:
  def test_model_returns_correct_number():
      model = RecommenderModel()
      recs = model.recommend(customer_id, top_k=5)
      assert len(recs) == 5
  3. Guardar modelo:
    - Pickle del modelo o JSON de la matriz de co-purchase

  Etapa 5: Documentación (15 min)

  1. README.md: Setup, cómo correr, cómo testear
  2. Analytics summary (1 página): KPIs + 2 insights + visualizaciones simples

  ✨ Diferenciadores Clave

  Para destacar:
  - ✅ Código modular con clases bien definidas
  - ✅ Insights con impacto de negocio claro
  - ✅ Métrica de evaluación bien justificada
  - ✅ Tests que validen lógica crítica
  - ✅ README claro y reproducible

  Evitar:
  - ❌ Notebooks como única entrega
  - ❌ Modelos complejos sin baseline
  - ❌ Sobre-ingeniería (KISS principle)

  🚀 Quick Wins

  1. Analytics: Gráfico de distribución de review scores + tabla de top categorías
  2. Model: Baseline popularity + co-purchase matrix
  3. Code: Estructura según template + 2-3 tests básicos
  4. Docs: README con comandos copy-paste + summary con bullets claros

  ¿Quieres que te ayude a empezar con alguna etapa específica? Puedo ayudarte a:
  - Crear la estructura del proyecto
  - Explorar los datos inicialmente
  - Implementar el modelo de recomendación
  - Configurar los tests