# Revisión: código vs. paper

Fecha: 2026-11-03

- **Paper:** `3-escribir-paper/papers/How Multiple Explanations Behind a Decision Compose in Prescriptive Process Monitoring/`
- **Código:** este repositorio (`timing-explanations-pspm`)

**Veredicto:** el código y el paper coinciden en resultados y metodología. Hay algunas inexactitudes de redacción en el paper y dos problemas de reproducibilidad en el código, pero ninguno cambia un número.

La verificación se hizo sobre una copia del repo en una carpeta temporal, con un entorno nuevo (Python 3.14, torch 2.15 CPU, stable-baselines3 2.9.0, pandas 2.3.3, pyarrow). No se modificó nada del repo ni del paper.

## Resultados

### `numbers.tex`

Se regeneró con `paper_numbers.py` a partir de los JSON del repo. Las 443 macros coinciden con el `numbers.tex` del paper; solo difiere el salto de línea final.

### Re-ejecución de scripts

| Script | Resultado frente al JSON del repo |
|---|---|
| `fidelity_test.py` | idéntico bit a bit |
| `compute_gain_table.py` | idéntico bit a bit |
| `run_risk_suite.py --skip-plots` | idéntico (solo cambia la lista `figures`, vacía al omitir figuras) |
| `run_effect_suite.py --skip-plots` | idéntico (ídem) |
| `run_compose_suite.py --skip-plots` | idéntico (ídem) |
| `leakage_audit.py` | igual salvo `agent_trained_on_training_state` (ver "Problemas en el código") |

### Checkpoints y datos

- Los checkpoints del paper (`models/variants/ppo_<log>_cate_retrained.zip`) tienen el mismo SHA-256 que los elegidos en `models/variants/sweep_oof/`:
  - BPIC2017: `ent1.0`, 600k pasos, recompensa media en validación 47.83.
  - BPIC2012: `ent0.5`, 600k pasos, recompensa media en validación 50.67.
  - En ambos logs hubo 10 candidatos.
- Los manifests indican que el entrenamiento usó `retrained_state_<log>_train.csv` (estado fuera de fold), con `--random-resources`, `episode=case`, `reward_scale=0.01` y semilla 42.
- `data/retrained_states_sha256.txt`: los 9 CSV pasan `sha256sum -c`.

### Afirmaciones cualitativas del texto

Todas se cumplen frente a los números:

- La cuota de riesgo es la menor en ambos logs y en ambas decisiones.
- La brecha del deletion test se reduce con k=2, y más en las decisiones de esperar de BPIC2017.
- Los atributos nativos pesan más en BPIC2017 que en BPIC2012.
- `Activity` es la razón de riesgo más fuerte: mueve ambos brazos en la mayoría de las decisiones y pierde más de un tercio de su peso, a diferencia de `FirstWithdrawalAmount_max`.
- El acuerdo de signo entre riesgo y efecto está apenas por encima del 50 %.
- El camino de riesgo en path patching queda más cerca del azar en BPIC2017 que en BPIC2012.
- Las referencias están a 0.10 log-odds o menos entre sí.
- Sin la salvaguarda, el canal de riesgo sube a 29 % solo en las decisiones de actuar de BPIC2012.
- En el caso de ejemplo, riesgo y efecto nombran atributos distintos.

## Metodología: coincide con el código

| Paper | Código |
|---|---|
| Split temporal: la primera mitad de los casos entrena; la segunda se divide al azar en validación y test | `risk_model.temporal_split` (train_ratio 0.5, val_ratio 0.5, semilla 22, split "strict") |
| Early stopping del riesgo en validación (iteración 117 de 1000) | `eval_set=(Xva, …)`, `use_best_model=True`; manifest `best_iteration=117` |
| 68 atributos por prefijo | `models/risk/BPIC2017_manifest.json` `n_features=68` |
| Tratamiento = segunda oferta | `treatment_nth` (2.º O_Created / O_SENT) |
| Regla de efecto positivo p_T < 0.5 ≤ p_U, signo corregido | `compose.positive_effect_rule`, `desired_outcome` |
| Efecto: un modelo por brazo, ponderado por IPW, AUC de cada brazo en su propio brazo | `effect_model.py`; claves `auc_treated_arm_on_treated_test` y `auc_untreated_arm_on_untreated_test` |
| Estado de entrenamiento con 5 folds fuera de fold | `crossfit.py` |
| IG con 128 pasos, referencia = media del pool, −ΔQ en las decisiones de esperar | `dual_level.integrated_gradients(n_steps=128)`, `reference = states.mean(axis=0)`, `compose.timing_attribution` |
| Composición, Eq. 5 | `compose.propagate` y `_weights` |
| Ratio de cancelación c_j | `compose.cancellation` |
| Salvaguarda ε = 0.05 | `compose.FALLBACK_RATIO = 0.05` |
| Pool: un prefijo por caso de test, recursos ciclados de 0 a 3 | `pools.bpic_pool`, `cycled_resources` (semilla 123) |
| Deletion test contra 20 extracciones aleatorias | `dual_level.deletion_test(n_random=20)` |
| Fondo de 100 prefijos para TreeSHAP; 20 permutaciones para la explicación directa | `run_compose_suite.py --n-background 100 --n-perm 20` |

## Discrepancias en el texto del paper (menores)

1. **El ejemplo del abstract no cuadra con el del paper.** El abstract dice que la oferta se recomienda "por el monto del préstamo", que llega a la decisión por el modelo de efecto. En el caso de la Sección 6, la explicación compuesta pone primero `Activity` (+1.06). `RequestedAmount` aparece primero solo en la explicación directa (+0.69), y no está en el top 3 de riesgo (`Activity`, `LoanGoal`, `NumberOfTerms_sum`) ni en el de efecto (`open_cases_min`, `month_mean`, `FirstWithdrawalAmount_max`).
2. **`reliability` en BPIC2012** (Sección 3, step ii). El paper la describe como "su probabilidad", es decir, la probabilidad de la clase predicha. Es cierto en BPIC2017 (`max(r, 1−r)`), pero en BPIC2012 el código la iguala a `deviation` (`compose.risk_features_from_r`).
3. **Las cuotas λ** (Definición 5 frente a la Tabla 1). λ_L(s_t) se define para cada estado, pero la tabla usa un cociente agregado sobre el pool: media de |φ| por feature dividida por la suma (`compose.level_shares`, `fidelity_test.py`). No es el promedio de los λ_L(s_t).
4. **Propiedad de completitud.** Se enuncia como Σ_j φ^{ΔQ∘x}_j = ΔQ(s_t) − ΔQ(s̄). En las decisiones de esperar el total es −(ΔQ(s_t) − ΔQ(s̄)), porque se atribuye −ΔQ.
5. **Reward** (Sección 5, "elsewhere, waiting pays a bonus while an employee is free"). Es exacto cuando el efecto es nulo. Cuando el efecto es negativo, esperar paga +100 haya o no personal libre. Esto ocurre en el 1.2 % de las filas de test de BPIC2017 y en el 6.7 % de las de BPIC2012.
6. **Hiperparámetro en el borde de la rejilla.** En BPIC2017 la validación eligió entropía 1.0, el valor más alto probado. No sale en el PDF (el bloque de hiperparámetros está comentado), pero un revisor podría preguntar si un valor más alto sería aún mejor.

## Problemas en el código (no afectan a los números)

- **`requirements.txt` incompleto:**
  - Falta `pyarrow`. Sin él, `run_risk_suite.py`, `run_effect_suite.py` y `run_compose_suite.py` fallan al leer el parquet de Sepsis (`risk_model.load_sepsis_events`).
  - Permite pandas 3, que rompe `leakage_audit.py`. En la línea 138, `dtype == object` deja de detectar las columnas de texto y aparece `ValueError: could not convert string to float: 'True'`.
  - Arreglo: fijar `pandas<3` y añadir `pyarrow`.
- **`leakage_audit.py` fallaría hoy en el repo real.** En la línea 194, `agent_trained_on_training_state` compara rutas absolutas. Los manifests guardan la ruta antigua del repo (`.../process-mining/prescriptive-process-monitoring/timing-explanations-pspm/data/...`), que ya no existe. El archivo es el mismo: la fecha de modificación coincide con `csv_mtime` y el hash con `retrained_states_sha256.txt`. Arreglo: comparar por nombre de archivo o por hash.
- **Documentación desactualizada (ya corregida):** la versión anterior de `README.md` tenía otro título ("How Risk, Effect and Timing Explanations Compose…") y decía `--ent-coef 0.3`. Daba ganancias de 55.3/36.2/7.8, mientras que el paper da 47.2 (BPIC2017) y 49.5 (BPIC2012). Hablaba de un pool de 500 casos, que en realidad son 7,853 en BPIC2017, y contaba 76 + 2 atributos en BPIC2017, que son 68 + 2. El README actual ya tiene las cifras correctas y una sección "Known issues on a fresh clone" con estos problemas.

## Lo que no se re-ejecutó

- El reentrenamiento de `risk_model.py`, `effect_model.py`, `build_retrained_state.py` y PPO. Se verificaron los checkpoints y los CSV ya guardados (hashes y manifests).
- `select_policy.py`, porque copiaría checkpoints dentro de `models/`. Su elección se verificó leyendo `policy_selection.json` y comparando hashes.
- Las figuras (`figures/make_*.py`) y `run_xai_suite.py` (no aparece en el paper).
