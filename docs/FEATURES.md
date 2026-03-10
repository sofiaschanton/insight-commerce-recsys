# Feature Matrix — Documentación

**Proyecto:** Next Basket Recommendation System
**Dataset:** Instacart Market Basket Analysis
**Base de datos:** NeonDB (PostgreSQL serverless)

---

## Flujo de generación

```
NeonDB
  ↓  data_loader_from_neon.py   → carga fact_order_products, dim_user, dim_product
  ↓  feature_engineering.py     → construye feature_matrix.parquet  (pares u×p, 26 cols)
  ↓  preprocessing.py           → split 80/20, imputa, clipea outliers, escala
```

El dataset se divide en dos conjuntos de órdenes por usuario:

- **prior** — todas las órdenes históricas menos la última. Fuente de todas las features.
- **train** — última orden real del usuario. Se usa exclusivamente para construir el label.

---

## Filtros aplicados

| Parámetro | Valor | Motivo |
|---|---|---|
| `MIN_USER_ORDERS` | 5 | Usuarios con menos órdenes tienen historial insuficiente |
| `MIN_PRODUCT_ORDERS` | 50 | Productos con menos compras tienen tasas de reorden ruidosas |

---

## Features — 26 columnas

### Identificadores

| Columna | Tipo | Descripción |
|---|---|---|
| `user_key` | int32 | ID único del usuario |
| `product_key` | int32 | ID único del producto |

### Features de Usuario (6)

| Columna | Tipo | Fórmula | Descripción |
|---|---|---|---|
| `user_total_orders` | int16 | `COUNT(DISTINCT order_key)` en prior | Total de órdenes históricas del usuario |
| `user_avg_basket_size` | float32 | `AVG(productos por orden)` en prior | Tamaño promedio del carrito |
| `user_days_since_last_order` | float32 | `days_since_prior_order` de la última orden prior | Días desde la última compra |
| `user_reorder_ratio` | float32 | `MEAN(reordered)` por usuario en prior | Proporción de productos repetidos — perfil de hábito global |
| `user_distinct_products` | int32 | `COUNT(DISTINCT product_key)` en prior | Variedad de productos — alto=explorador, bajo=rutinario |
| `user_segment_code` | int8 | Bins sobre `user_total_orders` | Segmento: 1=esporádico(1-5) 2=ocasional(6-10) 3=regular(11-20) 4=frecuente(21-40) 5=power user(41+) |

> `user_age` excluida — los datos de `dim_user` son sintéticos.

### Features de Producto (6)

| Columna | Tipo | Fórmula | Descripción |
|---|---|---|---|
| `product_total_purchases` | int32 | `COUNT(order_key)` en prior | Volumen total de compras del producto |
| `product_reorder_rate` | float32 | `MEAN(reordered)` por producto en prior | Tasa de reorden global del producto |
| `product_avg_add_to_cart` | float32 | `MEAN(add_to_cart_order)` en prior | Posición promedio en el carrito — valores cercanos a 1 indican producto de hábito |
| `product_unique_users` | int32 | `COUNT(DISTINCT user_key)` en prior | Popularidad real (usuarios únicos) vs. volumen bruto |
| `p_department_reorder_rate` | float32 | `MEAN(reordered)` del departamento en prior | Tasa de reorden promedio del departamento del producto |
| `p_aisle_reorder_rate` | float32 | `MEAN(reordered)` del aisle en prior | Tasa de reorden promedio del aisle — más granular que departamento |

### Features de Interacción u×p (10)

| Columna | Tipo | Fórmula | Descripción |
|---|---|---|---|
| `up_times_purchased` | int16 | `COUNT(order_key)` del par en prior | Cuántas veces compró este usuario este producto |
| `up_reorder_rate` | float32 | `up_times_purchased / user_total_orders` | Proporción de órdenes del usuario que incluyen este producto |
| `up_orders_since_last_purchase` | int16 | `user_total_orders - up_last_order_number` | Órdenes transcurridas desde la última compra del par |
| `up_first_order_number` | int16 | `MIN(order_number)` del par en prior | En qué orden el usuario compró este producto por primera vez |
| `up_last_order_number` | int16 | `MAX(order_number)` del par en prior | En qué orden el usuario compró este producto por última vez |
| `up_avg_add_to_cart_order` | float32 | `MEAN(add_to_cart_order)` del par en prior | Posición promedio en el carrito para este par específico |
| `up_days_since_last` | float32 | Suma de `days_since_prior_order` de las órdenes posteriores a `up_last_order_number` | Días transcurridos desde la última compra de este producto |
| `up_avg_days_between_orders` | float32 | `SUM(días del par) / (up_times_purchased - 1)` | Ciclo promedio de recompra del par. **NaN cuando `up_times_purchased == 1`** |
| `up_delta_days` | float32 | `up_days_since_last - up_avg_days_between_orders` | Días de retraso o adelanto respecto al ciclo esperado. Positivo = ya tocó recomprar. **NaN donde `up_avg_days_between_orders` es NaN** |
| `u_favorite_department` | int8 | Departamento con más compras del usuario en prior | Departamento favorito del usuario (codificado como int) |
| `u_favorite_aisle` | int16 | Aisle con más compras del usuario en prior | Aisle favorito del usuario (~134 aisles) |

### Label (1)

| Columna | Tipo | Valores | Construcción |
|---|---|---|---|
| `label` | int8 | `1` = reordenó · `0` = no reordenó | Left join contra pares de `train`. Ausencia en train → 0. |

---

## Tratamiento de nulos

No todos los nulos se tratan igual. El criterio es si el `0` tiene sentido semántico.

**Imputados con `0` en feature_engineering.py:**

| Columna | Por qué `0` es correcto |
|---|---|
| `up_days_since_last` | `0` = producto comprado en la última orden prior, sin órdenes posteriores |
| `label` | `0` = par no apareció en train = no reordenó |
| `user_segment_code` | `0` = caso borde fuera de bins |

**Se dejan como NaN — imputados en preprocessing.py:**

| Columna | Por qué NO `0` | Estrategia |
|---|---|---|
| `up_avg_days_between_orders` | `0` significaría ciclo de recompra instantáneo — ocurre cuando `up_times_purchased == 1` | `SimpleImputer(strategy='median')` con mediana de train |
| `up_delta_days` | Hereda NaN de `up_avg_days_between_orders`. Se recalcula post-imputación para mantener consistencia aritmética. | Recalculado: `up_days_since_last - up_avg_days_between_orders` |

---

## Codificación de categóricas

| Columna | Fuente en NeonDB | Cardinalidad |
|---|---|---|
| `user_segment_code` | Calculado sobre `user_total_orders` | 5 segmentos |
| `u_favorite_department` | Generado desde `department_name` | 21 departamentos |
| `u_favorite_aisle` | Generado desde `aisle_name` | 134 aisles |

Para LightGBM se pasan como enteros directamente (`categorical_encoding='passthrough'`).
Para otros modelos usar `categorical_encoding='onehot'` en `build_preprocessing_pipeline()`.
