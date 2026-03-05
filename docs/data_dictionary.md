# Data Dictionary – Recommender System Dataset

## 1. Objetivo del Dataset

Este dataset contiene información sobre **usuarios, órdenes y productos comprados**, lo que permite analizar patrones de compra y construir un **sistema de recomendación de productos** basado en el historial de compras de los usuarios.

Las tablas principales representan:

- Usuarios y sus órdenes
- Productos comprados
- Categorías de productos
- Relación entre órdenes y productos

---

# Estructura del Dataset

El dataset está compuesto por las siguientes tablas:

- `orders`
- `order_products_prior`
- `order_products__train`
- `products`
- `aisles`
- `departments`

---

# Tabla: `orders`

Información sobre cada orden realizada por los usuarios.

| Campo | Tipo | Descripción | Unidad |
|------|------|-------------|-------
| order_id | integer | Identificador único de la orden | ID |
| user_id | integer | Identificador único del usuario | ID |
| eval_set | string | Indica a qué conjunto pertenece la orden (`prior`, `train`, `test`) | categoría | 
| order_number | integer | Número de orden del usuario en orden cronológico | conteo |
| order_dow | integer | Día de la semana en que se realizó la orden | 0–6 |
| order_hour_of_day | integer | Hora del día en que se realizó la orden | 0–23 |
| days_since_prior_order | float | Días desde la orden anterior del usuario | días |

### Reglas del dataset

- `order_number = 1` indica la **primera orden del usuario**
- `days_since_prior_order` puede ser **NULL únicamente en la primera orden**

---

# Tabla: `order_products_prior`

Contiene los productos comprados en órdenes históricas.

| Campo | Tipo | Descripción | Unidad | Nulos | Ejemplo |
|------|------|-------------|-------|-------|--------|
| order_id | integer | Identificador de la orden | ID | No | 2 |
| product_id | integer | Identificador del producto | ID | No | 33120 |
| add_to_cart_order | integer | Orden en la que el producto fue agregado al carrito | posición | No | 1 |
| reordered | integer | Indica si el producto fue comprado previamente por el usuario | 0 / 1 | No | 1 |

### Reglas del dataset

- `reordered = 1` indica que el usuario **ya compró ese producto anteriormente**
- `add_to_cart_order` permite analizar **prioridad dentro del carrito**

---

# Tabla: `order_products__train`

Contiene los productos de las órdenes utilizadas para entrenar el modelo de recomendación.

| Campo | Tipo | Descripción | Unidad | 
|------|------|-------------|-------|
| order_id | integer | Identificador de la orden | ID |
| product_id | integer | Identificador del producto | ID |
| add_to_cart_order | integer | Posición del producto dentro del carrito | posición |
| reordered | integer | Variable objetivo que indica recompra | 0 / 1 | 

### Uso en el modelo

Esta tabla se utiliza para entrenar el modelo que predice:

**Qué productos volverá a comprar un usuario en su próxima orden.**

---

# Tabla: `products`

Información de cada producto.

| Campo | Tipo | Descripción | Unidad |
|------|------|-------------|-------|
| product_id | integer | Identificador único del producto | ID | 
| product_name | string | Nombre del producto | texto | 
| aisle_id | integer | Identificador del pasillo | ID | 
| department_id | integer | Identificador del departamento | ID | 

---

# Tabla: `aisles`

Categorías de pasillos dentro del supermercado.

| Campo | Tipo | Descripción | Unidad | 
|------|------|-------------|-------|-------|--------|
| aisle_id | integer | Identificador del pasillo | ID |
| aisle | string | Nombre del pasillo | texto |

---

# Tabla: `departments`

Categorías generales de productos.

| Campo | Tipo | Descripción | Unidad | 
|------|------|-------------|-------|
| department_id | integer | Identificador del departamento | ID |
| department | string | Nombre del departamento | texto |

---

# Relaciones entre Tablas

orders.order_id → order_products.order_id
products.product_id → order_products.product_id
products.aisle_id → aisles.aisle_id
products.department_id → departments.department_id

# Reglas de Calidad de Datos

1. `order_id` debe ser **único en la tabla orders**
2. `product_id` debe existir en la tabla **products**
3. `aisle_id` debe existir en la tabla **aisles**
4. `department_id` debe existir en la tabla **departments**
5. `days_since_prior_order` puede ser **NULL únicamente en la primera orden del usuario**