# insight-commerce-recsys
Sistema de recomendación de próxima compra - Proyecto Final Data Science 

## Git Workflow

### Ramas

```
main        →  producción (siempre estable)
develop     →  integración
feature/*   →  desarrollo de funcionalidades
```

### Flujo

```
feature/* → develop → main
```

1. Crear rama desde `develop`: `git checkout -b feature/nombre`
2. Abrir Pull Request hacia `develop`
3. Requiere **al menos 1 aprobación** para hacer merge
4. `develop` → `main` solo mediante PR aprobado

## Reglas

| Rama | Merge directo | Aprobaciones |
|------|:---:|:---:|
| `main` | ❌ | 1 |
| `develop` | ❌ | 1 |
| `feature/*` | ✅ | — |
