import random
import copy

# --- 1. CONFIGURACIÓN DEL PROYECTO ---

DEADLINE_PROYECTO = 50  # El proyecto DEBE terminar antes de la hora 50

# Definición de Recursos
RECURSOS_RENOVABLES = { # Se recuperan al terminar la tarea
    'mechanical': 40,
    'welder': 20,
    'crane': 2
}

RECURSOS_CONSUMIBLES = { # Se gastan (Inventario inicial)
    'steel_plates': 100, # Unidades
    'cement_bags': 200   # Sacos
}

COSTOS = {
    'hora_extra': 500,     # Costo por hora si nos pasamos del deadline (Penalidad suave)
    'material_steel': 50,  # Costo por unidad
    'material_cement': 20, # Costo por unidad
    'fijo_tarea': 100      # Costo administrativo por tarea
}

# --- 2. GENERADOR DE DATOS CON IMPORTANCIA Y MATERIALES ---
def generar_datos_complejos(num_tareas=25):
    datos = {}
    # Inicio
    datos[0] = {'nombre': 'Inicio', 'dur': 0, 'imp': 10, 'ren': {}, 'con': {}, 'pred': []}
    
    for i in range(1, num_tareas + 1):
        # Importancia (1: Baja, 10: Crítica/Alta Seguridad)
        importancia = random.choices(range(1, 11), weights=[1,1,2,2,3,3,4,2,1,1])[0]
        
        # Consumo de materiales (Solo algunas tareas usan materiales)
        consumibles = {}
        if random.random() < 0.4: consumibles['steel_plates'] = random.randint(1, 5)
        if random.random() < 0.3: consumibles['cement_bags'] = random.randint(5, 15)
        
        # Recursos renovables
        renovables = {}
        if random.random() < 0.8: renovables['mechanical'] = random.randint(2, 8)
        if random.random() < 0.5: renovables['welder'] = random.randint(2, 5)
        if random.random() < 0.15: renovables['crane'] = 1
        
        datos[i] = {
            'nombre': f"Actividad {i}",
            'dur': random.randint(2, 8),
            'imp': importancia,      # Factor de decisión
            'ren': renovables,       # Resources (Renewable)
            'con': consumibles,      # Resources (Consumable/Materials)
            'pred': []
        }
        
        # Precedencias
        if i > 1:
            candidatos = list(range(max(0, i-5), i))
            datos[i]['pred'] = random.sample(candidatos, random.randint(1, min(len(candidatos), 2)))
        else:
            datos[i]['pred'] = [0]
            
    # Fin
    datos[num_tareas + 1] = {'nombre': 'Fin', 'dur': 0, 'imp': 10, 'ren': {}, 'con': {}, 
                             'pred': list(range(1, num_tareas + 1))}
    return datos

# --- 3. EL DECODIFICADOR INTELIGENTE (SGS) ---
def evaluar_cronograma(cromosoma, datos, limite_tiempo):
    """
    Convierte los genes en un cronograma, respetando materiales y deadline.
    Retorna: (Costo Total, Makespan, Es_Valido)
    """
    
    # COPIA DE INVENTARIO (Porque se va a gastar)
    inventario_actual = RECURSOS_CONSUMIBLES.copy()
    
    # 1. Calcular Prioridad Combinada: GENÉTICA + IMPORTANCIA
    # La "astucia" del algoritmo: Multiplicamos el gen aleatorio por la importancia de la tarea.
    # Esto fuerza a que las tareas críticas tengan naturalmente más "fuerza" para ser elegidas antes.
    lista_prioridad = []
    ids = sorted(datos.keys())
    for idx, tid in enumerate(ids):
        gen_val = cromosoma[idx]
        importancia = datos[tid].get('imp', 1)
        # Fórmula de Prioridad Heurística
        score = gen_val * (importancia ** 0.5) 
        lista_prioridad.append((tid, score))
    
    orden_exec = sorted(lista_prioridad, key=lambda x: x[1], reverse=True)
    
    # Variables de simulación
    tiempos_fin = {t: 0 for t in datos}
    uso_renovables = {} # {tiempo: {recurso: cantidad}}
    terminados = set()
    costo_materiales = 0
    
    n = len(datos)
    while len(terminados) < n:
        candidato = -1
        
        for tid, _ in orden_exec:
            if tid in terminados: continue
            
            # A. Verificar Predecesores
            preds = datos[tid]['pred']
            if not preds or all(p in terminados for p in preds):
                
                # B. Verificar Materiales (Consumibles)
                materiales_ok = True
                req_mat = datos[tid]['con']
                for mat, cant in req_mat.items():
                    if inventario_actual.get(mat, 0) < cant:
                        materiales_ok = False # ¡No hay material! (Solución inválida por ahora)
                        # En un escenario real, esto implicaría pedir material y esperar.
                        # Aquí penalizaremos la solución si no alcanzan los materiales.
                        return float('inf'), float('inf'), False 
                
                if materiales_ok:
                    candidato = tid
                    break
        
        if candidato == -1: break # Bloqueo
        
        # Calcular tiempos
        inicio_min = 0
        if datos[candidato]['pred']:
            inicio_min = max(tiempos_fin[p] for p in datos[candidato]['pred'])
            
        dur = datos[candidato]['dur']
        req_ren = datos[candidato]['ren']
        
        # Buscar espacio en recursos renovables (Mecánicos, grúas)
        encontrado = False
        t = inicio_min
        
        # Límite de seguridad para evitar bucles infinitos si no hay solución
        while not encontrado and t < limite_tiempo * 2: 
            viable = True
            if req_ren:
                for k in range(t, t + dur):
                    for r, cant in req_ren.items():
                        if uso_renovables.get(k, {}).get(r, 0) + cant > RECURSOS_RENOVABLES[r]:
                            viable = False; break
                    if not viable: break
            
            if viable:
                # Agendar
                for k in range(t, t + dur):
                    if k not in uso_renovables: uso_renovables[k] = {}
                    for r, cant in req_ren.items():
                        uso_renovables[k][r] = uso_renovables[k].get(r, 0) + cant
                
                # Consumir Materiales
                req_mat = datos[candidato]['con']
                for mat, cant in req_mat.items():
                    inventario_actual[mat] -= cant
                    costo_materiales += cant * COSTOS.get(f'material_{mat.split("_")[1]}', 10)

                tiempos_fin[candidato] = t + dur
                terminados.add(candidato)
                encontrado = True
            else:
                t += 1
        
        if not encontrado: return float('inf'), float('inf'), False # Time out

    makespan = max(tiempos_fin.values())
    
    # --- CÁLCULO DE COSTO Y PENALIZACIONES ---
    costo_total = costo_materiales + (n * COSTOS['fijo_tarea'])
    
    # Penalización por DEADLINE
    violacion_tiempo = max(0, makespan - limite_tiempo)
    
    if violacion_tiempo > 0:
        # Penalización Cuadrática: cuanto más te pasas, MÁS duele.
        penalidad = (violacion_tiempo ** 2) * 1000 
        costo_total += penalidad
    
    return costo_total, makespan, (violacion_tiempo == 0)

# --- 4. ALGORITMO GENÉTICO ---
class Individuo:
    def __init__(self, num_genes):
        self.genes = [random.random() for _ in range(num_genes)]
        self.costo = float('inf')
        self.makespan = 0
        self.valido = False

def algoritmo_genetico_restriccion(datos, gens=50, pop_size=50):
    num_genes = len(datos)
    poblacion = [Individuo(num_genes) for _ in range(pop_size)]
    
    # Evaluar inicial
    best_global = None
    
    print(f"Optimizando... Objetivo: Terminar antes de T={DEADLINE_PROYECTO}")
    
    for g in range(gens):
        for ind in poblacion:
            c, m, v = evaluar_cronograma(ind.genes, datos, DEADLINE_PROYECTO)
            ind.costo = c
            ind.makespan = m
            ind.valido = v
            
        # Ordenar: Menor Costo es mejor
        poblacion.sort(key=lambda x: x.costo)
        
        if best_global is None or poblacion[0].costo < best_global.costo:
            best_global = copy.deepcopy(poblacion[0])
            estado = "VALIDO" if best_global.valido else "INVALIDO"
            print(f"Gen {g}: Costo ${best_global.costo:.2f} | Tiempo: {best_global.makespan}h ({estado})")
            
        # Selección y Reproducción
        nueva_poblacion = poblacion[:5] # Elitismo (Top 5)
        
        while len(nueva_poblacion) < pop_size:
            # Torneo
            parents = random.sample(poblacion[:20], 2) # Elegir de los mejores 20
            p1, p2 = parents[0], parents[1]
            
            child = Individuo(num_genes)
            # Crossover Uniforme
            for i in range(num_genes):
                child.genes[i] = p1.genes[i] if random.random() < 0.5 else p2.genes[i]
            
            # Mutación
            if random.random() < 0.1:
                idx = random.randint(0, num_genes-1)
                child.genes[idx] = random.random()
                
            nueva_poblacion.append(child)
            
        poblacion = nueva_poblacion

    return best_global

# --- 5. EJECUCIÓN ---
if __name__ == "__main__":
    DATA = generar_datos_complejos(num_tareas=30)
    
    solucion = algoritmo_genetico_restriccion(DATA, gens=40, pop_size=60)
    
    print("\n" + "="*60)
    print("MEJOR PLANIFICACIÓN ENCONTRADA")
    print(f"Costo Final: ${solucion.costo:.2f}")
    print(f"Duración: {solucion.makespan} horas (Límite: {DEADLINE_PROYECTO})")
    print("="*60)
    
    # Re-evaluar para obtener el detalle del cronograma para imprimir
    # (Nota: En un código prod, guardaríamos el cronograma dentro del objeto Individuo)
    # Aquí hacemos una simulación rápida solo para imprimir
    print(f"{'ID':<3} | {'ACTIVIDAD':<20} | {'IMP':<3} | {'MAT (Acero/Cem)'}")
    print("-" * 60)
    
    # Usamos la misma logica de ordenamiento para mostrar prioridad
    lista_prio = []
    ids = sorted(DATA.keys())
    for idx, tid in enumerate(ids):
        score = solucion.genes[idx] * (DATA[tid]['imp'] ** 0.5)
        lista_prio.append((tid, score))
    
    orden_final = sorted(lista_prio, key=lambda x: x[1], reverse=True)
    
    count = 0
    for tid, score in orden_final:
        if tid == 0 or tid == len(DATA): continue
        if count > 10: break # Mostrar solo top 10
        d = DATA[tid]
        mat_str = f"{d['con'].get('steel_plates',0)} / {d['con'].get('cement_bags',0)}"
        print(f"{tid:<3} | {d['nombre']:<20} | {d['imp']:<3} | {mat_str}")
        count += 1
    print("... (lista truncada)")